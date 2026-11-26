#!/usr/bin/env python3
"""Run local GAIA tasks with Claude Code in an isolated Linux filesystem.

Usage (from /home/claude-GAIA, with pyarrow installed in your Python environment):
    python -m pip install pyarrow
    sudo -v
    python run_gaia_claude --limit 1 --sudo-dumpcap
    python run_gaia_claude --split validation --level 1 --sudo-dumpcap
    python run_gaia_claude --split test --limit 1 --sudo-dumpcap
    python run_gaia_claude --list-tasks

Requires: Linux, bubblewrap (bwrap), the native Claude Code executable, dumpcap,
and pyarrow. Claude authentication is taken from the current environment or
the allowlisted env entries in ~/.claude/settings.json. OAuth credentials are
copied only if API credentials are absent. No other host Claude config is used.

Per task, retain exactly the Notion runner's five artifacts:
    raw_traffic.pcapng, claude_timeline.jsonl, claude_output.json,
    task_prompt.txt, result.json.
At experiment root retain only experiment.json and summary.csv (no MCP server
is used). Temporary attachments, settings, and native CLI files are deleted.

Only Question and the explicitly referenced attachment reach the solver.
Final answer stays in the supervisor for grading; Annotator Metadata is never
read. Metadata is loaded locally, without Hugging Face downloads. OS isolation
is mandatory: there is deliberately no unsandboxed fallback. This prevents
local answer leakage; it cannot remove information already learned by a model
or available on the public internet. The selected question/attachment must be
sent to the configured model service to solve it. Keep result artifacts private.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import selectors
import shutil
import signal
import string
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

CLAUDE_MAX_TURNS = 50
CLAUDE_TIMEOUT_S = 3600
RESULTS_ROOT_NAME = 'results_claude_gaia'
TOOLS = 'Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch'
AUTH_ENV_KEYS = {
    'ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL',
    'ANTHROPIC_MODEL', 'ANTHROPIC_DEFAULT_HAIKU_MODEL',
    'ANTHROPIC_DEFAULT_SONNET_MODEL', 'ANTHROPIC_DEFAULT_OPUS_MODEL',
    'ANTHROPIC_DEFAULT_FABLE_MODEL', 'CLAUDE_CODE_DISABLE_1M_CONTEXT',
    'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
    'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy',
}
WITHHELD_ANSWERS = {'', '?', 'not available', 'unknown', 'none', 'n/a', 'null'}


@dataclass(frozen=True)
class GaiaTask:
    task_id: str
    question: str = field(repr=False)
    level: int
    split: str
    attachment: Path | None = field(repr=False)
    expected_answer: str | None = field(repr=False)

    @property
    def name(self) -> str:
        return f'{self.split}__{self.task_id}'


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def load_tasks(dataset_root: Path, year: str, split: str) -> list[GaiaTask]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError('Parquet support requires: python -m pip install pyarrow') from exc
    split_dir = (dataset_root / year / split).resolve(strict=True)
    if not split_dir.is_relative_to(dataset_root):
        raise ValueError('Split directory must stay within the dataset root')
    metadata = split_dir / 'metadata.parquet'
    # Do not concatenate metadata.levelN.parquet: those repeat the same tasks.
    columns = ['task_id', 'Question', 'Level', 'Final answer', 'file_name', 'file_path']
    rows = pq.read_table(metadata, columns=columns).to_pylist()
    tasks = []
    seen = set()
    for row in rows:
        task_id = str(row['task_id']).strip()
        # The original test split also contains a non-UUID ID (0-0-0-0-0).
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', task_id):
            raise ValueError('Task ID must be a safe filename component')
        if task_id in seen:
            raise ValueError(f'Duplicate task ID: {task_id}')
        seen.add(task_id)
        question = row['Question']
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f'Missing question for task {task_id}')
        level = int(row['Level'])
        if level not in (1, 2, 3):
            raise ValueError(f'Invalid difficulty for task {task_id}')
        attachment = None
        file_name = str(row['file_name'] or '').strip()
        file_path = str(row['file_path'] or '').strip()
        if file_name:
            if Path(file_name).name != file_name or ':' in file_name:
                raise ValueError(f'Invalid attachment filename for task {task_id}')
            declared = split_dir / file_name
            if declared.is_symlink():
                raise ValueError(f'Symlink attachment is forbidden: task {task_id}')
            attachment = declared.resolve(strict=True)
            if not attachment.is_relative_to(split_dir) or not attachment.is_file():
                raise ValueError(f'Attachment outside task split: {task_id}')
            if attachment.name.startswith('metadata') or attachment.suffix.lower() in {'.parquet', '.jsonl'}:
                raise ValueError(f'Dataset metadata cannot be an attachment: {task_id}')
            if file_path and (dataset_root / file_path).resolve() != attachment:
                raise ValueError(f'Conflicting attachment paths for task {task_id}')
        elif file_path:
            raise ValueError(f'Attachment path without filename: task {task_id}')
        answer = str(row['Final answer'] or '').strip()
        expected = None if answer.lower() in WITHHELD_ANSWERS else answer
        tasks.append(GaiaTask(task_id, question, level, split, attachment, expected))
    return tasks


def stage_task(task: GaiaTask, work: Path) -> str:
    attachment_name = None
    if task.attachment is not None:
        attachment_name = 'attachment' + task.attachment.suffix.lower()
        shutil.copyfile(task.attachment, work / attachment_name)
    prompt = (
        'Solve the following question using the available tools.\n'
        'Use evidence and calculations where appropriate. Do not search for benchmark '
        'answer keys or use evaluation metadata.\n'
        'Your last line must be FINAL ANSWER: followed by only the requested answer, '
        'with no explanation on that line. Follow the question\'s requested formatting.\n\n'
        'Question:\n' + task.question.strip() + '\n'
    )
    if attachment_name:
        prompt += f'\nThe supplied attachment is available at /work/{attachment_name}.\n'
    return prompt


def solver_environment() -> dict[str, str]:
    """Export only model authentication/routing; never copy the runner's env."""
    values = {}
    settings = Path.home() / '.claude' / 'settings.json'
    if settings.is_file():
        parsed = json.loads(settings.read_text(encoding='utf-8'))
        configured = parsed.get('env', {})
        if isinstance(configured, dict):
            values.update({k: str(v) for k, v in configured.items() if k in AUTH_ENV_KEYS and isinstance(v, (str, int, bool))})
    values.update({k: os.environ[k] for k in AUTH_ENV_KEYS if k in os.environ})
    values.update({
        'HOME': '/home/runner', 'CLAUDE_CONFIG_DIR': '/home/runner/.claude',
        'XDG_CONFIG_HOME': '/home/runner/.config',
        'PATH': '/opt/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin',
        'TMPDIR': '/tmp', 'LANG': 'C.UTF-8', 'IS_SANDBOX': '1',
        'CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC': '1',
        'CLAUDE_CODE_DISABLE_AUTO_MEMORY': '1', 'NO_COLOR': '1',
    })
    return values


def prepare_runtime(runtime: Path, env: dict[str, str]) -> None:
    home = runtime / 'home'
    (home / '.claude').mkdir(parents=True)
    (runtime / 'work').mkdir()
    write_json(home / '.claude.json', {'hasCompletedOnboarding': True})
    write_json(home / '.claude' / 'settings.json', {})
    write_json(home / 'mcp.json', {'mcpServers': {}})
    # Do not copy host settings, hooks, project configs, sessions, or memories.
    if not (env.get('ANTHROPIC_API_KEY') or env.get('ANTHROPIC_AUTH_TOKEN')):
        credentials = Path.home() / '.claude' / '.credentials.json'
        if credentials.is_file():
            oauth = json.loads(credentials.read_text(encoding='utf-8')).get('claudeAiOauth')
            if isinstance(oauth, dict):
                write_json(home / '.claude' / '.credentials.json', {'claudeAiOauth': oauth})
    (runtime / 'passwd').write_text('root:x:0:0:Runner:/home/runner:/bin/bash\n', encoding='utf-8')
    (runtime / 'group').write_text('root:x:0:\n', encoding='utf-8')


def sandbox_command(runtime: Path, executable: Path, command: list[str]) -> list[str]:
    """Build an empty-root mount namespace with only explicitly allowed files."""
    bwrap = shutil.which('bwrap')
    if bwrap is None:
        raise RuntimeError('bubblewrap (bwrap) is required; refusing to expose the dataset')
    cmd = [bwrap, '--unshare-all', '--share-net', '--die-with-parent',
           '--new-session', '--cap-drop', 'ALL', '--ro-bind', '/usr', '/usr']
    for directory in ('/bin', '/sbin', '/lib', '/lib64'):
        if Path(directory).exists():
            cmd += ['--ro-bind', directory, directory]
    for path in ('/etc/resolv.conf', '/etc/hosts', '/etc/nsswitch.conf',
                 '/etc/ssl/certs', '/etc/ca-certificates', '/etc/os-release'):
        if Path(path).exists():
            cmd += ['--ro-bind', path, path]
    cmd += [
        '--ro-bind', str(runtime / 'passwd'), '/etc/passwd',
        '--ro-bind', str(runtime / 'group'), '/etc/group',
        '--ro-bind', str(executable), '/opt/bin/claude',
        '--bind', str(runtime / 'home'), '/home/runner',
        '--bind', str(runtime / 'work'), '/work',
        '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/tmp',
        '--chdir', '/work', '--',
    ]
    return cmd + command


def check_isolation(dataset_root: Path, executable: Path, env: dict[str, str]) -> None:
    if os.name != 'posix' or not Path('/proc').exists():
        raise RuntimeError('This runner requires Linux filesystem isolation')
    for directory in ('/usr', '/bin', '/sbin', '/lib', '/lib64'):
        if dataset_root.is_relative_to(Path(directory).resolve()):
            raise ValueError('Dataset root cannot be under a mounted system directory')
    if not executable.is_file():
        raise RuntimeError('Claude Code executable not found')
    with executable.open('rb') as binary:
        native = binary.read(4) == b'\x7fELF'
    if not native:
        raise RuntimeError('Use a native Linux Claude Code executable, not a shell/Node wrapper')
    with tempfile.TemporaryDirectory(prefix='gaia-isolation-check-', dir='/tmp') as temp:
        runtime = Path(temp)
        prepare_runtime(runtime, env)
        # Run a real namespaced filesystem check before any paid model calls.
        probe = [
            '/usr/bin/python3', '-c',
            'import os; assert not os.path.exists(os.environ["CHECK_DATASET"]); '
            'assert not os.path.exists("/home/claude-MCP"); '
            'assert not os.path.exists("/root/.claude"); '
            'assert os.getcwd()=="/work"; print("isolation-ok")',
        ]
        checked = subprocess.run(sandbox_command(runtime, executable, probe),
            env={**env, 'CHECK_DATASET': str(dataset_root)},
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=15, check=False)
        if checked.returncode != 0 or checked.stdout.strip() != 'isolation-ok':
            raise RuntimeError('Filesystem isolation failed; refusing to run Claude: ' + checked.stderr[-500:].strip())


def extract_answer(output: dict) -> str | None:
    text = output.get('result')
    if output.get('type') == 'result_unavailable' or not isinstance(text, str):
        return None
    matches = re.findall(r'^FINAL ANSWER:\s*(.+)$', text, flags=re.MULTILINE)
    if matches:
        return matches[-1].strip()
    # A short, bare final answer remains compatible with older CLI/model outputs.
    stripped = text.strip()
    return stripped if stripped and '\n' not in stripped else None


def normalize_number(value: str) -> float | None:
    try:
        return float(value.replace(',', '').replace('$', '').replace('%', '').strip())
    except ValueError:
        return None


def normalize_text(value: str, remove_punctuation: bool = True) -> str:
    value = ''.join(value.split()).lower()
    return value.translate(str.maketrans('', '', string.punctuation)) if remove_punctuation else value


def score_answer(predicted: str | None, expected: str | None) -> bool | None:
    """GAIA's numeric/list/text comparison, without an LLM judge or gold output."""
    if expected is None:
        return None
    if predicted is None:
        return False
    try:
        target_number = float(expected)
    except ValueError:
        target_number = None
    if target_number is not None:
        return normalize_number(predicted) == target_number
    if ',' in expected or ';' in expected:
        targets = re.split('[,;]', expected)
        answers = re.split('[,;]', predicted)
        if len(targets) != len(answers):
            return False
        for answer, target in zip(answers, targets):
            try:
                number = float(target)
            except ValueError:
                if normalize_text(answer, False) != normalize_text(target, False):
                    return False
            else:
                if normalize_number(answer) != number:
                    return False
        return True
    return normalize_text(predicted) == normalize_text(expected)


def stop_process(proc, sig=signal.SIGTERM, timeout_s=10.0):
    if proc is None or proc.poll() is not None:
        return

    try:
        # proc 是用 start_new_session=True 启动的，
        # 所以 proc.pid 同时也是新进程组的 PGID。
        os.killpg(proc.pid, sig)
        proc.wait(timeout=timeout_s)

    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=3)

    except ProcessLookupError:
        pass


def start_capture(pcap_path: Path, use_sudo: bool) -> subprocess.Popen:
    """Capture packets without creating capture.log."""
    base = ["dumpcap", "-q", "-i", "any", "-w", str(pcap_path)]
    cmd = (["sudo", "-n"] + base) if use_sudo else base
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(0.5)
    if proc.poll() is not None:
        raise RuntimeError(f"dumpcap exited early (code {proc.returncode})")
    return proc


def stop_capture(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            stop_process(proc, signal.SIGTERM, 5)


def timestamp_from_ns(epoch_ns: int) -> str:
    """Wall-clock UTC, compatible with absolute timestamps from PCAP."""
    return datetime.fromtimestamp(epoch_ns / 1_000_000_000, timezone.utc).isoformat(
        timespec="microseconds"
    )


class ClaudeTimelineRecorder:
    """Write just one JSONL timeline; no duplicate stream/history/tool files.

    event time: when this Python runner receives a complete Claude CLI JSONL
    record. These are *observations*, not exact wire-level send times.

    In particular, stream_event/message_start means MODEL RESPONSE START, not
    LLM REQUEST SENT. For actual LLM request start and MCP tool dispatch on
    the wire, use PCAP (LLM HTTP and the local /mcp Streamable HTTP endpoint).
    """

    def __init__(self, task_dir: Path, session_id: str):
        self.task_dir = task_dir
        self.session_id = session_id
        self.event_count = 0
        self.unparsed_lines = 0
        self.tool_calls: dict[str, dict] = {}
        self.final_result: dict | None = None
        self.model_response_count = 0
        self._started_messages: set[tuple[str, str]] = set()
        self._current_message: dict[str, str | None] = {}
        self._timeline = (task_dir / "claude_timeline.jsonl").open(
            "w", encoding="utf-8", buffering=1
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._timeline.close()

    def emit(self, event: str, epoch_ns: int | None = None, **details) -> None:
        if epoch_ns is None:
            epoch_ns = time.time_ns()
        row = {
            "timestamp_utc": timestamp_from_ns(epoch_ns),
            "unix_ns": epoch_ns,
            "event": event,
            "session_id": self.session_id,
            **details,
        }
        self._timeline.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def _call(self, tool_use_id: str) -> dict:
        return self.tool_calls.setdefault(tool_use_id, {
            "name": None,
            "generated": False,
            "dispatched": False,
            "result_seen": False,
            "dispatch_ns": None,
        })

    def record_line(self, raw: bytes) -> None:
        """Timestamp and distill one Claude --output-format stream-json line."""
        if not raw.strip():
            return
        observed_ns = time.time_ns()
        try:
            outer = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            self.unparsed_lines += 1
            return
        if not isinstance(outer, dict):
            return
        self.event_count += 1
        kind = outer.get("type")
        parent = str(outer.get("parent_tool_use_id") or "root")

        if kind == "stream_event":
            inner = outer.get("event") or {}
            if not isinstance(inner, dict):
                return
            event_kind = inner.get("type")

            if event_kind == "message_start":
                message = inner.get("message") or {}
                msg_id = message.get("id") if isinstance(message, dict) else None
                self._current_message[parent] = msg_id
                key = (parent, str(msg_id))
                if key not in self._started_messages:
                    self._started_messages.add(key)
                    self.model_response_count += 1
                    self.emit("llm_response_start", observed_ns,
                              message_id=msg_id,
                              parent_tool_use_id=outer.get("parent_tool_use_id"))

            elif event_kind == "message_stop":
                self.emit("llm_response_end", observed_ns,
                          message_id=self._current_message.get(parent),
                          parent_tool_use_id=outer.get("parent_tool_use_id"))
                self._current_message.pop(parent, None)

            elif event_kind == "content_block_start":
                block = inner.get("content_block") or {}
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    return
                call_id = block.get("id")
                if not call_id:
                    return
                call = self._call(str(call_id))
                call["name"] = block.get("name") or call["name"]
                if not call["generated"]:
                    call["generated"] = True
                    self.emit("tool_use_generated", observed_ns,
                              tool_use_id=str(call_id), tool_name=call["name"])
            return

        if kind == "assistant":
            message = outer.get("message") or {}
            if not isinstance(message, dict):
                return
            msg_id = message.get("id")
            key = (parent, str(msg_id))
            # Fallback for CLI versions that do not emit stream_event.
            if key not in self._started_messages:
                self._started_messages.add(key)
                self.model_response_count += 1
                self.emit("llm_response_observed", observed_ns,
                          message_id=msg_id, note="response already received; not request start")
            blocks = message.get("content") or []
            if not isinstance(blocks, list):
                return
            for block in blocks:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                call_id = block.get("id")
                if not call_id:
                    continue
                call = self._call(str(call_id))
                call["name"] = block.get("name") or call["name"]
                if not call["dispatched"]:
                    call["dispatched"] = True
                    call["dispatch_ns"] = observed_ns
                    # This is when the CLI reports the complete tool_use
                    # message; NOT verified MCP HTTP dispatch on the wire.
                    self.emit("tool_dispatch_observed", observed_ns,
                              tool_use_id=str(call_id), tool_name=call["name"])
            return

        if kind == "user":
            message = outer.get("message") or {}
            if not isinstance(message, dict):
                return
            blocks = message.get("content") or []
            if not isinstance(blocks, list):
                return
            for block in blocks:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                call_id = block.get("tool_use_id")
                if not call_id:
                    continue
                call = self._call(str(call_id))
                if call["result_seen"]:
                    continue
                call["result_seen"] = True
                elapsed = (observed_ns - call["dispatch_ns"]) / 1e9 if call["dispatch_ns"] else None
                self.emit("tool_result_observed", observed_ns,
                          tool_use_id=str(call_id), tool_name=call["name"],
                          is_error=bool(block.get("is_error", False)),
                          observed_elapsed_s=round(elapsed, 6) if elapsed is not None else None)
            return

        if kind == "result":
            self.final_result = outer

    def save_final_result(self, rc: int, timed_out: bool, stderr_excerpt: str) -> None:
        if self.final_result is not None:
            write_json(self.task_dir / "claude_output.json", self.final_result)
        else:
            write_json(self.task_dir / "claude_output.json", {
                "type": "result_unavailable",
                "session_id": self.session_id,
                "claude_exit_code": rc,
                "claude_timed_out": timed_out,
                "error": stderr_excerpt[-2000:] if rc != 0 else None,
            })


def run_claude(prompt: str, task_dir: Path, runtime: Path, executable: Path,
               solver_env: dict[str, str], model: str, effort: str, on_process_exit=None):
    """Run Claude and save only condensed timestamped events and the final result.

    on_process_exit stops PCAP before file postprocessing.
    Returns: exit_code, timed_out, error, metadata
    """
    session_id = str(uuid.uuid4())
    env = solver_env.copy()
    cmd = [
        "/opt/bin/claude", "-p", prompt,
        "--model", model, "--effort", effort,
        "--output-format", "stream-json", "--verbose",
        "--include-partial-messages", "--session-id", session_id,
        "--max-turns", str(CLAUDE_MAX_TURNS), "--no-session-persistence",
        "--strict-mcp-config", "--mcp-config", "/home/runner/mcp.json",
        "--settings", "/home/runner/.claude/settings.json",
        "--setting-sources", "", "--disable-slash-commands",
        "--tools", TOOLS, "--allowedTools", TOOLS,
        # Auto-approve tools only inside the mandatory OS sandbox.
        "--dangerously-skip-permissions",
    ]
    cmd = sandbox_command(runtime, executable, cmd)

    rc = -1
    timed_out = False
    stderr_excerpt = ""
    process_started_at = None
    process_ended_at = None
    with ClaudeTimelineRecorder(task_dir, session_id) as trace:
        # TemporaryFile is unlinked on close; no claude_stderr.log is generated.
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8", errors="replace") as stderr_file:
            proc = subprocess.Popen(
                cmd,
                cwd=str(runtime / "work"),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                start_new_session=True,
                bufsize=0,
            )
            started_ns = time.time_ns()
            process_started_at = timestamp_from_ns(started_ns)
            trace.emit("claude_process_start", started_ns)
            assert proc.stdout is not None
            buf = bytearray()
            deadline = time.monotonic() + CLAUDE_TIMEOUT_S
            exit_seen_at = None
            drain_deadline = None
            with selectors.DefaultSelector() as selector:
                selector.register(proc.stdout, selectors.EVENT_READ)
                try:
                    while selector.get_map():
                        now = time.monotonic()
                        if not timed_out and now >= deadline and proc.poll() is None:
                            timed_out = True
                            stop_process(proc, signal.SIGTERM, 5)
                            drain_deadline = time.monotonic() + 3
                        if drain_deadline is not None and now >= drain_deadline:
                            break

                        ready = selector.select(timeout=0.2)
                        for key, _ in ready:
                            chunk = os.read(key.fd, 65536)
                            if not chunk:
                                selector.unregister(key.fileobj)
                                continue
                            buf.extend(chunk)
                            while True:
                                pos = buf.find(b"\n")
                                if pos < 0:
                                    break
                                trace.record_line(bytes(buf[:pos]))
                                del buf[:pos + 1]

                        # A child can inherit stdout and keep it open after Claude
                        # exits; don't wait forever for that extra child.
                        if proc.poll() is not None:
                            if exit_seen_at is None:
                                exit_seen_at = time.monotonic()
                            elif time.monotonic() - exit_seen_at >= 3:
                                break
                except BaseException:
                    stop_process(proc, signal.SIGTERM, 5)
                    raise
                finally:
                    if buf.strip():
                        trace.record_line(bytes(buf))
                    proc.stdout.close()

            if proc.poll() is None:
                stop_process(proc, signal.SIGTERM, 5)
            rc = proc.wait()
            ended_ns = time.time_ns()
            process_ended_at = timestamp_from_ns(ended_ns)
            if timed_out:
                rc = 124
            trace.emit("claude_process_end", ended_ns, exit_code=rc,
                       timed_out=timed_out)
            # End the PCAP window immediately after the Claude process exits.
            # No expensive session export; stop PCAP before final result writing.
            if on_process_exit is not None:
                on_process_exit()
            stderr_file.seek(0)
            stderr_excerpt = stderr_file.read()[-2000:]

        # Only one timeline is retained; native/raw/duplicated logs are omitted.
        trace.save_final_result(rc, timed_out, stderr_excerpt)
        meta = {
            "claude_session_id": session_id,
            "claude_process_started_at_utc": process_started_at,
            "claude_process_ended_at_utc": process_ended_at,
            "claude_event_count": trace.event_count,
            "claude_model_response_count": trace.model_response_count,
            "claude_tool_call_count": sum(
                call["dispatched"] for call in trace.tool_calls.values()
            ),
            "claude_tool_result_count": sum(
                call["result_seen"] for call in trace.tool_calls.values()
            ),
            "claude_unparsed_stdout_lines": trace.unparsed_lines,
            "claude_timeline_path": str(task_dir / "claude_timeline.jsonl"),
        }

    if timed_out:
        error = f"Claude timed out after {CLAUDE_TIMEOUT_S}s"
    elif rc != 0:
        error = f"Claude CLI exited with code {rc}"
        if stderr_excerpt.strip():
            print(f"  [claude stderr excerpt] {stderr_excerpt[-500:].strip()}")
    else:
        error = None
    return rc, timed_out, error, meta



def main() -> int:
    global CLAUDE_MAX_TURNS, CLAUDE_TIMEOUT_S
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset-root', type=Path, default=project_root / 'DATASET' / 'GAIA')
    parser.add_argument('--year', default='2023')
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    parser.add_argument('--level', type=int, nargs='+', choices=(1, 2, 3))
    parser.add_argument('--task', help='One task ID, unique ID prefix, or split__ID')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--range', type=int, nargs=2, metavar=('START', 'END'),
                        help='Run tasks START through END (1-based, inclusive), after level filtering')
    parser.add_argument('--resume-dir', type=Path)
    parser.add_argument('--results-root', type=Path, default=project_root / RESULTS_ROOT_NAME)
    parser.add_argument('--sudo-dumpcap', action='store_true')
    parser.add_argument('--list-tasks', action='store_true', help='Print only task IDs, levels and attachment types; make no calls')
    parser.add_argument('--model', default='gpt-6.1-sol')
    parser.add_argument('--effort', choices=('low', 'medium', 'high', 'xhigh', 'max'), default='medium')
    parser.add_argument('--max-turns', type=int, default=CLAUDE_MAX_TURNS)
    parser.add_argument('--timeout', type=float, default=CLAUDE_TIMEOUT_S)
    parser.add_argument('--claude-bin', type=Path, help='Path to the native Linux Claude Code executable')
    args = parser.parse_args()
    if not re.fullmatch(r'\d{4}', args.year):
        parser.error('--year must be a four-digit year')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.range is not None:
        if args.task is not None or args.limit is not None:
            parser.error('--range cannot be combined with --task or --limit')
        if args.range[0] < 1 or args.range[1] < args.range[0]:
            parser.error('--range requires 1 <= START <= END')
    if args.max_turns < 1 or args.timeout <= 0:
        parser.error('--max-turns and --timeout must be positive')
    CLAUDE_MAX_TURNS, CLAUDE_TIMEOUT_S = args.max_turns, args.timeout
    try:
        dataset_root = args.dataset_root.resolve(strict=True)
        tasks = load_tasks(dataset_root, args.year, args.split)
        if args.level:
            tasks = [t for t in tasks if t.level in args.level]
        if args.range is not None:
            start, end = args.range
            if end > len(tasks):
                raise ValueError(f'--range END {end} exceeds task count {len(tasks)}')
            tasks = tasks[start - 1:end]
        if args.task:
            tasks = [t for t in tasks if t.task_id == args.task or t.name == args.task or t.task_id.startswith(args.task)]
            if len(tasks) != 1:
                raise ValueError('--task must identify exactly one task in the selected split/levels')
        if args.limit is not None:
            tasks = tasks[:args.limit]
        if not tasks:
            raise ValueError('No tasks match the selection')
        print(f'Tasks: {len(tasks)} (year={args.year}, split={args.split})')
        if args.list_tasks:
            for task in tasks:
                print(f'{task.task_id} level={task.level} attachment={task.attachment.suffix if task.attachment else "none"}')
            return 0
        for command in ('bwrap', 'dumpcap') + (('sudo',) if args.sudo_dumpcap else ()):
            if shutil.which(command) is None:
                raise RuntimeError(f'Required command not found: {command}')
        executable_path = args.claude_bin or shutil.which('claude')
        if not executable_path:
            raise RuntimeError('Claude Code not found in PATH')
        executable = Path(executable_path).resolve(strict=True)
        env = solver_environment()
        if not (env.get('ANTHROPIC_API_KEY') or env.get('ANTHROPIC_AUTH_TOKEN') or (Path.home()/'.claude'/'.credentials.json').is_file()):
            raise RuntimeError('Configure Claude API credentials in env or ~/.claude/settings.json, or sign in to Claude Code')
        check_isolation(dataset_root, executable, env)
        print('[BOOT] Filesystem isolation checked; original dataset is inaccessible to Claude.')
        run_root = (args.resume_dir or (args.results_root / datetime.now().strftime('%Y%m%d-%H%M%S-%f'))).resolve()
        if run_root.is_relative_to(dataset_root):
            raise ValueError('Results cannot be written inside the original dataset')
        if any(run_root.is_relative_to(Path(p).resolve()) for p in ('/usr', '/bin', '/sbin', '/lib', '/lib64')):
            raise ValueError('Results cannot be under directories exposed to the solver')
        experiment_path = run_root / 'experiment.json'
        selection = {'year': args.year, 'split': args.split, 'task_ids': [t.task_id for t in tasks]}
        if args.resume_dir:
            if not experiment_path.is_file():
                raise ValueError('--resume-dir must contain a GAIA experiment.json')
            previous = json.loads(experiment_path.read_text(encoding='utf-8'))
            if previous.get('selection') != selection or previous.get('claude_model') != args.model or previous.get('claude_effort') != args.effort:
                raise ValueError('Resume selection/model/effort do not match the existing experiment')
        else:
            run_root.mkdir(mode=0o700, parents=True, exist_ok=False)
            write_json(experiment_path, {
                'created_at': now_iso(), 'selection': selection, 'task_count': len(tasks),
                'dataset_format': 'local Parquet', 'claude_model': args.model, 'claude_effort': args.effort,
                'claude_max_turns': CLAUDE_MAX_TURNS, 'claude_timeout_s': CLAUDE_TIMEOUT_S,
                'claude_output_format': 'stream-json', 'claude_session_persistence': False,
                'claude_timeline_timestamp_source': 'runner reads Claude stdout (UTC)',
                'claude_timing_notice': (
                    'llm_response_start is not LLM HTTP request start; '
                    'tool_dispatch_observed is not exact tool dispatch time; use PCAP for wire-level timestamps'
                ),
                'isolation': 'bubblewrap mount/PID namespaces; only current attachment and temporary home are exposed',
                'tools': TOOLS.split(','), 'scoring': 'GAIA numeric/list/text normalization; withheld answers remain ungraded',
            })
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'[ERROR] {exc}')
        return 2

    print(f'Experiment dir: {run_root}')
    summary_rows = []
    interrupted = False
    try:
        for index, task in enumerate(tasks, 1):
            task_dir = run_root / task.name
            result_path = task_dir / 'result.json'
            if args.resume_dir and result_path.is_file():
                previous = json.loads(result_path.read_text(encoding='utf-8'))
                if previous.get('task_id') != task.task_id:
                    raise ValueError('Saved result task ID does not match its directory')
                summary_rows.append(previous)
                print(f'[{index}/{len(tasks)}] SKIP completed: {task.name}')
                continue
            task_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            print('=' * 72)
            print(f'[{index}/{len(tasks)}] {task.name}')
            capture_proc = None
            runtime_temp = None
            result = {
                'task_name': task.name, 'category_id': task.split, 'task_id': task.task_id,
                'level': task.level, 'started_at': now_iso(), 'setup_ok': False,
                'claude_exit_code': None, 'claude_timed_out': False,
                'claude_session_id': None, 'claude_event_count': 0, 'claude_tool_call_count': 0,
                'verify_passed': None, 'verify_returncode': None, 'cleanup_ok': False,
                'scoring_available': task.expected_answer is not None, 'predicted_answer': None,
                'error': None, 'pcap': str(task_dir / 'raw_traffic.pcapng'),
            }
            try:
                runtime_temp = tempfile.TemporaryDirectory(prefix='gaia-task-', dir='/tmp')
                runtime = Path(runtime_temp.name)
                prepare_runtime(runtime, env)
                prompt = stage_task(task, runtime / 'work')
                (task_dir / 'task_prompt.txt').write_text(prompt, encoding='utf-8')
                result['setup_ok'] = True
                print('  [pcap] Starting dumpcap...')
                capture_proc = start_capture(task_dir / 'raw_traffic.pcapng', args.sudo_dumpcap)
                print('  [claude] Running isolated task...')
                rc, timed_out, error, meta = run_claude(
                    prompt, task_dir, runtime, executable, env, args.model, args.effort,
                    on_process_exit=lambda: stop_capture(capture_proc),
                )
                result.update(meta)
                result.update({'claude_exit_code': rc, 'claude_timed_out': timed_out, 'error': error})
                result['claude_started_at'] = meta['claude_process_started_at_utc']
                result['claude_ended_at'] = meta['claude_process_ended_at_utc']
                stop_capture(capture_proc)
                capture_proc = None
                output = json.loads((task_dir / 'claude_output.json').read_text(encoding='utf-8'))
                if output.get('is_error') or output.get('type') == 'result_unavailable':
                    result['error'] = result['error'] or str(
                        output.get('errors') or output.get('error') or output.get('result')
                        or 'Claude did not return a successful result')
                predicted = extract_answer(output)
                result['predicted_answer'] = predicted
                matched = score_answer(predicted, task.expected_answer)
                if matched is not None and (rc != 0 or timed_out or output.get('is_error')):
                    matched = False
                result['verify_passed'] = matched
                result['verify_returncode'] = None if matched is None else (0 if matched else 1)
                print('  [verify] ' + ('UNSCORED (reference answer withheld)' if matched is None else 'PASS' if matched else 'FAIL'))
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                result['error'] = str(exc)
                print(f'  [ERROR] {exc}')
            finally:
                try:
                    if capture_proc is not None:
                        stop_capture(capture_proc)
                finally:
                    if runtime_temp is not None:
                        try:
                            runtime_temp.cleanup()
                            result['cleanup_ok'] = True
                        except Exception as exc:
                            result['error'] = (result['error'] + '; ' if result['error'] else '') + f'cleanup: {exc}'
                    result['ended_at'] = now_iso()
                    write_json(result_path, result)
                    summary_rows.append(result)
            if result['error'] or result['claude_timed_out'] or result['claude_exit_code'] not in (None, 0):
                print(f"[STOP] Task failed: {task.name}: {result['error']}")
                break
    except KeyboardInterrupt:
        interrupted = True
        print('\n[STOP] Interrupted by user.')
    finally:
        fields = ['task_name', 'category_id', 'task_id', 'level', 'setup_ok',
                  'claude_exit_code', 'claude_timed_out', 'scoring_available',
                  'verify_passed', 'verify_returncode', 'cleanup_ok', 'error', 'pcap']
        with (run_root / 'summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(summary_rows)
    graded = sum(row.get('verify_passed') is not None for row in summary_rows)
    passed = sum(row.get('verify_passed') is True for row in summary_rows)
    print(f'Completed: {len(summary_rows)}; verified: {passed}/{graded}; unscored: {len(summary_rows)-graded}')
    print(f'Results: {run_root}')
    if interrupted:
        return 130
    return 1 if any(row.get('error') for row in summary_rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
