"""Real-path acceptance for downstream personal Codex features on a packaged binary.

Python 3.11+, stdlib only. Usage:
    python personal_acceptance.py --codex ABSOLUTE\\PATH\\codex.exe [--case NAME|all]
    python personal_acceptance.py --self-test      (fixture server only, no binary)

Every case owns a temporary CODEX_HOME, a working directory, a loopback Responses
fixture serving SSE on /v1/responses, and one `codex app-server --stdio` process
configured with `-c` overrides. Assertions use captured model requests and
app-server responses/notifications only. Prints one PASS/FAIL/SKIP line per case
and exits non-zero when any case fails.
"""

import argparse
import http.client
import http.server
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid


TIMEOUT = 60
API_KEY_ENV = "PERSONAL_ACCEPTANCE_API_KEY"
API_KEY = "personal-acceptance-key"
RECOVERY_SNIPPET = "previous response did not provide a visible final answer"
RECOVERED_ANSWER = "ACCEPTANCE_RECOVERED_ANSWER"
SUMMARIZATION_MARKER = "CONTEXT CHECKPOINT COMPACTION"
CHILD_TASK = "ACCEPTANCE_CHILD_TASK"
IDENTITY_PREFIX = "You are a spawned subagent"
EXEC_LITERAL_ARGS = ["two words", "a & b", "x | y", "%PATH%", "100% done"]


class Skip(Exception):
    pass


class CaseDirectory(tempfile.TemporaryDirectory):
    def cleanup(self):
        # Windows can retain directory handles briefly after child processes exit.
        for attempt in range(20):
            try:
                super().cleanup()
                return
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.25)


# ---------------------------------------------------------------- model fixture


def assistant(text, phase=None):
    item = {"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": text}]}
    if phase:
        item["phase"] = phase
    return item


def reasoning():
    return {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Thinking."}],
            "encrypted_content": "YWNjZXB0YW5jZS1yZWFzb25pbmc="}


def function_call(call_id, name, arguments, namespace=None):
    item = {"type": "function_call", "call_id": call_id, "name": name,
            "arguments": json.dumps(arguments)}
    if namespace:
        item.update(namespace=namespace, encrypted_function_args=[])
    return item


def input_items(body):
    return body.get("input", [])


def message_texts(body, roles=("user", "developer")):
    return [
        part.get("text", "")
        for item in input_items(body)
        if item.get("type") in ("message", "agent_message") and item.get("role") in roles
        for part in item.get("content", []) or []
        if isinstance(part, dict)
    ]


def is_child_request(body):
    # Mirrors the continuation smoke: the task may arrive as a message or agent_message,
    # and tool-call arguments (which also contain the task) are excluded.
    return CHILD_TASK in "\n".join(
        part.get("text", "")
        for item in input_items(body)
        if item.get("type") in ("message", "agent_message") and item.get("role") != "assistant"
        for part in item.get("content", []) or []
        if isinstance(part, dict)
    )


def tool_output_text(body, call_id):
    for item in input_items(body):
        if item.get("type") == "function_call_output" and item.get("call_id") == call_id:
            output = item["output"]
            if isinstance(output, list):
                output = "".join(part.get("text", "") for part in output)
            return output
    return None


def tool_names(body):
    """Top-level tool names plus `namespace.child` names for namespaced tools."""
    names = set()
    for tool in body.get("tools", []) or []:
        name = tool.get("name") or tool.get("type")
        names.add(name)
        for child in tool.get("tools", []) or []:
            names.add(f"{name}.{child.get('name')}")
    return names


class Fixture(http.server.ThreadingHTTPServer):
    """Loopback Responses endpoint; `responder(body, index)` returns output items."""

    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def __init__(self, responder):
        super().__init__(("127.0.0.1", 0), FixtureHandler)
        self.responder = responder
        self.requests = []
        self.headers = []
        self.errors = []
        self.lock = threading.Lock()
        self.arrived = threading.Condition(self.lock)

    def bodies(self):
        with self.lock:
            return list(self.requests)

    def wait_for(self, predicate, timeout=TIMEOUT):
        """Waits until a captured request body satisfies `predicate`; returns it."""
        deadline = time.monotonic() + timeout
        with self.arrived:
            while True:
                for body in self.requests:
                    if predicate(body):
                        return body
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("expected model request never arrived")
                self.arrived.wait(remaining)


class FixtureHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        try:
            assert self.path.endswith("/responses"), f"unexpected path {self.path}"
            assert not self.headers.get("Content-Encoding"), "compressed request"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with self.server.arrived:
                index = len(self.server.requests)
                self.server.requests.append(body)
                self.server.headers.append({k.lower(): v for k, v in self.headers.items()})
                self.server.arrived.notify_all()
            response_id = f"acceptance-response-{index + 1}"
            output = self.server.responder(body, index)
            for position, item in enumerate(output):
                item.setdefault("id", f"{response_id}-item-{position}")
            events = [{"type": "response.created", "response": {"id": response_id}}]
            events.extend(
                {"type": "response.output_item.done", "output_index": i, "item": item}
                for i, item in enumerate(output)
            )
            events.append({"type": "response.completed", "response": {
                "id": response_id, "output": output,
                "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
            }})
            payload = "".join("data: " + json.dumps(e) + "\n\n" for e in events).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
        except Exception as error:
            self.server.errors.append(repr(error))
            try:
                self.send_error(500, str(error))
            except OSError:
                pass


# ---------------------------------------------------------------- app-server


def fixture_config(port, overrides):
    config = {
        "model": "fixture-model", "model_provider": "fixture",
        "model_context_window": 100000, "model_auto_compact_token_limit": 95000,
        "model_reasoning_effort": "low", "project_doc_max_bytes": 0,
        "features.code_mode": False, "features.code_mode_only": False,
        "features.enable_request_compression": False,
        "model_providers.fixture.name": "Acceptance Responses",
        "model_providers.fixture.base_url": f"http://127.0.0.1:{port}/v1",
        "model_providers.fixture.wire_api": "responses",
        "model_providers.fixture.supports_websockets": False,
        "model_providers.fixture.requires_openai_auth": False,
        "model_providers.fixture.request_max_retries": 0,
        "model_providers.fixture.stream_max_retries": 0,
    }
    config.update(overrides)
    return config


class Rpc:
    """Drives one app-server over stdio."""

    def __init__(self, binary, root, config, env_extra=None):
        env = os.environ.copy()
        for key in list(env):
            if key.upper().startswith(("OPENAI_", "CODEX_")) or key.upper() in (
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            ):
                env.pop(key)
        env.update(CODEX_HOME=str(root / "home"), NO_PROXY="127.0.0.1,localhost")
        env.update(env_extra or {})
        args = [str(binary), "app-server", "--stdio"]
        for key, value in config.items():
            args.extend(["-c", f"{key}={json.dumps(value)}"])
        self.queue = queue.Queue()
        self.pending = []
        self.events = []
        self.next_id = 1
        self.stderr = (root / "app-server.log").open("a", encoding="utf-8")
        try:
            self.proc = subprocess.Popen(
                args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
                text=True, encoding="utf-8", env=env, cwd=root / "work",
            )
        except Exception:
            self.stderr.close()
            raise
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()
        try:
            self.request("initialize", {
                "clientInfo": {"name": "personal_acceptance", "version": "1"},
                "capabilities": {"experimentalApi": True},
            })
            self.send({"method": "initialized"})
        except Exception:
            self.close()
            raise

    def read(self):
        for line in self.proc.stdout:
            try:
                self.queue.put(json.loads(line))
            except json.JSONDecodeError:
                self.queue.put({"readerError": line})
        self.queue.put({"readerError": "app-server stdout closed"})

    def send(self, message):
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def request(self, method, params):
        request_id = self.next_id
        self.next_id += 1
        self.send({"id": request_id, "method": method, "params": params})
        result = self.wait(lambda message: message.get("id") == request_id)
        if "error" in result:
            raise RuntimeError(f"{method}: {result['error']}")
        return result["result"]

    def wait(self, predicate, timeout=TIMEOUT):
        for index, message in enumerate(self.pending):
            if predicate(message):
                return self.pending.pop(index)
        deadline = time.monotonic() + timeout
        while True:
            try:
                message = self.queue.get(timeout=max(0, deadline - time.monotonic()))
            except queue.Empty:
                raise TimeoutError("app-server event deadline expired") from None
            self.events.append(message)
            if "readerError" in message:
                raise RuntimeError(message["readerError"])
            if "method" in message and "id" in message:
                raise AssertionError(f"unexpected interactive RPC: {message['method']}")
            if predicate(message):
                return message
            self.pending.append(message)

    def notifications(self, method, thread_id):
        return [
            event["params"] for event in self.events
            if event.get("method") == method and event["params"].get("threadId") == thread_id
        ]

    def start_thread(self, root):
        return self.request("thread/start", {
            "cwd": str(root / "work"), "approvalPolicy": "never",
            "sandbox": "danger-full-access",
        })["thread"]["id"]

    def completed(self, thread_id, turn_id, status="completed"):
        turn = self.wait(lambda e: (
            e.get("method") == "turn/completed"
            and e["params"]["threadId"] == thread_id
            and e["params"]["turn"]["id"] == turn_id
        ))["params"]["turn"]
        assert turn["status"] == status and not turn.get("error"), f"turn ended {turn}"
        return turn

    def turn(self, thread_id, text, **params):
        turn_id = self.request("turn/start", {
            "threadId": thread_id, "input": [{"type": "text", "text": text}], **params,
        })["turn"]["id"]
        self.completed(thread_id, turn_id)
        return turn_id

    def close(self):
        try:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
            self.reader.join(timeout=5)
            self.proc.stdout.close()
        finally:
            self.stderr.close()


class Harness:
    """Per-case temp CODEX_HOME + fixture + app-server, torn down on exit."""

    def __init__(self, binary, name, responder):
        self.binary = binary
        self.name = name
        self.responder = responder

    def __enter__(self):
        self.directory = CaseDirectory(prefix=f"codex-acceptance-{self.name}-")
        self.root = Path(self.directory.name)
        (self.root / "home").mkdir()
        (self.root / "work").mkdir()
        self.model = Fixture(self.responder)
        self.server = threading.Thread(target=self.model.serve_forever, daemon=True)
        self.server.start()
        self.rpc = None
        return self

    def start(self, overrides=None, env_extra=None):
        config = fixture_config(self.model.server_port, overrides or {})
        self.rpc = Rpc(self.binary, self.root, config, env_extra)
        self.thread = self.rpc.start_thread(self.root)
        return self.rpc

    def write_catalog(self, models):
        path = self.root / "home" / "acceptance-models.json"
        path.write_text(json.dumps({"models": models}), encoding="utf-8")
        return str(path)

    def __exit__(self, kind, error, _traceback):
        try:
            if error is not None and not isinstance(error, Skip):
                if self.model.errors:
                    print(f"[{self.name}] fixture errors: {self.model.errors}", file=sys.stderr)
                log = self.root / "app-server.log"
                if log.exists():
                    tail = log.read_text(encoding="utf-8", errors="replace")[-4000:]
                    print(f"[{self.name}] app-server log tail:\n{tail}", file=sys.stderr)
            if self.rpc is not None:
                self.rpc.close()
        finally:
            self.model.shutdown()
            self.model.server_close()
            self.server.join(timeout=5)
            self.directory.cleanup()
        return False


def catalog_model(slug, **overrides):
    """Minimal ModelInfo entry accepted by `model_catalog_json`."""
    model = {
        "slug": slug, "display_name": slug, "description": "Acceptance fixture model",
        "base_instructions": "You are a deterministic acceptance fixture model.",
        "default_reasoning_level": "low",
        "supported_reasoning_levels": [{"effort": "low", "description": "Low"}],
        "shell_type": "shell_command", "visibility": "list", "supported_in_api": True,
        "priority": 1, "availability_nux": None, "upgrade": None,
        "support_verbosity": False, "default_verbosity": None,
        "apply_patch_tool_type": None,
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "context_window": 272000, "max_context_window": 272000,
        "experimental_supported_tools": [], "input_modalities": ["text", "image"],
    }
    model.update(overrides)
    return model


def assert_no_fixture_errors(h):
    assert not h.model.errors, f"fixture errors: {h.model.errors}"


# ---------------------------------------------------------------- cases


def case_exec_argv(binary, enabled=True):
    code = "import json, sys; print(json.dumps(sys.argv[1:]))"
    argv = [sys.executable, "-c", code, *EXEC_LITERAL_ARGS]

    def respond(body, index):
        if index == 0 and enabled:
            return [function_call("acceptance-exec", "exec_argv",
                                  {"argv": argv, "yield_time_ms": 10000})]
        return [assistant("exec_argv fixture done")]

    with Harness(binary, "exec_argv", respond) as h:
        rpc = h.start({"features.exec_argv": enabled})
        rpc.turn(h.thread, "Run the argv fixture.")
        bodies = h.model.bodies()
        assert_no_fixture_errors(h)
        if not enabled:
            assert "exec_argv" not in tool_names(bodies[0]), "exec_argv exposed with flag off"
            return "exec_argv absent from tool list with flag off"
        assert "exec_argv" in tool_names(bodies[0]), "exec_argv missing from tool list"
        assert len(bodies) == 2, f"expected 2 model requests, saw {len(bodies)}"
        output = tool_output_text(bodies[1], "acceptance-exec")
        assert output is not None, "follow-up request lacks exec_argv output"
        expected = json.dumps(EXEC_LITERAL_ARGS)
        # Accept the output either as plain text or as a JSON-encoded string.
        assert expected in output or json.dumps(expected)[1:-1] in output, (
            f"arguments not literal; output={output[-600:]!r}"
        )
        return "tool listed; child received spaces, &, |, %PATH% literally"


def case_final_answer_retry(binary, first, enabled=True):
    def respond(body, index):
        if index == 0:
            return [assistant(" \n\t", phase="final_answer")] if first == "empty" else [reasoning()]
        return [assistant(RECOVERED_ANSWER)]

    with Harness(binary, f"final_answer_retry_{first}", respond) as h:
        rpc = h.start({"features.final_answer_retry": enabled})
        turn = rpc.turn(h.thread, "Please answer.")
        time.sleep(1)  # bounded window for an unwanted extra request
        bodies = h.model.bodies()
        assert_no_fixture_errors(h)
        prompts = [any(RECOVERY_SNIPPET in text for text in message_texts(b, ("developer",)))
                   for b in bodies]
        if not enabled:
            assert len(bodies) == 1, f"flag off sent {len(bodies)} requests"
            assert not any(prompts), "recovery prompt sent with flag off"
            return "single request, no recovery prompt with flag off"
        assert len(bodies) == 2, f"expected exactly 2 requests, saw {len(bodies)}"
        assert prompts == [False, True], f"recovery prompt placement {prompts}"
        messages = [
            item["text"] for item in (
                params["item"] for params in rpc.notifications("item/completed", h.thread)
                if params.get("turnId", turn) == turn
            ) if item.get("type") == "agentMessage"
        ]
        assert RECOVERED_ANSWER in messages, f"recovered answer not completed: {messages}"
        return f"{first} first response -> one recovery request -> recovered answer"


def case_imagegen(binary, enabled=True):
    def respond(body, index):
        return [assistant("image fixture done")]

    with Harness(binary, "imagegen_auth_independent", respond) as h:
        catalog = h.write_catalog([catalog_model("fixture-vision")])
        rpc = h.start({
            "model": "fixture-vision", "model_catalog_json": catalog,
            "model_providers.fixture.env_key": API_KEY_ENV,
            "features.image_generation": True,
            "features.auth_independent_tools": enabled,
        }, {API_KEY_ENV: API_KEY})
        rpc.turn(h.thread, "Show me your tools.")
        body = h.model.bodies()[0]
        assert_no_fixture_errors(h)
        auth = h.model.headers[0].get("authorization")
        assert auth == f"Bearer {API_KEY}", f"request did not use env_key credentials: {auth!r}"
        namespace = next((t for t in body.get("tools", []) if t.get("name") == "image_gen"), None)
        if not enabled:
            assert namespace is None, "image tool exposed with auth_independent_tools off"
            return "custom provider without flag exposes no image tool"
        assert namespace is not None, f"image_gen namespace missing; tools={sorted(tool_names(body))}"
        tool = next((t for t in namespace.get("tools", []) if t.get("name") == "imagegen"), None)
        assert tool is not None, "imagegen function missing from image_gen namespace"
        parameters = tool["parameters"]
        required = parameters.get("required", [])
        for field in ("model", "size", "quality"):
            assert field in parameters.get("properties", {}), f"schema lacks {field}"
            assert field not in required, f"{field} is required"
        quality = json.dumps(parameters["properties"]["quality"])
        missing = [v for v in ("low", "medium", "high", "xhigh", "max", "auto")
                   if f'"{v}"' not in quality]
        assert not missing, f"quality schema lacks {missing}"
        return "custom env_key provider exposes image_gen.imagegen with optional model/size/quality incl. xhigh/max"


def case_context_usage(binary):
    def respond(body, index):
        return [assistant("context usage fixture done")]

    with Harness(binary, "context_usage", respond) as h:
        rpc = h.start()
        turn = rpc.turn(h.thread, "Answer briefly.")
        updates = [p for p in rpc.notifications("thread/tokenUsage/updated", h.thread)
                   if p["turnId"] == turn]
        assert updates, "no thread/tokenUsage/updated for the turn"
        usage = updates[-1]["tokenUsage"]
        context = usage.get("contextUsage")
        assert context, f"notification lacks contextUsage: {usage}"
        assert context["tokens"] > 0, context
        assert sum(context["breakdown"].values()) == context["tokens"], context
        assert context.get("autoCompactTokenLimit"), f"no compaction trigger: {context}"
        read = rpc.request("thread/tokenUsage/read", {"threadId": h.thread})["data"]
        entry = next((e for e in read if e["turnId"] == turn), None)
        assert entry, f"thread/tokenUsage/read lacks turn {turn}: {read}"
        assert entry["tokenUsage"].get("contextUsage") == context, (
            f"read differs from live: {entry['tokenUsage'].get('contextUsage')} vs {context}"
        )
        return (f"live contextUsage tokens={context['tokens']} "
                f"limit={context['autoCompactTokenLimit']} matches thread/tokenUsage/read")


def case_model_aware_compaction(binary):
    def is_trigger(body):
        return any(item.get("type") == "compaction_trigger" for item in input_items(body))

    def is_summarization(body):
        return SUMMARIZATION_MARKER in json.dumps(input_items(body))

    def respond(body, index):
        if is_trigger(body):
            return [{"type": "compaction", "encrypted_content": "acceptance-remote-summary"}]
        if is_summarization(body):
            return [assistant("ACCEPTANCE_LOCAL_SUMMARY")]
        return [assistant("compaction fixture answer")]

    def compact(rpc, thread, seen):
        rpc.request("thread/compact/start", {"threadId": thread})
        turn = rpc.wait(lambda e: (
            e.get("method") == "turn/started" and e["params"]["threadId"] == thread
            and e["params"]["turn"]["id"] not in seen
        ))["params"]["turn"]["id"]
        seen.add(turn)
        rpc.completed(thread, turn)

    with Harness(binary, "model_aware_compaction", respond) as h:
        # Equal comp_hash keeps the model switch itself from forcing a compaction.
        catalog = h.write_catalog([
            catalog_model("fixture-remote", comp_hash="acceptance"),
            catalog_model("fixture-local", comp_hash="acceptance",
                          supports_remote_compaction=False),
        ])
        rpc = h.start({
            "model": "fixture-remote", "model_catalog_json": catalog,
            "model_providers.fixture.capabilities.remote_compaction": "v2",
        })
        seen = {rpc.turn(h.thread, "Seed history on the remote-capable model.")}
        compact(rpc, h.thread, seen)
        remote_count = len(h.model.bodies())
        seen.add(rpc.turn(h.thread, "Continue on the local-only model.", model="fixture-local"))
        compact(rpc, h.thread, seen)
        bodies = h.model.bodies()
        assert_no_fixture_errors(h)
        remote = [b for b in bodies[:remote_count] if is_trigger(b) or is_summarization(b)]
        local = [b for b in bodies[remote_count:] if is_trigger(b) or is_summarization(b)]
        assert len(remote) == 1 and len(local) == 1, (
            f"compaction requests remote={len(remote)} local={len(local)}"
        )
        assert remote[0]["model"] == "fixture-remote" and is_trigger(remote[0]), (
            "remote-enabled model did not send compaction_trigger"
        )
        assert not is_summarization(remote[0]), "remote compaction carried summarization prompt"
        assert local[0]["model"] == "fixture-local" and is_summarization(local[0]), (
            "remote-disabled model did not send local summarization"
        )
        assert not is_trigger(local[0]), "remote-disabled model sent compaction_trigger"
        return "same provider: fixture-remote -> compaction_trigger, fixture-local -> summarization prompt"


def case_subagent_identity(binary, enabled=True):
    child_seen = threading.Event()

    def respond(body, index):
        if is_child_request(body):
            child_seen.set()
            return [assistant("child fixture done")]
        if not any(item.get("type") == "function_call_output" for item in input_items(body)):
            return [function_call("acceptance-spawn", "spawn_agent", {
                "message": CHILD_TASK, "task_name": "worker", "fork_turns": "none",
            }, namespace="collaboration")]
        return [assistant("parent fixture done")]

    with Harness(binary, "subagent_identity", respond) as h:
        rpc = h.start({
            "features.multi_agent": True,
            "features.multi_agent_v2.enabled": True,
            "features.multi_agent_v2.tool_namespace": "collaboration",
            "features.subagent_identity": enabled,
        })
        rpc.turn(h.thread, "Spawn the fixture worker.")
        assert child_seen.wait(TIMEOUT), "child model request never arrived"
        child = h.model.wait_for(is_child_request)
        assert_no_fixture_errors(h)
        texts = "\n".join(message_texts(child))
        if not enabled:
            assert IDENTITY_PREFIX not in texts, "identity context present with flag off"
            return "child request has no identity context with flag off"
        expected = "Your current canonical agent path is `/root/worker`"
        assert expected in texts, "child request lacks identity naming /root/worker"
        return "child's first request names canonical path /root/worker"


def write_orphaned_rollout(home, cwd, paginated):
    """Turn A starts and never finishes; turn B starts later and completes."""
    thread_id = str(uuid.uuid4())
    stamp = "2025-01-05T12:00:00.000Z"
    directory = home / "sessions" / "2025" / "01" / "05"
    directory.mkdir(parents=True)
    path = directory / f"rollout-2025-01-05T12-00-00-{thread_id}.jsonl"
    meta = {
        "id": thread_id, "session_id": thread_id, "timestamp": stamp, "cwd": str(cwd),
        "originator": "codex", "cli_version": "0.0.0", "source": "vscode",
        "model_provider": "fixture", "base_instructions": None,
        "history_mode": "paginated" if paginated else "legacy",
    }

    def user(text):
        return [
            ("response_item", {"type": "message", "role": "user",
                               "content": [{"type": "input_text", "text": text}]}),
            ("event_msg", {"type": "user_message", "message": text}),
        ]

    records = [("session_meta", meta)]
    records.append(("event_msg", {"type": "task_started", "turn_id": "turn-orphan",
                                  "started_at": 1736078400, "model_context_window": None}))
    records += user("ORPHAN_A_REQUEST")
    records.append(("event_msg", {"type": "agent_message", "message": "ORPHAN_A_PARTIAL"}))
    records.append(("event_msg", {"type": "task_started", "turn_id": "turn-next",
                                  "started_at": 1736078460, "model_context_window": None}))
    records += user("NEXT_B_REQUEST")
    records.append(("event_msg", {"type": "agent_message", "message": "NEXT_B_ANSWER"}))
    records.append(("event_msg", {"type": "task_complete", "turn_id": "turn-next",
                                  "last_agent_message": "NEXT_B_ANSWER",
                                  "completed_at": 1736078470, "duration_ms": 10000}))
    lines = []
    for ordinal, (kind, payload) in enumerate(records):
        line = {"timestamp": stamp, "type": kind, "payload": payload}
        if paginated:
            line["ordinal"] = ordinal
        lines.append(json.dumps(line))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return thread_id, path


def case_superseded_turn(binary, paginated):
    def respond(body, index):
        return [assistant("unused")]

    name = "superseded_turn_closure_" + ("paginated" if paginated else "legacy")
    with Harness(binary, name, respond) as h:
        source, source_path = write_orphaned_rollout(h.root / "home", h.root / "work", paginated)
        config = fixture_config(h.model.server_port, {})
        rpc = h.rpc = Rpc(binary, h.root, config)
        if paginated:
            # Resuming by path projects the paginated rollout into the thread store.
            rpc.request("thread/resume", {"threadId": source, "path": str(source_path)})

        def turns(thread_id):
            if paginated:
                return rpc.request("thread/turns/list", {
                    "threadId": thread_id, "sortDirection": "asc", "itemsView": "full",
                })["data"]
            return rpc.request("thread/read", {"threadId": thread_id,
                                               "includeTurns": True})["thread"]["turns"]

        status = {turn["id"]: turn["status"] for turn in turns(source)}
        assert status.get("turn-orphan") == "interrupted", f"source turn statuses {status}"
        assert status.get("turn-next") == "completed", f"source turn statuses {status}"
        fork = rpc.request("thread/fork", {
            "threadId": source, "lastTurnId": "turn-orphan", "excludeTurns": paginated,
        })["thread"]
        forked = fork["turns"] if not paginated else turns(fork["id"])
        if paginated:
            # Inherited paginated turns are listed by id without hydrated items.
            ids = [turn["id"] for turn in forked]
            assert ids == ["turn-orphan"], f"forked turn ids {ids}"
        else:
            dump = json.dumps(forked)
            assert "ORPHAN_A_REQUEST" in dump, "fork lost the orphaned turn's records"
            assert "NEXT_B_REQUEST" not in dump and "NEXT_B_ANSWER" not in dump, (
                "fork included records from the superseding turn"
            )
        assert forked[-1]["status"] == "interrupted", (
            f"forked anchor status {forked[-1]['status']}"
        )
        return "orphaned turn reads interrupted; lastTurnId fork keeps A and excludes B"


CASES = {
    "exec_argv": lambda b: case_exec_argv(b),
    "exec_argv_off": lambda b: case_exec_argv(b, enabled=False),
    "final_answer_retry_empty": lambda b: case_final_answer_retry(b, "empty"),
    "final_answer_retry_reasoning": lambda b: case_final_answer_retry(b, "reasoning"),
    "final_answer_retry_off": lambda b: case_final_answer_retry(b, "empty", enabled=False),
    "imagegen_auth_independent": lambda b: case_imagegen(b),
    "imagegen_auth_independent_off": lambda b: case_imagegen(b, enabled=False),
    "context_usage": case_context_usage,
    "model_aware_compaction": case_model_aware_compaction,
    "subagent_identity": lambda b: case_subagent_identity(b),
    "subagent_identity_off": lambda b: case_subagent_identity(b, enabled=False),
    "superseded_turn_closure_legacy": lambda b: case_superseded_turn(b, paginated=False),
    "superseded_turn_closure_paginated": lambda b: case_superseded_turn(b, paginated=True),
}


# ---------------------------------------------------------------- fixture self-test


def self_test():
    """Exercises the fixture server, catalog, and rollout writer without a binary."""

    def respond(body, index):
        if index == 0:
            return [function_call("call-1", "spawn_agent", {"message": "x"}, "collaboration"),
                    reasoning()]
        return [assistant("ok", phase="final_answer")]

    model = Fixture(respond)
    server = threading.Thread(target=model.serve_forever, daemon=True)
    server.start()
    try:
        for index in range(2):
            body = json.dumps({"model": "m", "input": [
                {"type": "message", "role": "developer",
                 "content": [{"type": "input_text", "text": "dev"}]},
            ], "tools": [{"type": "namespace", "name": "image_gen",
                          "tools": [{"type": "function", "name": "imagegen"}]}]})
            connection = http.client.HTTPConnection("127.0.0.1", model.server_port, timeout=10)
            connection.request("POST", "/v1/responses", body,
                               {"Content-Type": "application/json",
                                "Authorization": f"Bearer {API_KEY}"})
            response = connection.getresponse()
            assert response.status == 200, response.status
            assert response.getheader("Content-Type") == "text/event-stream"
            events = [json.loads(chunk[len("data: "):])
                      for chunk in response.read().decode().split("\n\n") if chunk]
            connection.close()
            types = [e["type"] for e in events]
            assert types[0] == "response.created" and types[-1] == "response.completed", types
            items = [e["item"] for e in events if e["type"] == "response.output_item.done"]
            assert all(item.get("id") for item in items), items
            assert events[-1]["response"]["output"] == items
            if index == 0:
                assert items[0]["namespace"] == "collaboration"
                assert json.loads(items[0]["arguments"]) == {"message": "x"}
        assert len(model.bodies()) == 2 and not model.errors, model.errors
        assert model.headers[0]["authorization"] == f"Bearer {API_KEY}"
        assert message_texts(model.bodies()[0], ("developer",)) == ["dev"]
        assert "image_gen.imagegen" in tool_names(model.bodies()[0])
        assert model.wait_for(lambda b: b["model"] == "m", timeout=1)
        connection = http.client.HTTPConnection("127.0.0.1", model.server_port, timeout=10)
        connection.request("POST", "/v1/other", "{}", {"Content-Type": "application/json"})
        assert connection.getresponse().status == 500 and model.errors
        connection.close()
    finally:
        model.shutdown()
        model.server_close()
        server.join(timeout=5)
    args = [f"{k}={json.dumps(v)}" for k, v in fixture_config(1234, {
        "model_catalog_json": r"C:\temp\models.json"}).items()]
    assert r'model_catalog_json="C:\\temp\\models.json"' in args, args
    json.dumps({"models": [catalog_model("a"), catalog_model("b", supports_remote_compaction=False)]})
    with CaseDirectory(prefix="codex-acceptance-selftest-") as temporary:
        root = Path(temporary)
        for paginated in (False, True):
            home = root / ("paginated" if paginated else "legacy")
            thread_id, _ = write_orphaned_rollout(home, root, paginated)
            [path] = list(home.rglob("rollout-*.jsonl"))
            assert thread_id in path.name
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            assert lines[0]["type"] == "session_meta"
            assert [line.get("ordinal") for line in lines][-1] == (len(lines) - 1 if paginated else None)
    print("PASS self_test: fixture SSE, capture, error path, config args, catalog, rollout writer")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--codex", type=Path, help="absolute packaged codex.exe path")
    parser.add_argument("--case", default="all", choices=("all", *CASES))
    parser.add_argument("--self-test", action="store_true",
                        help="exercise the fixture server only; no binary needed")
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        parser.error("Python 3.11+ is required")
    if args.self_test:
        self_test()
        return
    if args.codex is None or not args.codex.is_absolute() or not args.codex.is_file():
        parser.error("--codex must name an existing absolute binary path")
    names = list(CASES) if args.case == "all" else [args.case]
    failed = False
    for name in names:
        try:
            reason = CASES[name](args.codex)
            print(f"PASS {name}: {reason}", flush=True)
        except Skip as skip:
            print(f"SKIP {name}: {skip}", flush=True)
        except Exception as error:
            failed = True
            message = " ".join(str(error).split()) or type(error).__name__
            print(f"FAIL {name}: {type(error).__name__}: {message[:400]}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
