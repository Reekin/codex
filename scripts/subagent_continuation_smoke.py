"""Exercise packaged app-server subagent continuation against a loopback Responses fixture.

Python 3.11+. Run with --codex ABSOLUTE/PATH/TO/codex.exe; no build or credentials.
Each V1/V2 case owns a temporary CODEX_HOME, working directory, and server process.
The interrupted-parent assertion observes a bounded idle window, not indefinite liveness.
Unsubscribe cases hold children beyond the unload delay and require autonomous consumption.
Run from this checkout: the adjacent local_context_compaction_smoke module is required.
"""

import argparse
import http.server
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time

from local_context_compaction_smoke import Rpc as BaseRpc
from local_context_compaction_smoke import SmokeDirectory, assistant


CHILD_TASK = "CONTINUATION_SMOKE_CHILD_TASK"
CHILD_RESULT = "CONTINUATION_SMOKE_CHILD_RESULT"
REUSE_TASK = "CONTINUATION_SMOKE_REUSED_TASK"
REUSE_RESULT = "CONTINUATION_SMOKE_REUSED_RESULT"
STEERING = "CONTINUATION_SMOKE_USER_STEERING"
TIMEOUT = 30
WAIT_MS = 120_000


def input_text(body):
    # Exclude tool-call arguments: the parent's spawn call contains the child marker.
    return "\n".join(
        part.get("text", "")
        for item in body.get("input", [])
        if item.get("type") in ("message", "agent_message")
        and item.get("role") != "assistant"
        for part in item.get("content", [])
    )


def tool_result(body, call_id):
    item = next(
        item for item in body["input"]
        if item.get("type") == "function_call_output"
        and item.get("call_id") == call_id
    )
    output = item["output"]
    if isinstance(output, list):
        output = "".join(part.get("text", "") for part in output)
    return json.loads(output)


class Model(http.server.ThreadingHTTPServer):
    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def __init__(self, version, scenario):
        super().__init__(("127.0.0.1", 0), Handler)
        self.version = version
        self.scenario = scenario
        self.parents = []
        self.errors = []
        self.lock = threading.Lock()
        self.child_started = threading.Event()
        self.release_child = threading.Event()
        self.child_sent = threading.Event()
        self.reuse_started = threading.Event()
        self.release_reuse = threading.Event()
        self.reuse_sent = threading.Event()
        self.followup_started = threading.Event()
        self.sequence = 0
        self.resume_prelude = 0

    def call(self, name, arguments, call_id):
        return {
            "type": "function_call", "id": call_id, "call_id": call_id,
            "name": name,
            "namespace": "multi_agent_v1" if self.version == "v1" else "collaboration",
            "arguments": json.dumps(arguments),
            "encrypted_function_args": [],
        }

    def respond(self, body):
        if REUSE_TASK in input_text(body):
            assert self.scenario == "reuse" and not self.reuse_started.is_set()
            assert CHILD_RESULT in json.dumps(body["input"]), "reused child lost its first turn"
            self.reuse_started.set()
            assert self.release_reuse.wait(TIMEOUT * 4), "reused child gate expired"
            return [assistant(REUSE_RESULT)], self.reuse_sent
        if CHILD_TASK in input_text(body):
            assert not self.child_started.is_set(), "unexpected second child request"
            self.child_started.set()
            assert self.release_child.wait(TIMEOUT * 4), "child gate expired"
            return [assistant(CHILD_RESULT)], self.child_sent
        with self.lock:
            self.parents.append(body)
            number = len(self.parents)
        if number == 1:
            arguments = {"message": CHILD_TASK}
            if self.version == "v1":
                arguments["fork_context"] = False
            else:
                arguments.update(task_name="worker", fork_turns="none")
            return [self.call("spawn_agent", arguments, "smoke-spawn")], False
        if number == 2:
            result = tool_result(body, "smoke-spawn")
            assert ("agent_id" if self.version == "v1" else "task_name") in result, result
            assert self.child_started.wait(TIMEOUT), "spawn did not request a child response"
            if self.scenario in ("continue", "reuse", "unsubscribe"):
                return [assistant("Parent reply complete; child remains in progress.")], False
            arguments = {"timeout_ms": WAIT_MS}
            if self.version == "v1":
                arguments["targets"] = [result["agent_id"]]
            return [self.call("wait_agent", arguments, "smoke-wait")], False
        assert number <= {"reuse": 5, "steer": 4, "interrupt": 4}.get(self.scenario, 3), (
            "duplicate or unexpected parent request", number
        )
        if self.scenario == "interrupt" and number == 3 and CHILD_RESULT not in input_text(body):
            # Explicit user input may be sampled before queued child mail is drained.
            assert "Process the retained child result." in input_text(body)
            self.resume_prelude = 1
            return [], False
        if self.scenario == "reuse":
            if number == 3:
                assert CHILD_RESULT in input_text(body), "first child result missing"
                self.followup_started.set()
                result = tool_result(body, "smoke-spawn")
                name = "send_input" if self.version == "v1" else "followup_task"
                target = result["agent_id" if self.version == "v1" else "task_name"]
                return [self.call(name, {"target": target, "message": REUSE_TASK}, "smoke-reuse")], False
            if number == 4:
                assert any(item.get("type") == "function_call_output"
                           and item.get("call_id") == "smoke-reuse" for item in body["input"])
                assert self.reuse_started.wait(TIMEOUT), "reused child request missing"
                assert not self.release_reuse.is_set()
                return [assistant("Parent reply complete; reused child remains in progress.")], False
            assert REUSE_RESULT in input_text(body), "second child result missing"
            self.followup_started.set()
            return [assistant("Parent processed the reused child's result.")], False
        if self.scenario == "steer" and number == 3:
            assert not self.release_child.is_set(), "steering waited for child completion"
            assert STEERING in input_text(body), "steering absent from next model request"
            assert tool_result(body, "smoke-wait")["timed_out"] is False
        else:
            assert CHILD_RESULT in input_text(body), "child result absent from parent input"
        self.followup_started.set()
        return [assistant("Parent processed the fixture input.")], False


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        try:
            assert self.path == "/v1/responses", self.path
            assert not self.headers.get("Content-Encoding"), "compressed request"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with self.server.lock:
                self.server.sequence += 1
                response_id = f"smoke-response-{self.server.sequence}"
            output, child = self.server.respond(body)
            for index, item in enumerate(output):
                item["id"] = f"{response_id}-item-{index}"
            events = [{"type": "response.created", "response": {"id": response_id}}]
            events.extend(
                {"type": "response.output_item.done", "output_index": i, "item": item}
                for i, item in enumerate(output)
            )
            events.append({
                "type": "response.completed",
                "response": {
                    "id": response_id, "output": output,
                    "usage": {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                },
            })
            payload = "".join("data: " + json.dumps(e) + "\n\n" for e in events).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            if child:
                child.set()
        except Exception as error:
            self.server.errors.append(repr(error))
            try:
                self.send_error(500, str(error))
            except OSError:
                pass


class Rpc(BaseRpc):
    """Reuse the existing stdout JSON reader and request framing, with fixture config."""

    def __init__(self, binary, root, model):
        env = os.environ.copy()
        for key in list(env):
            if key.upper().startswith(("OPENAI_", "CODEX_")) or key.upper() in (
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            ):
                env.pop(key)
        env.update(CODEX_HOME=str(root / "home"), NO_PROXY="127.0.0.1,localhost")
        config = {
            "model": "fixture-model", "model_provider": "fixture",
            "model_context_window": 100000, "model_auto_compact_token_limit": 95000,
            "model_reasoning_effort": "low", "project_doc_max_bytes": 0,
            "thread_unload_delay_secs": 1,
            "features.multi_agent": True,
            "features.multi_agent_v2.enabled": model.version == "v2",
            "features.multi_agent_v2.wait_agent_enabled": True,
            "features.multi_agent_v2.tool_namespace": "collaboration",
            "features.code_mode": False, "features.code_mode_only": False,
            "features.enable_request_compression": False,
            "model_providers.fixture.name": "Subagent continuation fixture",
            "model_providers.fixture.base_url": f"http://127.0.0.1:{model.server_port}/v1",
            "model_providers.fixture.wire_api": "responses",
            "model_providers.fixture.supports_websockets": False,
            "model_providers.fixture.requires_openai_auth": False,
            "model_providers.fixture.request_max_retries": 0,
            "model_providers.fixture.stream_max_retries": 0,
        }
        args = [str(binary), "app-server", "--stdio"]
        for key, value in config.items():
            args.extend(["-c", f"{key}={json.dumps(value)}"])
        self.queue = queue.Queue()
        self.pending = []
        self.events = []
        self.next_id = 1
        self.stderr = (root / "app-server.log").open("w", encoding="utf-8")
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
                "clientInfo": {"name": "subagent_continuation_smoke", "version": "1"},
                "capabilities": {"experimentalApi": True},
            })
            self.send({"method": "initialized"})
        except Exception:
            self.close()
            raise

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
                raise AssertionError(f"Unexpected interactive RPC: {message['method']}")
            if predicate(message):
                return message
            self.pending.append(message)

    def completed(self, thread_id, turn_id, status="completed"):
        event = self.wait(lambda e: (
            e.get("method") == "turn/completed"
            and e["params"]["threadId"] == thread_id
            and e["params"]["turn"]["id"] == turn_id
        ))
        turn = event["params"]["turn"]
        assert turn["status"] == status and not turn.get("error"), turn

    def continued(self, thread_id, previous_turn):
        event = self.wait(lambda e: e.get("method") == "turn/started"
                         and e["params"]["threadId"] == thread_id)
        turn = event["params"]["turn"]["id"]
        assert turn != previous_turn, "continuation reused completed turn id"
        return turn

    def idle(self, model, thread_id, count, seconds):
        # Consume stdout during the negative window so lifecycle events are checked too.
        try:
            unexpected = self.wait(lambda e: (
                e.get("method") == "turn/started"
                and e["params"]["threadId"] == thread_id
            ), timeout=seconds)
        except TimeoutError:
            pass
        else:
            raise AssertionError(f"Unexpected autonomous turn: {unexpected}")
        assert len(model.parents) == count, "parent made an unexpected model request"
        assert not model.errors, model.errors

    def unsubscribe_held(self, model, thread_id, count, seconds):
        result = self.request("thread/unsubscribe", {"threadId": thread_id})
        assert result["status"] == "unsubscribed", result
        # The child gate stays closed across the configured one-second unload delay.
        self.idle(model, thread_id, count, max(2, seconds))

    def consumed_unsubscribed(self, model, thread_id, previous_turn, count):
        # No RPC or user submission may wake the parent before this fixture assertion.
        # Model.respond sets the event only after checking the child result in input.
        assert model.followup_started.wait(TIMEOUT), (
            "unsubscribed parent never consumed child result"
        )
        assert len(model.parents) == count, (
            "unexpected parent request count before reattachment"
        )
        state = self.request("thread/resume", {"threadId": thread_id})["thread"]
        deadline = time.monotonic() + TIMEOUT
        while True:
            turns = state["turns"]
            if turns and turns[-1]["id"] != previous_turn:
                turn = turns[-1]
                assert turn["status"] in ("inProgress", "completed"), turn
                assert not turn.get("error"), turn
                if turn["status"] == "completed":
                    return turn["id"]
            assert time.monotonic() < deadline, (
                "autonomous parent turn did not complete"
            )
            time.sleep(0.05)
            state = self.request(
                "thread/read",
                {
                    "threadId": thread_id,
                    "includeTurns": True,
                },
            )["thread"]

    def unload_idle(self, model, thread_id, count):
        result = self.request("thread/unsubscribe", {"threadId": thread_id})
        assert result["status"] == "unsubscribed", result
        deadline = time.monotonic() + TIMEOUT
        while True:
            state = self.request("thread/read", {"threadId": thread_id})["thread"]
            assert len(model.parents) == count, (
                "parent restarted after result consumption"
            )
            assert not model.errors, model.errors
            if state["status"]["type"] == "notLoaded":
                return
            assert time.monotonic() < deadline, (
                "idle parent stayed loaded after unsubscribe"
            )
            time.sleep(0.05)

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


def start_turn(rpc, thread_id, text):
    return rpc.request("turn/start", {
        "threadId": thread_id, "input": [{"type": "text", "text": text}],
    })["turn"]["id"]


def run_case(binary, version, scenario, idle_seconds):
    with SmokeDirectory(prefix=f"codex-subagent-{version}-{scenario}-") as temporary:
        root = Path(temporary)
        (root / "home").mkdir()
        (root / "work").mkdir()
        model = Model(version, scenario)
        server = threading.Thread(target=model.serve_forever, daemon=True)
        server.start()
        rpc = None
        try:
            rpc = Rpc(binary, root, model)
            parent = rpc.request("thread/start", {
                "cwd": str(root / "work"), "approvalPolicy": "never",
                "sandbox": "danger-full-access",
            })["thread"]["id"]
            turn = start_turn(rpc, parent, "Run the isolated subagent fixture.")
            # Consume initial lifecycle start before asserting absence of future starts.
            rpc.wait(lambda e: e.get("method") == "turn/started"
                     and e["params"]["threadId"] == parent
                     and e["params"]["turn"]["id"] == turn)
            spawn = rpc.wait(lambda e: (
                e.get("method") == "item/completed"
                and e["params"]["threadId"] == parent
                and e["params"]["item"].get("id") == "smoke-spawn"
            ))["params"]["item"]
            if version == "v2":
                assert spawn["type"] == "subAgentActivity" and spawn["kind"] == "started", spawn
                child = spawn["agentThreadId"]
            else:
                assert spawn["status"] == "completed", spawn
                child = spawn["receiverThreadIds"][0]
            assert model.child_started.wait(TIMEOUT), "child request missing"
            if scenario in ("continue", "reuse", "unsubscribe"):
                rpc.completed(parent, turn)
            else:
                rpc.wait(lambda e: (
                    e.get("method") == "item/started"
                    and e["params"]["threadId"] == parent
                    and e["params"]["item"].get("id") == "smoke-wait"
                    and e["params"]["item"].get("tool") == "wait"
                ))
                if scenario == "steer":
                    started = time.monotonic()
                    rpc.request("turn/steer", {
                        "threadId": parent, "expectedTurnId": turn,
                        "input": [{"type": "text", "text": STEERING}],
                    })
                    assert model.followup_started.wait(TIMEOUT), "steering did not wake parent"
                    assert time.monotonic() - started < WAIT_MS / 1000
                    assert not model.release_child.is_set()
                    rpc.completed(parent, turn)
                else:
                    rpc.request("turn/interrupt", {"threadId": parent, "turnId": turn})
                    rpc.completed(parent, turn, "interrupted")
            count = len(model.parents)
            assert count == (3 if scenario == "steer" else 2), count
            if scenario == "steer":
                # Steering is independent of idle-parent continuation: keep the child
                # gated through this assertion so the old V2 baseline can pass it.
                assert not model.child_sent.is_set()
                assert not model.errors, model.errors
                return {"version": version, "scenario": scenario, "status": "passed",
                        "parent_requests": count, "child_held_through_steering": True}
            model.followup_started.clear()
            if scenario == "unsubscribe":
                assert not model.child_sent.is_set()
                rpc.unsubscribe_held(model, parent, count, idle_seconds)
            model.release_child.set()
            assert model.child_sent.wait(TIMEOUT), "child response was not delivered"
            if scenario == "interrupt":
                # thread/read confirms actual child completion, not merely an SSE write.
                deadline = time.monotonic() + TIMEOUT
                while True:
                    state = rpc.request("thread/read", {"threadId": child, "includeTurns": True})
                    turns = state["thread"]["turns"]
                    if turns and turns[-1]["status"] == "completed":
                        break
                    assert time.monotonic() < deadline, "child did not complete after parent interrupt"
                    time.sleep(0.05)
                rpc.idle(model, parent, count, idle_seconds)
                turn = start_turn(rpc, parent, "Process the retained child result.")
                rpc.wait(lambda e: e.get("method") == "turn/started"
                         and e["params"]["threadId"] == parent
                         and e["params"]["turn"]["id"] == turn)
            elif scenario == "unsubscribe":
                turn = rpc.consumed_unsubscribed(model, parent, turn, count + 1)
            else:
                turn = rpc.continued(parent, turn)
            if scenario != "unsubscribe":
                rpc.completed(parent, turn)
            assert model.followup_started.is_set(), "parent never consumed child result"
            if scenario == "reuse":
                assert len(model.parents) == 4, "parent did not submit the follow-up task"
                activity = rpc.wait(lambda e: (
                    e.get("method") == "item/completed"
                    and e["params"]["threadId"] == parent
                    and e["params"]["item"].get("id") == "smoke-reuse"
                ))["params"]["item"]
                if version == "v2":
                    assert activity["type"] == "subAgentActivity" and activity["kind"] == "interacted"
                    assert activity["agentThreadId"] == child, "followup_task changed child identity"
                else:
                    assert activity["status"] == "completed", activity
                    assert activity["receiverThreadIds"] == [child], "send_input changed child identity"
                child_turns = []
                for _ in range(2):
                    event = rpc.wait(lambda e: e.get("method") == "turn/started"
                                     and e["params"]["threadId"] == child)
                    child_turns.append(event["params"]["turn"]["id"])
                assert child_turns[0] != child_turns[1], "child did not start a second turn"
                assert model.reuse_started.is_set() and not model.reuse_sent.is_set()
                if version == "v1":
                    rpc.unsubscribe_held(model, parent, 4, idle_seconds)
                else:
                    rpc.idle(model, parent, 4, idle_seconds)
                model.followup_started.clear()
                model.release_reuse.set()
                assert model.reuse_sent.wait(TIMEOUT), "second child response was not delivered"
                if version == "v1":
                    turn = rpc.consumed_unsubscribed(model, parent, turn, 5)
                else:
                    turn = rpc.continued(parent, turn)
                    rpc.completed(parent, turn)
                assert model.followup_started.is_set(), "parent never consumed second child result"
                count = 4
            if scenario == "unsubscribe" or (scenario == "reuse" and version == "v1"):
                rpc.unload_idle(model, parent, count + 1)
            else:
                rpc.idle(model, parent, count + 1 + model.resume_prelude, idle_seconds)
            return {"version": version, "scenario": scenario, "status": "passed",
                    "parent_requests": len(model.parents), "idle_seconds": idle_seconds}
        except Exception:
            print(json.dumps({"version": version, "scenario": scenario,
                              "model_errors": model.errors,
                              "parent_requests": len(model.parents)}), file=sys.stderr)
            if rpc is not None:
                print(json.dumps(rpc.events[-12:])[-12000:], file=sys.stderr)
            print((root / "app-server.log").read_text(encoding="utf-8", errors="replace")[-12000:],
                  file=sys.stderr)
            raise
        finally:
            model.release_child.set()
            model.release_reuse.set()
            try:
                if rpc is not None:
                    rpc.close()
            finally:
                model.shutdown()
                model.server_close()
                server.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex", required=True, type=Path, help="absolute packaged binary path")
    parser.add_argument("--version", choices=("v1", "v2", "both"), default="both")
    parser.add_argument("--scenario", choices=("continue", "reuse", "unsubscribe", "steer", "interrupt", "all"), default="all")
    parser.add_argument("--idle-seconds", type=float, default=2,
                        help="bounded observation window for unwanted restarts (default: 2)")
    args = parser.parse_args()
    if not args.codex.is_absolute() or not args.codex.is_file():
        parser.error("--codex must name an existing absolute binary path")
    if not 0 < args.idle_seconds <= 30:
        parser.error("--idle-seconds must be in (0, 30]")
    versions = ("v1", "v2") if args.version == "both" else (args.version,)
    scenarios = ("continue", "reuse", "unsubscribe", "steer", "interrupt") if args.scenario == "all" else (args.scenario,)
    for version in versions:
        for scenario in scenarios:
            print(json.dumps(run_case(args.codex, version, scenario, args.idle_seconds)), flush=True)


if __name__ == "__main__":
    main()
