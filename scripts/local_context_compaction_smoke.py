"""Exercise packaged Codex local compaction against an isolated deterministic model.

Usage: python scripts/local_context_compaction_smoke.py --codex PATH/TO/codex.exe
Requires Python 3.11+. No credentials or external model service are used.
"""

import argparse
import http.server
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time


class SmokeDirectory(tempfile.TemporaryDirectory):
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


def analysis_payload(body, marker):
    for item in reversed(body.get("input", [])):
        for content in item.get("content", []):
            text = content.get("text", "")
            if text.startswith(marker):
                return json.loads(text[len(marker) :].strip())
    return None


def assistant(text):
    return {
        "type": "message",
        "id": "fixture-assistant",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


class Model(http.server.ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.requests = []
        self.decisions = []
        self.phase = "tools"
        self.recall_id = None
        self.errors = []
        self.mark_started = threading.Event()
        self.release_mark = threading.Event()
        self.mark_sent = threading.Event()
        self.sequence = 0
        self.sequence_lock = threading.Lock()

    def respond(self, body):
        self.requests.append(body)
        payload = analysis_payload(body, "LOCAL_COMPACTION_CLASSIFY")
        if payload is not None:
            visible_calls = {
                item.get("call_id")
                for item in body.get("input", [])
                if item.get("type") == "function_call_output"
            }
            candidates = payload["candidates"]
            assert {c["call_id"] for c in candidates} <= visible_calls, (
                "candidates must name model-visible tool results"
            )
            assert body.get("tools"), "classifier must repeat the ordinary tool prefix"
            assert "LOCAL_COMPACTION_SOURCE" not in json.dumps(body), (
                "classifier must not rewrite the shared prefix"
            )
            decisions = []
            for index, candidate in enumerate(candidates):
                action = ["keep", "shorten", "drop"][index % 3]
                decision = {"id": candidate["id"], "action": action}
                if action == "shorten":
                    decision["text"] = (
                        "Fixture tool was unavailable; no command was executed."
                    )
                decisions.append(decision)
            self.decisions.extend(decisions)
            self.mark_started.set()
            assert self.release_mark.wait(30), "marking gate was not released"
            return [assistant(json.dumps({"decisions": decisions}))]
        if analysis_payload(body, "LOCAL_COMPACTION_SUMMARIZE") is not None:
            return [
                assistant(
                    "Earlier fixture work used unavailable tools; verification is pending. "
                    "Use only the isolated fixture and preserve the offline-only constraint."
                )
            ]
        if self.phase == "tools":
            self.phase = "answer"
            return [
                {
                    "type": "function_call",
                    "id": f"fixture-call-{i}",
                    "call_id": f"fixture-{i}",
                    "name": f"unavailable_{i}_" + "evidence_" * 3600,
                    "arguments": "{}",
                }
                for i in range(4)
            ]
        if self.phase == "recall":
            self.phase = "answer"
            return [
                {
                    "type": "function_call",
                    "id": "fixture-recall-call",
                    "call_id": "fixture-recall",
                    "name": "recall_read_item",
                    "arguments": json.dumps(
                        {"item_id": self.recall_id, "max_chars": 1000}
                    ),
                }
            ]
        return [assistant("Fixture task complete; offline-only constraint preserved.")]


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        try:
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            if self.headers.get("Content-Encoding"):
                raise AssertionError(
                    "Fixture provider must use uncompressed HTTP requests"
                )
            body = json.loads(raw)
            if not self.path.endswith("/responses"):
                raise AssertionError(f"Unexpected remote service path: {self.path}")
            with self.server.sequence_lock:
                self.server.sequence += 1
                response_id = f"fixture-response-{self.server.sequence}"
            marking = analysis_payload(body, "LOCAL_COMPACTION_CLASSIFY") is not None
            output = self.server.respond(body)
            for index, item in enumerate(output):
                if item["type"] == "message":
                    item["id"] = f"{response_id}-message-{index}"
            events = [{"type": "response.created", "response": {"id": response_id}}]
            events += [
                {"type": "response.output_item.done", "output_index": i, "item": item}
                for i, item in enumerate(output)
            ]
            events.append(
                {
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "output": output,
                        "usage": {
                            "input_tokens": 700 if marking else 1000,
                            "output_tokens": 77 if marking else 100,
                            "total_tokens": 777 if marking else 1100,
                        },
                    },
                }
            )
            payload = "".join(
                "data: " + json.dumps(event) + "\n\n" for event in events
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            if analysis_payload(body, "LOCAL_COMPACTION_CLASSIFY") is not None:
                self.server.mark_sent.set()
        except Exception as error:
            self.server.errors.append(str(error))
            self.send_error(500, str(error))


class Rpc:
    def __init__(self, binary, root, model):
        env = os.environ.copy()
        for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_API_KEY"):
            env.pop(name, None)
        env["CODEX_HOME"] = str(root / "home")
        self.stderr = (root / "app-server.log").open("a", encoding="utf-8")
        config = {
            "model": '"fixture-model"',
            "model_provider": '"fixture"',
            "model_context_window": "100000",
            "model_auto_compact_token_limit": "95000",
            "local_compaction.force_local": "true",
            "local_compaction.reclaim_percent": "1",
            "local_compaction.mark_after_tokens_percent": "1",
            "local_compaction.compact_target_percent": "30",
            "model_reasoning_effort": '"low"',
            "features.code_mode": "false",
            "features.enable_request_compression": "false",
            "features.local_thread_store_compression": "false",
            "model_providers.fixture.name": '"Local compaction fixture"',
            "model_providers.fixture.base_url": json.dumps(
                f"http://127.0.0.1:{model.server_port}/v1"
            ),
            "model_providers.fixture.wire_api": '"responses"',
            "model_providers.fixture.supports_websockets": "false",
            "model_providers.fixture.requires_openai_auth": "false",
            "model_providers.fixture.request_max_retries": "0",
            "model_providers.fixture.stream_max_retries": "0",
        }
        args = [str(binary), "app-server", "--stdio"]
        for key, value in config.items():
            args.extend(["-c", f"{key}={value}"])
        self.proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            encoding="utf-8",
            env=env,
            cwd=root / "work",
        )
        self.queue = queue.Queue()
        self.pending = []
        self.next_id = 1
        threading.Thread(target=self.read, daemon=True).start()
        try:
            self.request(
                "initialize",
                {
                    "clientInfo": {"name": "local_compaction_smoke", "version": "1"},
                    "capabilities": {"experimentalApi": True},
                },
            )
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

    def wait(self, predicate):
        deadline = time.monotonic() + 180
        for index, message in enumerate(self.pending):
            if predicate(message):
                return self.pending.pop(index)
        while time.monotonic() < deadline:
            message = self.queue.get(timeout=max(0.1, deadline - time.monotonic()))
            if "readerError" in message:
                raise RuntimeError(message["readerError"])
            if predicate(message):
                return message
            if "method" in message and "id" in message:
                raise AssertionError(
                    f"Unexpected interactive request: {message['method']}"
                )
            self.pending.append(message)
        raise TimeoutError("app-server response timed out")

    def request(self, method, params):
        request_id = self.next_id
        self.next_id += 1
        self.send({"id": request_id, "method": method, "params": params})
        result = self.wait(lambda message: message.get("id") == request_id)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result["result"]

    def turn(self, thread_id, text):
        result = self.request(
            "turn/start",
            {"threadId": thread_id, "input": [{"type": "text", "text": text}]},
        )
        turn_id = result["turn"]["id"]
        event = self.wait(
            lambda message: (
                message.get("method") == "turn/completed"
                and message["params"]["turn"]["id"] == turn_id
            )
        )
        if event["params"]["turn"].get("error"):
            raise RuntimeError(event["params"]["turn"]["error"])

    def close(self):
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        self.stderr.close()


def run(binary):
    for subcommand in ("app-server", "recall"):
        subprocess.run(
            [str(binary), subcommand, "--help"], check=True, capture_output=True
        )
    with SmokeDirectory(prefix="codex-local-context-") as temporary:
        root = Path(temporary)
        (root / "home").mkdir()
        (root / "work").mkdir()
        model = Model()
        threading.Thread(target=model.serve_forever, daemon=True).start()
        rpc = None
        try:
            rpc = Rpc(binary, root, model)
            thread = rpc.request(
                "thread/start",
                {
                    "cwd": str(root / "work"),
                    "approvalPolicy": "never",
                    "sandbox": "danger-full-access",
                },
            )["thread"]
            rpc.turn(
                thread["id"],
                "Use the fixture tools. Offline-only is a strict constraint.",
            )
            rpc.turn(
                thread["id"],
                "The fixture batch is complete. Keep the offline-only constraint and continue.",
            )
            assert model.mark_started.wait(10), "background marking did not start"
            rpc.turn(thread["id"], "Continue ordinary work while marking is held.")
            assert not model.mark_sent.is_set(), (
                "marking gate did not hold the response"
            )
            assert (
                sum(
                    analysis_payload(body, "LOCAL_COMPACTION_CLASSIFY") is not None
                    for body in model.requests
                )
                == 1
            ), "multiple marking jobs ran concurrently"
            held = rpc.request("thread/toolCleanup/read", {"threadId": thread["id"]})
            assert held["status"]["enabled"] and held["status"]["marking"], held
            assert not any(
                analysis_payload(body, "LOCAL_COMPACTION_SUMMARIZE") is not None
                for body in model.requests
            ), "soft cleanup escalated into full compaction"
            foreground_usage = [
                event["params"]["tokenUsage"]
                for event in rpc.pending
                if event.get("method") == "thread/tokenUsage/updated"
                and event["params"]["threadId"] == thread["id"]
            ][-1]
            model.release_mark.set()
            assert model.mark_sent.wait(10), "marking response was not sent"
            ready = rpc.wait(
                lambda event: (
                    event.get("method") == "thread/tokenUsage/updated"
                    and event["params"]["threadId"] == thread["id"]
                    and event["params"]["tokenUsage"]["total"]["totalTokens"]
                    == foreground_usage["total"]["totalTokens"] + 777
                )
            )["params"]["tokenUsage"]
            for field in ("last", "modelContextWindow", "contextUsage"):
                assert ready[field] == foreground_usage[field], (
                    field,
                    foreground_usage,
                    ready,
                )
            # No turn runs between the ready barrier and these calls, so the manual
            # request, not a sampling boundary, installs the validated marks.
            marked = rpc.request("thread/toolCleanup/read", {"threadId": thread["id"]})
            marked = marked["status"]
            assert not marked["marking"] and marked["pendingSavingsTokens"] > 0, marked
            applied = rpc.request(
                "thread/toolCleanup/apply", {"threadId": thread["id"]}
            )
            assert applied["releasedTokens"] == marked["pendingSavingsTokens"], applied
            assert applied["status"]["pendingSavingsTokens"] == 0, applied
            rpc.turn(thread["id"], "Continue with the cleaned tool evidence.")
            ordinary = [
                body
                for body in model.requests
                if analysis_payload(body, "LOCAL_COMPACTION_CLASSIFY") is None
                and analysis_payload(body, "LOCAL_COMPACTION_SUMMARIZE") is None
            ]
            assert any(
                "recall_read_item" in str(item.get("output", ""))
                for item in ordinary[-1].get("input", [])
                if item.get("type") == "function_call_output"
            ), "manual tool cleanup was not visible to the next request"
            rpc.close()
            rpc = None
            marks = root / "home" / "local_compaction" / f"{thread['id']}.jsonl"
            persisted = sorted(
                mark["decision"]["id"]
                for line in marks.read_text(encoding="utf-8").splitlines()
                for mark in json.loads(line)
            )
            assert persisted == sorted(d["id"] for d in model.decisions), persisted
            rollouts = list((root / "home" / "sessions").rglob("rollout-*.jsonl"))
            assert len(rollouts) == 1, rollouts
            records = [
                json.loads(line)
                for line in rollouts[0].read_text(encoding="utf-8").splitlines()
            ]
            compacted = [r["payload"] for r in records if r["type"] == "compacted"]
            assert compacted, "tool cleanup did not install a checkpoint"
            assert model.decisions, "no tool classification request observed"
            original = next(
                r["payload"]
                for r in records
                if r["type"] == "response_item"
                and r["payload"].get("type") == "function_call_output"
            )
            model.recall_id = original["id"]
            query = json.dumps(
                {"action": "read_item", "item_id": model.recall_id, "max_chars": 1000}
            )
            recalled = subprocess.run(
                [
                    str(binary),
                    "recall",
                    "--rollout",
                    str(rollouts[0]),
                    "--query",
                    query,
                ],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
            result = json.loads(recalled.stdout)
            assert result["item_id"] == model.recall_id
            assert "Output omitted" not in result["text"]
            assert len(recalled.stdout.encode()) <= 8001
            model.phase = "recall"
            rpc = Rpc(binary, root, model)
            rpc.request("thread/resume", {"threadId": thread["id"]})
            resumed = rpc.request("thread/toolCleanup/read", {"threadId": thread["id"]})
            assert resumed["status"]["pendingSavingsTokens"] == 0, resumed
            rpc.turn(
                thread["id"],
                "Read the original archived evidence using local recall, then finish.",
            )
            recall_outputs = [
                item
                for body in model.requests
                for item in body.get("input", [])
                if item.get("type") == "function_call_output"
                and item.get("call_id") == "fixture-recall"
            ]
            assert recall_outputs, "model recall tool did not complete"
            assert model.recall_id in json.dumps(recall_outputs[-1])
            assert not model.errors, model.errors
            result = {
                "status": "passed",
                "requests": len(model.requests),
                "checkpoints": len(compacted),
                "classified": len(model.decisions),
                "cli_recall": True,
                "model_recall_after_resume": True,
                "foreground_completed_while_marking_held": True,
                "background_billing_preserves_foreground_occupancy": True,
                "manual_tool_cleanup_rpc": True,
                "marks_persisted": True,
            }
        except Exception:
            print(
                (root / "app-server.log").read_text(encoding="utf-8", errors="replace")[
                    -12000:
                ]
            )
            raise
        finally:
            model.release_mark.set()
            if rpc is not None:
                rpc.close()
            model.shutdown()
            model.server_close()
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codex", type=Path, required=True)
    run(parser.parse_args().codex.resolve())
