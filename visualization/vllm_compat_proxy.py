#!/usr/bin/env python3
"""Tiny HTTP proxy between rrt_echo and our vLLM, so the author's code runs unmodified (09-16).

Why: `perception/correspondence.py` sends `"structured_outputs": {"disable_any_whitespace": true}` next to
`response_format.json_schema`. vLLM 0.19.1 validates `structured_outputs` and rejects it when it carries no
constraint ("You must use one kind of structured outputs constraint but none are specified") -> every identity
comparison batch died with HTTP 400 (24/24 on the 3-min clip). The joint-observation calls (vlm.py) don't send that
field and were fine. Verified 09-16 with /tmp/probe.py: the same request minus `structured_outputs` is accepted.

What it does: forwards every request to the upstream unchanged, except that a JSON body whose `structured_outputs`
has none of json/regex/choice/grammar/json_object/structural_tag gets that key removed (response_format is kept, so
the schema still constrains the output). Nothing else is touched; the response is streamed back verbatim.

    python vllm_compat_proxy.py --listen 8010 --upstream http://127.0.0.1:8000
then run rrt_echo with --base-url http://127.0.0.1:8010/v1
"""

import argparse
import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CONSTRAINTS = ("json", "regex", "choice", "grammar", "json_object", "structural_tag")
UPSTREAM = "http://127.0.0.1:8000"
STATS = {"requests": 0, "rewritten": 0}


def rewrite(raw):
    try:
        body = json.loads(raw)
    except Exception:
        return raw, False
    so = body.get("structured_outputs")
    if isinstance(so, dict) and not any(so.get(k) is not None for k in CONSTRAINTS):
        body.pop("structured_outputs")
        return json.dumps(body).encode(), True
    return raw, False


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _forward(self, data):
        STATS["requests"] += 1
        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in ("host", "content-length", "connection")
        }
        if data is not None:
            data, changed = rewrite(data)
            STATS["rewritten"] += int(changed)
            headers["Content-Length"] = str(len(data))
        req = urllib.request.Request(
            UPSTREAM + self.path, data=data, headers=headers, method=self.command
        )
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                payload = r.read()
                self._reply(r.status, r.headers.get("Content-Type", "application/json"), payload)
        except urllib.error.HTTPError as e:
            self._reply(e.code, e.headers.get("Content-Type", "application/json"), e.read())
        except Exception as e:  # upstream down etc.
            self._reply(
                502, "application/json", json.dumps({"error": {"message": f"proxy: {e}"}}).encode()
            )
        if STATS["requests"] % 20 == 0:
            print("proxy stats", STATS, flush=True)

    def _reply(self, code, ctype, payload):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        self._forward(self.rfile.read(n) if n else b"")

    def do_GET(self):
        self._forward(None)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=8010)
    ap.add_argument("--upstream", default=UPSTREAM)
    a = ap.parse_args()
    UPSTREAM = a.upstream.rstrip("/")
    print(f"proxy :{a.listen} -> {UPSTREAM}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.listen), H).serve_forever()
