import base64
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.client import RemoteDisconnected
from io import BytesIO
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from PIL import Image

import ai_analyzer
import llama_cpp_decision
import stream_camera


URL = "http://example.test:8080/v1/systemone"
CHOICE = {
    "activity": {
        "type": "choice",
        "choice": "drinking",
        "probabilities": {
            "drinking": 0.7,
            "faucet": 0.1,
            "microwave": 0.1,
            "other": 0.1,
        },
    }
}
BINARY = {
    "drinking": {"type": "noul", "noul": 0.8},
    "faucet": {"type": "noul", "noul": 0.2},
    "microwave": {"type": "noul", "noul": 0.6},
}


class Response:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def read(self):
        return self.value


class LlamaCppDecisionTests(unittest.TestCase):
    def test_choice_and_binary_requests_and_scores(self):
        for mode, answers in (("choice", CHOICE), ("binary", BINARY)):
            with self.subTest(mode=mode), patch.dict(
                os.environ,
                {
                    "AI_PROVIDER": "llama_cpp",
                    "LLAMA_CPP_URL": URL,
                    "LLAMA_CPP_API_KEY": "test-key",
                    "LLAMA_CPP_TIMEOUT_SECONDS": "75",
                    "MODEL": "ignored",
                    "QUESTION_MODE": mode,
                },
                clear=True,
            ):
                analyzer = ai_analyzer.create_analyzer()
                self.assertIsInstance(analyzer, llama_cpp_decision.LlamaCppDecisionAnalyzer)
                response = Response(json.dumps({"answers": answers}).encode())
                with patch.object(llama_cpp_decision, "urlopen", return_value=response) as send:
                    scores = analyzer.analyze_frame(b"jpeg", True)
                request = send.call_args.args[0]
                payload = json.loads(request.data)
                self.assertEqual(request.full_url, URL)
                self.assertEqual(request.get_method(), "POST")
                self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
                self.assertEqual(send.call_args.kwargs["timeout"], 75)
                self.assertIsNone(payload["state"])
                self.assertEqual(payload["images"], ["data:image/jpeg;base64,anBlZw=="])
                self.assertNotIn("model", payload)
                self.assertEqual(payload["questions"], ai_analyzer.build_questions(True))
                if mode == "choice":
                    self.assertEqual((scores.drinking, scores.faucet, scores.microwave, scores.choice),
                                     (0.7, 0.1, 0.1, "drinking"))
                else:
                    self.assertEqual((scores.drinking, scores.faucet, scores.microwave, scores.choice),
                                     (0.8, 0.2, 0.6, None))

    def test_image_resize_and_person_gate(self):
        with patch.dict(os.environ, {"LLAMA_CPP_URL": URL}, clear=True):
            analyzer = llama_cpp_decision.LlamaCppDecisionAnalyzer()
            source = BytesIO()
            Image.new("RGB", (1920, 1080), "blue").save(source, format="JPEG")
            response = Response(json.dumps({"answers": CHOICE}).encode())
            with patch.object(llama_cpp_decision, "urlopen", return_value=response) as send:
                stream_camera.analyze_captured_frame(analyzer, source.getvalue(), True)
                payload = json.loads(send.call_args.args[0].data)
                encoded = payload["images"][0].split(",", 1)[1]
                with Image.open(BytesIO(base64.b64decode(encoded))) as image:
                    self.assertEqual(image.size, (1280, 720))
                self.assertIsNone(send.call_args.args[0].get_header("Authorization"))
                send.reset_mock()
                with self.assertRaises(ai_analyzer.AnalyzerError):
                    analyzer.analyze_frame(b"jpeg", False)
                send.assert_not_called()

    def test_invalid_configuration(self):
        for url in ("", "http://REPLACE_SERVER_HOST:8080/v1/systemone",
                    "ftp://example.test/v1/systemone", "http://example.test",
                    "http://user:pass@example.test/v1/systemone",
                    "http://@example.test/v1/systemone",
                    "http://example.test/v1/systemone?secret=x",
                    "http://example.test:abc/v1/systemone", "http://bad host/v1/systemone"):
            with self.subTest(url=url), patch.dict(os.environ, {"LLAMA_CPP_URL": url}, clear=True):
                with self.assertRaisesRegex(ai_analyzer.AnalyzerError, "LLAMA_CPP_URL"):
                    llama_cpp_decision.LlamaCppDecisionAnalyzer()
        for timeout in ("", "0", "-1", "nan", "inf", "abc"):
            with self.subTest(timeout=timeout), patch.dict(
                os.environ, {"LLAMA_CPP_URL": URL, "LLAMA_CPP_TIMEOUT_SECONDS": timeout}, clear=True
            ):
                with self.assertRaisesRegex(ai_analyzer.AnalyzerError, "LLAMA_CPP_TIMEOUT_SECONDS"):
                    llama_cpp_decision.LlamaCppDecisionAnalyzer()
        for key in ("REPLACE_TOKEN", "token\nextra"):
            with self.subTest(key=key), patch.dict(
                os.environ, {"LLAMA_CPP_URL": URL, "LLAMA_CPP_API_KEY": key}, clear=True
            ):
                with self.assertRaisesRegex(ai_analyzer.AnalyzerError, "LLAMA_CPP_API_KEY"):
                    llama_cpp_decision.LlamaCppDecisionAnalyzer()

    def test_invalid_answers_and_network_errors(self):
        invalid_responses = [
            b"not json", b"\xff", b"[]", b'{"error": "failed"}',
            b'{"answers": {}}', json.dumps({"answers": BINARY}).encode(),
            json.dumps({"answers": {"activity": {
                "type": "choice", "choice": "drinking", "probabilities": {"drinking": 0.9}
            }}}).encode(),
        ]
        with patch.dict(os.environ, {"LLAMA_CPP_URL": URL}, clear=True):
            analyzer = llama_cpp_decision.LlamaCppDecisionAnalyzer()
            for body in invalid_responses:
                with self.subTest(body=body), patch.object(
                    llama_cpp_decision, "urlopen", return_value=Response(body)
                ):
                    with self.assertRaises(ai_analyzer.AnalyzerError):
                        analyzer.analyze_frame(b"jpeg", True)
            for error in (HTTPError(URL, 401, "unauthorized", {}, None),
                          URLError("unreachable"), TimeoutError("timed out"),
                          RemoteDisconnected("disconnected")):
                with self.subTest(error=error), patch.object(
                    llama_cpp_decision, "urlopen", side_effect=error
                ):
                    with self.assertRaises(ai_analyzer.AnalyzerError):
                        analyzer.analyze_frame(b"jpeg", True)

    def test_real_http_request_and_redirect_rejection(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                requests.append((self.path, self.headers.get("Authorization"), json.loads(body)))
                if self.path.startswith("/redirect/"):
                    self.send_response(307)
                    self.send_header("Location", "/v1/systemone")
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"answers": CHOICE}).encode())

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with patch.dict(os.environ, {"LLAMA_CPP_URL": base + "/v1/systemone"}, clear=True):
                scores = llama_cpp_decision.LlamaCppDecisionAnalyzer().analyze_frame(b"jpeg", True)
            self.assertEqual(scores.drinking, 0.7)
            self.assertEqual(requests[0][0], "/v1/systemone")
            self.assertIsNone(requests[0][1])
            with patch.dict(os.environ, {
                "LLAMA_CPP_URL": base + "/redirect/v1/systemone", "LLAMA_CPP_API_KEY": "key"
            }, clear=True):
                with self.assertRaisesRegex(ai_analyzer.AnalyzerError, "HTTP 307"):
                    llama_cpp_decision.LlamaCppDecisionAnalyzer().analyze_frame(b"jpeg", True)
            self.assertEqual(len(requests), 2)
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
