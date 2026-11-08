import base64
import json
import os
import tempfile
import unittest
from io import BytesIO, StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from PIL import Image

import ai_analyzer
import azure_blob
import cloudflare_clef
import openrouter_decision
import stream_camera


class Response:
    status = 201

    def __init__(self, value=None):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        pass

    def read(self):
        return json.dumps(self.value).encode()


class DetectionTests(unittest.TestCase):
    def test_ai_frame_size_and_original_upload(self):
        class FakeUploader:
            def __init__(self):
                self.image = None

            def upload_then_delete(self, path, categories):
                self.image = path.read_bytes()
                path.unlink()

        env = {
            "CLOUDFLARE_ACCOUNT_ID": "account",
            "CLOUDFLARE_API_TOKEN": "token",
            "OPENROUTER_API_KEY": "key",
            "QUESTION_MODE": "binary",
        }
        cases = [
            ((1920, 1080), (1280, 720), (1280, 720)),
            ((1920, 1440), (1280, 720), (960, 720)),
            ((1080, 1920), (1280, 720), (405, 720)),
            ((640, 480), (1280, 720), (640, 480)),
            ((1920, 1080), (640, 360), (640, 360)),
            ((1920, 1080), (1920, 1080), (1920, 1080)),
            ((640, 480), (1920, 1080), (640, 480)),
        ]
        with patch.dict(os.environ, env), tempfile.TemporaryDirectory() as directory:
            providers = [
                (cloudflare_clef, cloudflare_clef.CloudflareClefAnalyzer(), True),
                (openrouter_decision, openrouter_decision.OpenRouterDecisionAnalyzer(), False),
            ]
            for input_size, max_size, expected_size in cases:
                source = BytesIO()
                Image.new("RGB", input_size, "blue").save(source, format="JPEG")
                original = source.getvalue()
                for module, analyzer, wrapped in providers:
                    with self.subTest(provider=module.__name__, size=input_size, max_size=max_size):
                        result = {"answers": {
                            "faucet": {"type": "noul", "noul": 0.9},
                            "drinking": {"type": "noul", "noul": 0.1},
                        }}
                        if wrapped:
                            result = {"success": True, "result": result}
                        settings = {
                            "AI_IMAGE_MAX_WIDTH": str(max_size[0]),
                            "AI_IMAGE_MAX_HEIGHT": str(max_size[1]),
                        }
                        with patch.dict(os.environ, settings), \
                             patch.object(module, "urlopen", return_value=Response(result)) as send:
                            scores = stream_camera.analyze_captured_frame(
                                analyzer, original, True, stream_camera.get_ai_image_max_size()
                            )
                        payload = json.loads(send.call_args.args[0].data)
                        if wrapped:
                            encoded = payload["images"][0]["base64"]
                        else:
                            encoded = payload["state"][1]["image_url"]["url"].split(",", 1)[1]
                        ai_image = base64.b64decode(encoded)
                        with Image.open(BytesIO(ai_image)) as decoded:
                            self.assertEqual(decoded.size, expected_size)
                        if input_size == expected_size:
                            self.assertEqual(ai_image, original)
                        uploader = FakeUploader()
                        stream_camera.handle_detections(
                            original, scores, Path(directory), uploader
                        )
                        self.assertEqual(uploader.image, original)

    def test_ai_image_size_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(stream_camera.get_ai_image_max_size(), (1280, 720))
            for name in ("AI_IMAGE_MAX_WIDTH", "AI_IMAGE_MAX_HEIGHT"):
                for value in ("0", "-1", "1.5", "", "abc"):
                    with self.subTest(name=name, value=value):
                        with patch.dict(os.environ, {name: value}):
                            with self.assertRaisesRegex(ValueError, name):
                                stream_camera.get_ai_image_max_size()

    def test_invalid_ai_image_configuration_stops_startup(self):
        with patch.dict(os.environ, {"AI_IMAGE_MAX_WIDTH": "0"}), \
             patch.object(stream_camera, "load_dotenv"), \
             patch.object(stream_camera, "create_analyzer") as create, \
             patch("sys.stderr", new_callable=StringIO) as stderr:
            self.assertEqual(stream_camera.run(), 2)
            self.assertIn("AI_IMAGE_MAX_WIDTH", stderr.getvalue())
            create.assert_not_called()

    def test_invalid_frame_reports_analysis_error(self):
        with self.assertRaises(ai_analyzer.AnalyzerError):
            stream_camera.analyze_captured_frame(None, b"not a jpeg", False)

    def test_both_providers_select_questions_and_parse_scores(self):
        env = {
            "CLOUDFLARE_ACCOUNT_ID": "account",
            "CLOUDFLARE_API_TOKEN": "token",
            "OPENROUTER_API_KEY": "key",
            "QUESTION_MODE": "binary",
        }
        with patch.dict(os.environ, env):
            providers = [
                (cloudflare_clef, cloudflare_clef.CloudflareClefAnalyzer(), True),
                (openrouter_decision, openrouter_decision.OpenRouterDecisionAnalyzer(), False),
            ]
            for module, analyzer, wrapped in providers:
                for person in (False, True):
                    with self.subTest(provider=module.__name__, person=person):
                        if not person:
                            with patch.object(module, "urlopen") as send:
                                with self.assertRaises(ai_analyzer.AnalyzerError):
                                    analyzer.analyze_frame(b"jpeg", False)
                                send.assert_not_called()
                            continue
                        answers = {
                            "faucet": {"type": "noul", "noul": 0.8},
                            "drinking": {"type": "noul", "noul": 0.7},
                        }
                        result = {"answers": answers}
                        if wrapped:
                            result = {"success": True, "result": result}
                        with patch.object(module, "urlopen", return_value=Response(result)) as send:
                            scores = analyzer.analyze_frame(b"jpeg", person)
                        self.assertEqual(scores.faucet, 0.8)
                        self.assertEqual(scores.drinking, 0.7)
                        request = json.loads(send.call_args.args[0].data)
                        image_data = base64.b64encode(b"jpeg").decode("ascii")
                        if wrapped:
                            self.assertEqual(request["images"][0]["base64"], image_data)
                        else:
                            self.assertEqual(
                                request["state"][1]["image_url"]["url"],
                                f"data:image/jpeg;base64,{image_data}",
                            )
                        self.assertEqual(
                            set(request["questions"]),
                            {"faucet", "drinking"},
                        )
                        faucet_prompt = request["questions"]["faucet"]["instructions"]
                        self.assertEqual(faucet_prompt, ai_analyzer.FAUCET_INSTRUCTIONS)
                        self.assertIn("water is visibly flowing", faucet_prompt)
                        self.assertIn("unattended running tap do not qualify", faucet_prompt)
                        self.assertIn("no clear interaction", faucet_prompt)
                        self.assertIn("kettle or water boiler alone does not qualify", faucet_prompt)
                        self.assertEqual(
                            request["questions"]["drinking"]["instructions"],
                            ai_analyzer.DRINKING_INSTRUCTIONS,
                        )

    def test_capture_loop_only_analyzes_when_yolo_detects_person(self):
        for detection in (False, True, stream_camera.PersonFilterError("failed")):
            with self.subTest(detection=detection):
                stream_camera.stop_event.clear()
                self.addCleanup(stream_camera.stop_event.clear)
                person_filter = MagicMock()
                if isinstance(detection, Exception):
                    person_filter.detects_person.side_effect = detection
                else:
                    person_filter.detects_person.return_value = detection
                process = MagicMock()
                process.poll.side_effect = [None, 0]
                frames = MagicMock()
                frames.get.return_value = b"jpeg"
                frames.get_nowait.side_effect = stream_camera.queue.Empty
                analyzer = MagicMock()
                uploader = MagicMock()
                scores = ai_analyzer.DetectionScores(0.7, 0.8)
                with patch.dict(os.environ, {
                         "CAMERA_RTSP_URL": "rtsp://example",
                         "AI_IMAGE_MAX_WIDTH": "640",
                         "AI_IMAGE_MAX_HEIGHT": "360",
                     }), \
                     patch.object(stream_camera, "load_dotenv"), \
                     patch.object(stream_camera.shutil, "which", return_value="ffmpeg"), \
                     patch.object(stream_camera, "create_analyzer", return_value=analyzer), \
                     patch.object(stream_camera, "PersonFilter", return_value=person_filter), \
                     patch.object(stream_camera, "AzureBlobUploader", return_value=uploader), \
                     patch.object(stream_camera.signal, "signal"), \
                     patch.object(stream_camera.subprocess, "Popen", return_value=process), \
                     patch.object(stream_camera.queue, "Queue", return_value=frames), \
                     patch.object(stream_camera.threading, "Thread"), \
                     patch.object(stream_camera.stop_event, "wait", side_effect=lambda _: stream_camera.stop_event.set()), \
                     patch.object(stream_camera, "analyze_captured_frame", return_value=scores) as analyze, \
                     patch.object(stream_camera, "handle_detections") as handle:
                    self.assertEqual(stream_camera.run(), 0)
                person_filter.detects_person.assert_called_once_with(b"jpeg")
                if detection is True:
                    analyze.assert_called_once_with(analyzer, b"jpeg", True, (640, 360))
                    handle.assert_called_once()
                else:
                    analyze.assert_not_called()
                    handle.assert_not_called()

    def test_invalid_or_missing_scores_are_errors(self):
        for value in (None, True, 1.1, -0.1, float("nan")):
            with self.subTest(value=value):
                with self.assertRaises(ai_analyzer.AnalyzerError):
                    ai_analyzer.parse_scores(
                        {"faucet": {"type": "noul", "noul": value}},
                        False,
                        "Provider",
                    )
        with self.assertRaises(ai_analyzer.AnalyzerError):
            ai_analyzer.parse_scores(
                {"faucet": {"type": "noul", "noul": 0.9}}, True, "Provider"
            )

    def test_upload_paths_and_delete_only_when_all_succeed(self):
        env = {
            "AZURE_BLOB_CONTAINER_URL": "https://example.blob.core.windows.net/container",
            "AZURE_BLOB_SAS_TOKEN": "sp=cw&sig=fake",
        }
        with patch.dict(os.environ, env), tempfile.TemporaryDirectory() as directory:
            uploader = azure_blob.AzureBlobUploader()
            image = Path(directory) / "frame.jpg"
            image.write_bytes(b"jpeg")
            with patch.object(azure_blob, "urlopen", return_value=Response()) as send:
                uploader.upload_then_delete(image, ["drinking", "faucet"])
            urls = [call.args[0].full_url.split("?")[0] for call in send.call_args_list]
            self.assertEqual(
                urls,
                [
                    "https://example.blob.core.windows.net/container/drinking/frame.jpg",
                    "https://example.blob.core.windows.net/container/faucet/frame.jpg",
                ],
            )
            self.assertFalse(image.exists())

            image.write_bytes(b"jpeg")
            error = HTTPError("https://example.invalid", 403, "forbidden", {}, None)
            with patch.object(azure_blob, "urlopen", side_effect=[error, Response()]) as send:
                with self.assertRaises(azure_blob.AzureUploadError):
                    uploader.upload_then_delete(image, ["drinking", "faucet"])
                self.assertEqual(send.call_count, 2)
            self.assertEqual(image.read_bytes(), b"jpeg")

    def test_strict_threshold_and_category_routing(self):
        class FakeUploader:
            def __init__(self):
                self.calls = []

            def upload_then_delete(self, path, categories):
                self.calls.append((path.read_bytes(), list(categories)))
                path.unlink()

        with tempfile.TemporaryDirectory() as directory:
            uploader = FakeUploader()
            cases = [
                (ai_analyzer.DetectionScores(0.5, 0.5), None),
                (ai_analyzer.DetectionScores(0.6, 0.5), ["drinking"]),
                (ai_analyzer.DetectionScores(None, 0.6), ["faucet"]),
                (ai_analyzer.DetectionScores(0.6, 0.7), ["drinking", "faucet"]),
            ]
            for scores, expected in cases:
                with self.subTest(scores=scores):
                    previous = len(uploader.calls)
                    stream_camera.handle_detections(
                        b"jpeg", scores, Path(directory), uploader
                    )
                    if expected is None:
                        self.assertEqual(len(uploader.calls), previous)
                    else:
                        self.assertEqual(uploader.calls[-1], (b"jpeg", expected))


    def test_choice_mode_sends_one_exclusive_question(self):
        env = {
            "CLOUDFLARE_ACCOUNT_ID": "account",
            "CLOUDFLARE_API_TOKEN": "token",
            "OPENROUTER_API_KEY": "key",
            "QUESTION_MODE": "choice",
        }
        with patch.dict(os.environ, env):
            providers = [
                (cloudflare_clef, cloudflare_clef.CloudflareClefAnalyzer(), True),
                (openrouter_decision, openrouter_decision.OpenRouterDecisionAnalyzer(), False),
            ]
            for module, analyzer, wrapped in providers:
                with self.subTest(provider=module.__name__):
                    answers = {
                        "activity": {
                            "type": "choice",
                            "choice": "drinking",
                            "probabilities": {
                                "drinking": 0.82,
                                "faucet": 0.11,
                                "other": 0.07,
                            },
                        }
                    }
                    result = {"answers": answers}
                    if wrapped:
                        result = {"success": True, "result": result}
                    with patch.object(module, "urlopen", return_value=Response(result)) as send:
                        scores = analyzer.analyze_frame(b"jpeg", True)
                    self.assertEqual(scores.drinking, 0.82)
                    self.assertEqual(scores.faucet, 0.11)
                    request = json.loads(send.call_args.args[0].data)
                    self.assertEqual(set(request["questions"]), {"activity"})
                    question = request["questions"]["activity"]
                    self.assertEqual(question["type"], "choice")
                    self.assertEqual(
                        set(question["criteria"]), {"drinking", "faucet", "other"}
                    )
                    self.assertEqual(
                        question["instructions"], ai_analyzer.CHOICE_INSTRUCTIONS
                    )

    def test_choice_probabilities_never_produce_two_categories(self):
        for drinking, faucet in ((0.51, 0.49), (0.49, 0.51), (0.34, 0.33), (0.9, 0.05)):
            with self.subTest(drinking=drinking, faucet=faucet):
                scores = ai_analyzer.parse_scores(
                    {
                        "activity": {
                            "type": "choice",
                            "choice": "drinking" if drinking > faucet else "faucet",
                            "probabilities": {
                                "drinking": drinking,
                                "faucet": faucet,
                                "other": round(1 - drinking - faucet, 2),
                            },
                        }
                    },
                    True,
                    "Provider",
                )
                categories = [
                    name
                    for name, score in (
                        ("drinking", scores.drinking),
                        ("faucet", scores.faucet),
                    )
                    if score is not None and score > stream_camera.POSITIVE_THRESHOLD
                ]
                self.assertLessEqual(len(categories), 1)

    def test_choice_label_without_probabilities_uses_the_label(self):
        cases = [
            ("drinking", (1.0, 0.0)),
            ("faucet", (0.0, 1.0)),
            ("other", (0.0, 0.0)),
        ]
        for label, expected in cases:
            with self.subTest(label=label):
                scores = ai_analyzer.parse_scores(
                    {"activity": {"type": "choice", "choice": label}}, True, "Provider"
                )
                self.assertEqual((scores.drinking, scores.faucet), expected)

    def test_invalid_choice_answers_are_errors(self):
        cases = [
            {"type": "noul", "noul": 0.9},
            {"type": "choice", "choice": "washing"},
            {"type": "choice", "probabilities": {"drinking": 1.4, "faucet": 0.0}},
            {"type": "choice"},
        ]
        for answer in cases:
            with self.subTest(answer=answer):
                with self.assertRaises(ai_analyzer.AnalyzerError):
                    ai_analyzer.parse_scores({"activity": answer}, True, "Provider")

    def test_question_mode_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ai_analyzer.get_question_mode(), "choice")
        with patch.dict(os.environ, {"QUESTION_MODE": "binary"}):
            self.assertEqual(ai_analyzer.get_question_mode(), "binary")
        with patch.dict(os.environ, {"QUESTION_MODE": "CHOICE"}):
            self.assertEqual(ai_analyzer.get_question_mode(), "choice")
        with patch.dict(os.environ, {"QUESTION_MODE": "bogus"}):
            with self.assertRaisesRegex(ai_analyzer.AnalyzerError, "QUESTION_MODE"):
                ai_analyzer.get_question_mode()

    def test_upload_cooldown_suppresses_repeated_categories(self):
        class FakeUploader:
            def __init__(self):
                self.calls = []

            def upload_then_delete(self, path, categories):
                self.calls.append(list(categories))
                path.unlink()

        with tempfile.TemporaryDirectory() as directory:
            uploader = FakeUploader()
            state = {}
            both = ai_analyzer.DetectionScores(0.7, 0.8)
            stream_camera.handle_detections(
                b"jpeg", both, Path(directory), uploader, 10.0, state
            )
            self.assertEqual(uploader.calls, [["drinking", "faucet"]])
            stream_camera.handle_detections(
                b"jpeg", both, Path(directory), uploader, 10.0, state
            )
            self.assertEqual(len(uploader.calls), 1)
            state["drinking"] -= 100
            stream_camera.handle_detections(
                b"jpeg", both, Path(directory), uploader, 10.0, state
            )
            self.assertEqual(uploader.calls[-1], ["drinking"])

    def test_capture_and_request_rate_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(stream_camera.get_capture_fps(), 2)
            self.assertEqual(stream_camera.get_ai_request_interval(), 2.0)
        with patch.dict(
            os.environ,
            {
                stream_camera.CAPTURE_FPS_ENV: "1",
                stream_camera.AI_REQUEST_INTERVAL_ENV: "0.5",
            },
        ):
            self.assertEqual(stream_camera.get_capture_fps(), 1)
            self.assertEqual(stream_camera.get_ai_request_interval(), 0.5)
        for value in ("0", "-2", "abc", "2.5"):
            with self.subTest(fps=value):
                with patch.dict(os.environ, {stream_camera.CAPTURE_FPS_ENV: value}):
                    with self.assertRaisesRegex(ValueError, "CAPTURE_FPS"):
                        stream_camera.get_capture_fps()
        for value in ("0", "-1", "abc", "nan", "inf"):
            with self.subTest(interval=value):
                with patch.dict(
                    os.environ, {stream_camera.AI_REQUEST_INTERVAL_ENV: value}
                ):
                    with self.assertRaisesRegex(
                        ValueError, "AI_REQUEST_INTERVAL_SECONDS"
                    ):
                        stream_camera.get_ai_request_interval()

    def test_upload_cooldown_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(stream_camera.get_upload_cooldown(), 0.0)
        with patch.dict(os.environ, {stream_camera.UPLOAD_COOLDOWN_ENV: "12.5"}):
            self.assertEqual(stream_camera.get_upload_cooldown(), 12.5)
        for value in ("-1", "abc", "nan", "inf"):
            with self.subTest(value=value):
                with patch.dict(
                    os.environ, {stream_camera.UPLOAD_COOLDOWN_ENV: value}
                ):
                    with self.assertRaisesRegex(ValueError, "UPLOAD_COOLDOWN_SECONDS"):
                        stream_camera.get_upload_cooldown()


if __name__ == "__main__":
    unittest.main()
