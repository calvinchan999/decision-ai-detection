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
                            "boilwater": {"type": "noul", "noul": 0.9},
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
                            "boilwater": {"type": "noul", "noul": 0.8},
                            "drinking": {"type": "noul", "noul": 0.7},
                        }
                        result = {"answers": answers}
                        if wrapped:
                            result = {"success": True, "result": result}
                        with patch.object(module, "urlopen", return_value=Response(result)) as send:
                            scores = analyzer.analyze_frame(b"jpeg", person)
                        self.assertEqual(scores.boilwater, 0.8)
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
                            {"boilwater", "drinking"},
                        )
                        tap_prompt = request["questions"]["boilwater"]["instructions"]
                        self.assertEqual(tap_prompt, ai_analyzer.BOILWATER_INSTRUCTIONS)
                        self.assertIn("water is visibly flowing", tap_prompt)
                        self.assertIn("unattended running tap do not qualify", tap_prompt)
                        self.assertIn("no clear interaction", tap_prompt)
                        self.assertIn("kettle or water boiler alone does not qualify", tap_prompt)
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
                        {"boilwater": {"type": "noul", "noul": value}},
                        False,
                        "Provider",
                    )
        with self.assertRaises(ai_analyzer.AnalyzerError):
            ai_analyzer.parse_scores(
                {"boilwater": {"type": "noul", "noul": 0.9}}, True, "Provider"
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
                uploader.upload_then_delete(image, ["drinking", "boilwater"])
            urls = [call.args[0].full_url.split("?")[0] for call in send.call_args_list]
            self.assertEqual(
                urls,
                [
                    "https://example.blob.core.windows.net/container/drinking/frame.jpg",
                    "https://example.blob.core.windows.net/container/boilwater/frame.jpg",
                ],
            )
            self.assertFalse(image.exists())

            image.write_bytes(b"jpeg")
            error = HTTPError("https://example.invalid", 403, "forbidden", {}, None)
            with patch.object(azure_blob, "urlopen", side_effect=[error, Response()]) as send:
                with self.assertRaises(azure_blob.AzureUploadError):
                    uploader.upload_then_delete(image, ["drinking", "boilwater"])
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
                (ai_analyzer.DetectionScores(None, 0.6), ["boilwater"]),
                (ai_analyzer.DetectionScores(0.6, 0.7), ["drinking", "boilwater"]),
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


if __name__ == "__main__":
    unittest.main()
