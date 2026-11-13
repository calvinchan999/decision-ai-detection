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
        with patch.dict(os.environ, env, clear=True), tempfile.TemporaryDirectory() as directory:
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
                            "microwave": {"type": "noul", "noul": 0.1},
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
        with patch.dict(os.environ, env, clear=True):
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
                            "microwave": {"type": "noul", "noul": 0.9},
                        }
                        result = {"answers": answers}
                        if wrapped:
                            result = {"success": True, "result": result}
                        with patch.object(module, "urlopen", return_value=Response(result)) as send:
                            scores = analyzer.analyze_frame(b"jpeg", person)
                        self.assertEqual(scores.faucet, 0.8)
                        self.assertEqual(scores.drinking, 0.7)
                        self.assertEqual(scores.microwave, 0.9)
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
                            {"faucet", "drinking", "microwave"},
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
                        self.assertEqual(
                            request["questions"]["microwave"]["instructions"],
                            ai_analyzer.MICROWAVE_INSTRUCTIONS,
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
                scores = ai_analyzer.DetectionScores(0.7, 0.8, 0.1)
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
        with self.assertRaisesRegex(ai_analyzer.AnalyzerError, "microwave"):
            ai_analyzer.parse_scores(
                {
                    "faucet": {"type": "noul", "noul": 0.9},
                    "drinking": {"type": "noul", "noul": 0.1},
                },
                True,
                "Provider",
            )

    def test_provider_selection_and_shared_model(self):
        env = {
            "CLOUDFLARE_ACCOUNT_ID": "account",
            "CLOUDFLARE_API_TOKEN": "token",
            "OPENROUTER_API_KEY": "key",
            "QUESTION_MODE": "binary",
        }
        cases = [
            ("cloudflare", "clef", cloudflare_clef, True),
            ("cloudflare", "clef-flash", cloudflare_clef, True),
            ("openrouter", "liquid/d1.1", openrouter_decision, False),
            ("openrouter", "example/decision-model", openrouter_decision, False),
        ]
        answers = {
            "faucet": {"type": "noul", "noul": 0.8},
            "drinking": {"type": "noul", "noul": 0.2},
            "microwave": {"type": "noul", "noul": 0.1},
        }
        for provider, model, module, wrapped in cases:
            with self.subTest(provider=provider, model=model):
                settings = dict(env, AI_PROVIDER=provider, MODEL=model)
                result = {"answers": answers}
                if wrapped:
                    result = {"success": True, "result": result}
                with patch.dict(os.environ, settings, clear=True):
                    analyzer = ai_analyzer.create_analyzer()
                    with patch.object(module, "urlopen", return_value=Response(result)) as send:
                        analyzer.analyze_frame(b"jpeg", True)
                request = send.call_args.args[0]
                self.assertEqual(json.loads(request.data)["model"], model)
                if wrapped:
                    self.assertTrue(request.full_url.endswith("/@cf/cloudflare/" + model))
                else:
                    self.assertEqual(request.full_url, "https://openrouter.ai/api/alpha/decisions")

        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(cloudflare_clef.CloudflareClefAnalyzer().model, "clef")
            self.assertEqual(openrouter_decision.OpenRouterDecisionAnalyzer().model, "liquid/d1.1")
            for model in ("", "liquid/d1.1", "../clef"):
                with self.subTest(invalid_cloudflare_model=model):
                    with patch.dict(os.environ, {"MODEL": model}):
                        with self.assertRaisesRegex(ai_analyzer.AnalyzerError, "MODEL"):
                            cloudflare_clef.CloudflareClefAnalyzer()

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
                uploader.upload_then_delete(image, ["drinking", "faucet", "microwave"])
            urls = [call.args[0].full_url.split("?")[0] for call in send.call_args_list]
            self.assertEqual(
                urls,
                [
                    "https://example.blob.core.windows.net/container/drinking/frame.jpg",
                    "https://example.blob.core.windows.net/container/faucet/frame.jpg",
                    "https://example.blob.core.windows.net/container/microwave/frame.jpg",
                ],
            )
            self.assertFalse(image.exists())

            image.write_bytes(b"jpeg")
            error = HTTPError("https://example.invalid", 403, "forbidden", {}, None)
            with patch.object(azure_blob, "urlopen", side_effect=[Response(), Response(), error]) as send:
                with self.assertRaises(azure_blob.AzureUploadError):
                    uploader.upload_then_delete(image, ["drinking", "faucet", "microwave"])
                self.assertEqual(send.call_count, 3)
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
                (ai_analyzer.DetectionScores(0.5, 0.5, 0.5), None),
                (ai_analyzer.DetectionScores(0.6, 0.5, 0.5), ["drinking"]),
                (ai_analyzer.DetectionScores(None, 0.6, 0.5), ["faucet"]),
                (ai_analyzer.DetectionScores(0.1, 0.1, 0.6), ["microwave"]),
                (ai_analyzer.DetectionScores(0.6, 0.7, 0.9), ["drinking", "faucet", "microwave"]),
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
        with patch.dict(os.environ, env, clear=True):
            providers = [
                (cloudflare_clef, cloudflare_clef.CloudflareClefAnalyzer(), True),
                (openrouter_decision, openrouter_decision.OpenRouterDecisionAnalyzer(), False),
            ]
            for module, analyzer, wrapped in providers:
                with self.subTest(provider=module.__name__):
                    answers = {
                        "activity": {
                            "type": "choice",
                            "choice": "microwave",
                            "probabilities": {
                                "drinking": 0.05,
                                "faucet": 0.11,
                                "microwave": 0.82,
                                "other": 0.02,
                            },
                        }
                    }
                    result = {"answers": answers}
                    if wrapped:
                        result = {"success": True, "result": result}
                    with patch.object(module, "urlopen", return_value=Response(result)) as send:
                        scores = analyzer.analyze_frame(b"jpeg", True)
                    self.assertEqual(scores.drinking, 0.05)
                    self.assertEqual(scores.faucet, 0.11)
                    self.assertEqual(scores.microwave, 0.82)
                    self.assertEqual(scores.choice, "microwave")
                    request = json.loads(send.call_args.args[0].data)
                    self.assertEqual(set(request["questions"]), {"activity"})
                    question = request["questions"]["activity"]
                    self.assertEqual(question["type"], "choice")
                    self.assertEqual(
                        set(question["criteria"]), {"drinking", "faucet", "microwave", "other"}
                    )
                    self.assertEqual(
                        question["instructions"], ai_analyzer.CHOICE_INSTRUCTIONS
                    )

    def test_microwave_detection_routes_to_azure_for_both_modes_and_providers(self):
        env = {
            "CLOUDFLARE_ACCOUNT_ID": "account",
            "CLOUDFLARE_API_TOKEN": "token",
            "OPENROUTER_API_KEY": "key",
            "AZURE_BLOB_CONTAINER_URL": "https://example.blob.core.windows.net/container",
            "AZURE_BLOB_SAS_TOKEN": "sp=cw&sig=fake",
        }
        cases = [
            ("microwave", "choice", {
                "activity": {
                    "type": "choice",
                    "choice": "microwave",
                    "probabilities": {
                        "drinking": 0.05, "faucet": 0.05,
                        "microwave": 0.8, "other": 0.1,
                    },
                },
            }),
            ("microwave", "binary", {
                "drinking": {"type": "noul", "noul": 0.1},
                "faucet": {"type": "noul", "noul": 0.1},
                "microwave": {"type": "noul", "noul": 0.8},
            }),
        ]
        with patch.dict(os.environ, env, clear=True), tempfile.TemporaryDirectory() as directory:
            providers = [
                (cloudflare_clef, cloudflare_clef.CloudflareClefAnalyzer(), True),
                (openrouter_decision, openrouter_decision.OpenRouterDecisionAnalyzer(), False),
            ]
            for category, mode, answers in cases:
                for module, analyzer, wrapped in providers:
                    with self.subTest(provider=module.__name__, mode=mode, category=category):
                        result = {"answers": answers}
                        if wrapped:
                            result = {"success": True, "result": result}
                        with patch.dict(os.environ, {"QUESTION_MODE": mode}), \
                             patch.object(module, "urlopen", return_value=Response(result)) as send:
                            scores = analyzer.analyze_frame(b"original image", True)
                        questions = json.loads(send.call_args.args[0].data)["questions"]
                        if mode == "choice":
                            self.assertEqual(
                                questions["activity"]["criteria"][category],
                                ai_analyzer.CHOICE_CRITERIA[category],
                            )
                        else:
                            self.assertEqual(
                                questions[category]["instructions"],
                                ai_analyzer.MICROWAVE_INSTRUCTIONS,
                            )
                        uploader = azure_blob.AzureBlobUploader()
                        with patch.object(azure_blob, "urlopen", return_value=Response()) as upload:
                            stream_camera.handle_detections(
                                b"original image", scores, Path(directory), uploader
                            )
                        request = upload.call_args.args[0]
                        upload.assert_called_once()
                        self.assertIn(f"/container/{category}/", request.full_url)
                        self.assertEqual(request.data, b"original image")
                        self.assertEqual(list(Path(directory).iterdir()), [])

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
                                "microwave": 0,
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
                        ("microwave", scores.microwave),
                    )
                    if score is not None and score > stream_camera.POSITIVE_THRESHOLD
                ]
                self.assertLessEqual(len(categories), 1)

    def test_invalid_choice_answers_are_errors(self):
        cases = [
            {"type": "noul", "noul": 0.9},
            {"type": "choice", "choice": "washing"},
            {"type": "choice", "probabilities": {"drinking": 1.4, "faucet": 0.0}},
            {"type": "choice"},
            {"type": "choice", "choice": "microwave"},
            {"type": "choice", "choice": "microwave", "probabilities": {
                "drinking": 0.1, "faucet": 0.1, "other": 0.8,
            }},
            {"type": "choice", "choice": "microwave", "probabilities": {
                "drinking": 0.05, "faucet": 0.05, "other": 0.9,
            }},
            {"type": "choice", "choice": "microwave", "probabilities": {
                "drinking": 0.9, "faucet": 0.9, "microwave": 0.0, "other": 0.0,
            }},
            {"type": "choice", "choice": "microwave", "probabilities": {
                "drinking": 0.5004, "faucet": 0.0, "microwave": 0.5004, "other": 0.0,
            }},
            {"type": "choice", "choice": "microwave", "probabilities": {
                "drinking": 0.1, "faucet": 0.1, "microwave": float("nan"), "other": 0.0,
            }},
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
            both = ai_analyzer.DetectionScores(0.7, 0.8, 0.95)
            stream_camera.handle_detections(
                b"jpeg", both, Path(directory), uploader, 10.0, state
            )
            self.assertEqual(uploader.calls, [["drinking", "faucet", "microwave"]])
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

    def test_same_choice_cooldown_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(stream_camera.get_same_choice_cooldown(), 5.0)
        for value, expected in (("0", 0.0), ("2.5", 2.5)):
            with patch.dict(os.environ, {stream_camera.AI_SAME_CHOICE_COOLDOWN_ENV: value}):
                self.assertEqual(stream_camera.get_same_choice_cooldown(), expected)
        for value in ("-1", "", "abc", "nan", "inf"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {stream_camera.AI_SAME_CHOICE_COOLDOWN_ENV: value}):
                    with self.assertRaisesRegex(ValueError, "AI_SAME_CHOICE_COOLDOWN_SECONDS"):
                        stream_camera.get_same_choice_cooldown()
        with patch.dict(os.environ, {"AI_SAME_CHOICE_COOLDOWN_SECONDS": "-1"}), \
             patch.object(stream_camera, "load_dotenv"), \
             patch.object(stream_camera, "create_analyzer") as create, \
             patch("sys.stderr", new_callable=StringIO) as stderr:
            self.assertEqual(stream_camera.run(), 2)
            self.assertIn("AI_SAME_CHOICE_COOLDOWN_SECONDS", stderr.getvalue())
            create.assert_not_called()

    def test_request_cooldown_after_repeated_choice(self):
        cases = [
            (2, 5, ["other", "other", "drinking", "microwave", "microwave", "microwave"],
             [0, 2, 8, 10, 12, 18]),
            (2, 0, ["other", "other", "other"], [0, 2, 4]),
            (2, 5, [None, None, None], [0, 2, 4]),
            (2, 5, ["other", "error", "other", "other", "other"], [0, 2, 4, 6, 12]),
            (10, 5, ["other", "other", "other"], [0, 10, 20]),
        ]
        for interval, cooldown, choices, expected in cases:
            with self.subTest(interval=interval, cooldown=cooldown, choices=choices):
                stream_camera.stop_event.clear()
                self.addCleanup(stream_camera.stop_event.clear)
                now = [0.0]
                starts = []
                process = MagicMock()
                process.poll.return_value = None
                process.returncode = 0
                frames = MagicMock()
                first_frame = [True]

                def get_frame(**_kwargs):
                    if first_frame[0]:
                        first_frame[0] = False
                    else:
                        now[0] += 0.5
                    return b"jpeg"

                frames.get.side_effect = get_frame
                frames.get_nowait.side_effect = stream_camera.queue.Empty
                person_filter = MagicMock()
                person_filter.detects_person.return_value = True
                pending = iter(choices)

                def analyze(*_args):
                    starts.append(now[0])
                    choice = next(pending)
                    now[0] += 1
                    if len(starts) == len(choices):
                        stream_camera.stop_event.set()
                    if choice == "error":
                        raise ai_analyzer.AnalyzerError("test failure")
                    return ai_analyzer.DetectionScores(0.1, 0.1, 0.1, choice)

                def wait(seconds):
                    self.fail("Request cooldown must not block YOLO processing")

                settings = {
                    "CAMERA_RTSP_URL": "rtsp://example",
                    "AI_REQUEST_INTERVAL_SECONDS": str(interval),
                    "AI_SAME_CHOICE_COOLDOWN_SECONDS": str(cooldown),
                }
                with patch.dict(os.environ, settings, clear=True), \
                     patch.object(stream_camera, "load_dotenv"), \
                     patch.object(stream_camera.shutil, "which", return_value="ffmpeg"), \
                     patch.object(stream_camera, "create_analyzer"), \
                     patch.object(stream_camera, "PersonFilter", return_value=person_filter), \
                     patch.object(stream_camera, "AzureBlobUploader"), \
                     patch.object(stream_camera.signal, "signal"), \
                     patch.object(stream_camera.subprocess, "Popen", return_value=process), \
                     patch.object(stream_camera.queue, "Queue", return_value=frames), \
                     patch.object(stream_camera.threading, "Thread"), \
                     patch.object(stream_camera.time, "monotonic", side_effect=lambda: now[0]), \
                     patch.object(stream_camera.stop_event, "wait", side_effect=wait), \
                     patch.object(stream_camera, "analyze_captured_frame", side_effect=analyze), \
                     patch.object(stream_camera, "handle_detections"):
                    self.assertEqual(stream_camera.run(), 0)
                self.assertEqual(starts, expected)

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

    def test_no_person_and_reconnect_reset_choice_cooldown(self):
        for reconnect in (False, True):
            with self.subTest(reconnect=reconnect):
                stream_camera.stop_event.clear()
                self.addCleanup(stream_camera.stop_event.clear)
                now = [0.0]
                starts = []
                frame_times = iter([0, 1, 7] if reconnect else [0, 1, 2, 3])
                process = MagicMock()
                process.returncode = 0
                process.poll.side_effect = (
                    [None, None, 0, None, None] if reconnect
                    else [None, None, None, None, None]
                )
                frames = MagicMock()

                def get_frame(**_kwargs):
                    now[0] = next(frame_times)
                    return b"jpeg"

                frames.get.side_effect = get_frame
                frames.get_nowait.side_effect = stream_camera.queue.Empty
                person_filter = MagicMock()
                person_filter.detects_person.side_effect = (
                    [True, True, True] if reconnect else [True, True, False, True]
                )

                def analyze(*_args):
                    starts.append(now[0])
                    if len(starts) == 3:
                        stream_camera.stop_event.set()
                    return ai_analyzer.DetectionScores(0.1, 0.1, 0.1, "other")

                settings = {
                    "CAMERA_RTSP_URL": "rtsp://example",
                    "AI_REQUEST_INTERVAL_SECONDS": "1",
                    "AI_SAME_CHOICE_COOLDOWN_SECONDS": "5",
                }
                with patch.dict(os.environ, settings, clear=True), \
                     patch.object(stream_camera, "load_dotenv"), \
                     patch.object(stream_camera.shutil, "which", return_value="ffmpeg"), \
                     patch.object(stream_camera, "create_analyzer"), \
                     patch.object(stream_camera, "PersonFilter", return_value=person_filter), \
                     patch.object(stream_camera, "AzureBlobUploader"), \
                     patch.object(stream_camera.signal, "signal"), \
                     patch.object(stream_camera.subprocess, "Popen", return_value=process), \
                     patch.object(stream_camera.queue, "Queue", return_value=frames), \
                     patch.object(stream_camera.threading, "Thread"), \
                     patch.object(stream_camera.time, "monotonic", side_effect=lambda: now[0]), \
                     patch.object(stream_camera.stop_event, "wait", return_value=False), \
                     patch.object(stream_camera, "analyze_captured_frame", side_effect=analyze), \
                     patch.object(stream_camera, "handle_detections"), \
                     patch("sys.stdout", new_callable=StringIO) as output:
                    self.assertEqual(stream_camera.run(), 0)
                self.assertEqual(starts, [0, 1, 7] if reconnect else [0, 1, 3])
                self.assertEqual(output.getvalue().count("Repeated choice 'other'"), 1)
                self.assertEqual(person_filter.detects_person.call_count, 3 if reconnect else 4)


if __name__ == "__main__":
    unittest.main()
