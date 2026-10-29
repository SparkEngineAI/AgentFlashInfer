"""Tests for VideoAnalysis tool."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from examples.deepresearch.tools.file_tools.video_analysis import (
    MAX_FILE_SIZE,
    SUPPORTED_AUDIO_TYPES,
    SUPPORTED_VIDEO_TYPES,
    temp_directory,
    VideoAnalysis,
)


@pytest.fixture
def mock_openai():
    with patch("examples.deepresearch.tools.file_tools.video_analysis.OpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_requests_session():
    with patch("examples.deepresearch.tools.file_tools.video_analysis.requests.Session") as mock_cls:
        mock_session = MagicMock()
        mock_cls.return_value = mock_session
        yield mock_session


@pytest.fixture
def video_tool(mock_openai, mock_requests_session):
    with patch.object(VideoAnalysis, "_check_dependencies"):
        tool = VideoAnalysis()
        tool.ffmpeg = None
        tool._scene_detect_available = False
    return tool


class TestTempDirectory:
    def test_creates_and_cleans_up(self):
        with temp_directory() as tmp_dir:
            assert tmp_dir.exists()
            # Write a file to verify the directory works
            (tmp_dir / "test.txt").write_text("hello")
            assert (tmp_dir / "test.txt").exists()
        # After context, directory should be cleaned up
        assert not tmp_dir.exists()


class TestVideoAnalysisInit:
    def test_init_config(self, mock_openai, mock_requests_session):
        with patch.object(VideoAnalysis, "_check_dependencies"):
            tool = VideoAnalysis()
        assert tool.config["api_key"] is not None
        assert tool.config["max_file_size"] == MAX_FILE_SIZE
        assert tool.config["max_frames"] <= 50
        assert tool.config["timeout"] <= 300

    def test_init_custom_config(self, mock_openai, mock_requests_session):
        with patch.object(VideoAnalysis, "_check_dependencies"):
            tool = VideoAnalysis(cfg={"timeout": 500, "max_frames": 100})
        assert tool.config["timeout"] == 300  # Capped at 300
        assert tool.config["max_frames"] == 50  # Capped at 50

    def test_check_dependencies_no_ffmpeg(self, mock_openai, mock_requests_session):
        with patch("examples.deepresearch.tools.file_tools.video_analysis.logger"):
            tool = VideoAnalysis()
        assert tool.ffmpeg is None or hasattr(tool.ffmpeg, "input")


class TestParseParams:
    def test_dict_params(self, video_tool):
        result = video_tool._parse_params({"url": "https://example.com/video.mp4", "prompt": "Describe"})
        assert result["url"] == "https://example.com/video.mp4"
        assert result["prompt"] == "Describe"
        assert result["num_frames"] == 8

    def test_string_params(self, video_tool):
        result = video_tool._parse_params('{"url": "/path/to/video.mp4", "prompt": "Analyze"}')
        assert result["url"] == "/path/to/video.mp4"

    def test_missing_required(self, video_tool):
        with pytest.raises(ValueError, match="Missing required"):
            video_tool._parse_params({"url": "test.mp4"})

    def test_invalid_json(self, video_tool):
        with pytest.raises(ValueError, match="Invalid JSON"):
            video_tool._parse_params("not json")

    def test_num_frames_capped(self, video_tool):
        result = video_tool._parse_params({"url": "test.mp4", "prompt": "desc", "num_frames": 100})
        assert result["num_frames"] == video_tool.config["max_frames"]


class TestIsSupportedType:
    def test_mp4(self, video_tool):
        assert video_tool._is_supported_type(".mp4") is True

    def test_mp3(self, video_tool):
        assert video_tool._is_supported_type(".mp3") is True

    def test_unsupported(self, video_tool):
        assert video_tool._is_supported_type(".exe") is False

    def test_without_dot(self, video_tool):
        assert video_tool._is_supported_type("mp4") is True


class TestGetFileExtension:
    def test_from_content_type(self, video_tool):
        assert video_tool._get_file_extension("video/mp4", "") == ".mp4"

    def test_from_url(self, video_tool):
        assert video_tool._get_file_extension("", "https://example.com/video.mov") == ".mov"

    def test_default_mp4(self, video_tool):
        assert video_tool._get_file_extension("", "https://example.com/video") == ".mp4"

    def test_content_type_with_charset(self, video_tool):
        assert video_tool._get_file_extension("video/mp4; charset=utf-8", "") == ".mp4"


class TestResolveLocalPath:
    def test_absolute_existing(self, video_tool, tmp_path):
        f = tmp_path / "video.mp4"
        f.write_bytes(b"\x00" * 100)
        result = video_tool._resolve_local_path(str(f))
        assert result == f

    def test_nonexistent_raises(self, video_tool):
        with pytest.raises(FileNotFoundError):
            video_tool._resolve_local_path("/nonexistent/video.mp4")

    def test_relative_path(self, video_tool, tmp_path):
        f = tmp_path / "video.mp4"
        f.write_bytes(b"\x00" * 100)
        with patch.dict(os.environ, {"PROJECT_ROOT": str(tmp_path)}):
            result = video_tool._resolve_local_path("video.mp4")
            assert result.exists()


class TestValidateMediaFile:
    def test_valid_file(self, video_tool, tmp_path):
        f = tmp_path / "video.mp4"
        f.write_bytes(b"\x00" * 100)
        video_tool._validate_media_file(f)  # Should not raise

    def test_nonexistent_raises(self, video_tool):
        with pytest.raises(FileNotFoundError):
            video_tool._validate_media_file(Path("/nonexistent.mp4"))

    def test_too_large_raises(self, video_tool, tmp_path):
        f = tmp_path / "video.mp4"
        f.write_bytes(b"\x00" * 100)
        video_tool.config["max_file_size"] = 10
        with pytest.raises(ValueError, match="exceeds limit"):
            video_tool._validate_media_file(f)

    def test_unsupported_type_raises(self, video_tool, tmp_path):
        f = tmp_path / "video.exe"
        f.write_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="Unsupported"):
            video_tool._validate_media_file(f)


class TestIsAudioOnly:
    def test_audio_extension(self, video_tool, tmp_path):
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"\x00" * 100)
        assert video_tool._is_audio_only(f) is True

    def test_video_extension(self, video_tool, tmp_path):
        f = tmp_path / "video.mp4"
        f.write_bytes(b"\x00" * 100)
        assert video_tool._is_audio_only(f) is False


class TestCall:
    def test_missing_file_returns_error(self, video_tool):
        result = video_tool.call({"url": "/nonexistent.mp4", "prompt": "Describe"})
        assert result["status"] == "error"
        assert result["error"] is not None

    def test_unsupported_type_returns_error(self, video_tool, tmp_path):
        f = tmp_path / "doc.exe"
        f.write_bytes(b"\x00" * 100)
        result = video_tool.call({"url": str(f), "prompt": "Describe"})
        assert result["status"] == "error"

    def test_audio_analysis(self, video_tool, tmp_path):
        f = tmp_path / "audio.mp3"
        f.write_bytes(b"\x00" * 100)
        with patch.object(video_tool, "_is_audio_only", return_value=True), \
             patch.object(video_tool, "_transcribe_audio", return_value="transcribed text"), \
             patch.object(video_tool, "_analyze_media", return_value="analysis result"):
            result = video_tool.call({"url": str(f), "prompt": "Describe audio"})
            assert result["status"] == "success"
            assert result["data"]["transcript"] == "transcribed text"

    def test_video_analysis(self, video_tool, tmp_path):
        f = tmp_path / "video.mp4"
        f.write_bytes(b"\x00" * 100)
        with patch.object(video_tool, "_is_audio_only", return_value=False), \
             patch.object(video_tool, "_extract_audio", return_value=f), \
             patch.object(video_tool, "_transcribe_audio", return_value="transcribed"), \
             patch.object(video_tool, "_extract_keyframes", return_value=["base64frame"]), \
             patch.object(video_tool, "_analyze_media", return_value="analysis"):
            result = video_tool.call({"url": str(f), "prompt": "Describe video"})
            assert result["status"] == "success"
            assert result["data"]["frame_count"] == 1


class TestProcessInput:
    def test_http_url(self, video_tool, tmp_path):
        with patch.object(video_tool, "_download_media", return_value=tmp_path / "video.mp4") as mock_dl:
            result = video_tool._process_input("https://example.com/video.mp4", tmp_path)
            mock_dl.assert_called_once()

    def test_local_path(self, video_tool, tmp_path):
        f = tmp_path / "video.mp4"
        f.write_bytes(b"\x00" * 100)
        with patch.object(video_tool, "_resolve_local_path", return_value=f) as mock_resolve:
            result = video_tool._process_input(str(f), tmp_path)
            mock_resolve.assert_called_once()


class TestBuildAnalysisMessages:
    def test_audio_messages(self, video_tool):
        messages = video_tool._build_analysis_messages(
            prompt="What is said?",
            transcript="Hello world",
            frames=[],
            is_audio=True
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_video_messages_include_frames(self, video_tool):
        messages = video_tool._build_analysis_messages(
            prompt="Describe scene",
            transcript="Audio content",
            frames=["base64img1", "base64img2"],
            is_audio=False
        )
        user_content = messages[1]["content"]
        image_items = [c for c in user_content if c.get("type") == "image_url"]
        assert len(image_items) == 2


class TestAnalyzeMedia:
    def test_success(self, video_tool, mock_openai):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Analysis result"
        mock_openai.chat.completions.create.return_value = mock_response

        result = video_tool._analyze_media("prompt", "transcript", [], False)
        assert result == "Analysis result"

    def test_failure_returns_error_message(self, video_tool, mock_openai):
        mock_openai.chat.completions.create.side_effect = Exception("API error")
        result = video_tool._analyze_media("prompt", "transcript", [], False)
        assert "failed" in result.lower()


class TestTranscribeAudio:
    def test_success(self, video_tool, mock_openai, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"\x00" * 100)

        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta.content = "Hello "
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = "World"
        mock_openai.chat.completions.create.return_value = [mock_chunk1, mock_chunk2]

        result = video_tool._transcribe_audio(audio)
        assert "Hello" in result
        assert "World" in result

    def test_failure_returns_empty(self, video_tool, mock_openai, tmp_path):
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"\x00" * 100)
        mock_openai.chat.completions.create.side_effect = Exception("API error")

        result = video_tool._transcribe_audio(audio)
        assert result == ""


class TestFrameToBase64:
    def test_converts_image(self, video_tool):
        from PIL import Image
        import io
        img = Image.new("RGB", (10, 10), color="red")
        result = video_tool._frame_to_base64(img)
        assert isinstance(result, str)
        # Verify it's valid base64
        import base64
        decoded = base64.b64decode(result)
        assert len(decoded) > 0


class TestGetVideoDuration:
    def test_default_on_failure(self, video_tool, tmp_path):
        f = tmp_path / "video.mp4"
        f.write_bytes(b"\x00" * 100)
        duration = video_tool._get_video_duration(f)
        assert duration == 60.0  # Default fallback