"""Pytest tests for file upload functionality."""

import io
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

from basalt._internal.http import HTTPClient, HTTPResponse
from basalt.datasets.client import DatasetsClient
from basalt.datasets.file_upload import (
    FileAttachment,
    FileUploadHandler,
    PresignedUploadResponse,
    _detect_content_type,
    _read_file_bytes,
    _validate_file_size,
)
from basalt.types.exceptions import FileUploadError, FileValidationError

# Fixtures


@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary test file."""
    test_file = tmp_path / "test.png"
    test_file.write_bytes(b"fake png content")
    return test_file


@pytest.fixture
def http_client():
    """Create a mock HTTP client."""
    return Mock(spec=HTTPClient)


@pytest.fixture
def file_upload_handler(http_client):
    """Create a FileUploadHandler instance."""
    return FileUploadHandler(
        http_client=http_client,
        base_url="https://api.example.com",
        api_key="test-api-key",
    )


# FileAttachment Tests


class TestFileAttachment:
    """Tests for FileAttachment dataclass."""

    def test_from_path_string(self, temp_file):
        """Test creating FileAttachment from string path."""
        attachment = FileAttachment(source=str(temp_file))
        assert attachment.filename == "test.png"
        assert attachment.source == str(temp_file)
        assert attachment.content_type is None

    def test_from_path_object(self, temp_file):
        """Test creating FileAttachment from Path object."""
        attachment = FileAttachment(source=temp_file)
        assert attachment.filename == "test.png"
        assert attachment.source == temp_file

    def test_from_bytes_with_filename(self):
        """Test creating FileAttachment from bytes with filename."""
        attachment = FileAttachment(
            source=b"content", filename="test.jpg", content_type="image/jpeg"
        )
        assert attachment.filename == "test.jpg"
        assert attachment.content_type == "image/jpeg"
        assert attachment.source == b"content"

    def test_from_bytes_without_filename_raises(self):
        """Test that creating FileAttachment from bytes without filename raises error."""
        with pytest.raises(FileValidationError, match="filename is required"):
            FileAttachment(source=b"content")

    def test_from_file_like_object(self, tmp_path):
        """Test creating FileAttachment from file-like object."""
        test_file = tmp_path / "test.pdf"
        test_file.write_bytes(b"pdf content")

        with open(test_file, "rb") as f:
            attachment = FileAttachment(source=f)
            assert attachment.filename == "test.pdf"

    def test_from_bytesio_without_name(self):
        """Test creating FileAttachment from BytesIO without name attribute."""
        bio = io.BytesIO(b"content")
        with pytest.raises(FileValidationError, match="filename is required for file-like objects"):
            FileAttachment(source=bio)

    def test_explicit_content_type(self, temp_file):
        """Test explicit content type override."""
        attachment = FileAttachment(source=temp_file, content_type="image/png")
        assert attachment.content_type == "image/png"

    def test_invalid_source_type(self):
        """Test that invalid source type raises error."""
        with pytest.raises(FileValidationError, match="source must be"):
            FileAttachment(source=123)  # type: ignore


# Content Type Detection Tests


class TestContentTypeDetection:
    """Tests for _detect_content_type function."""

    def test_detect_png(self):
        """Test PNG content type detection."""
        attachment = FileAttachment(source="test.png")
        content_type = _detect_content_type(attachment)
        assert content_type == "image/png"

    def test_detect_jpeg(self):
        """Test JPEG content type detection."""
        attachment = FileAttachment(source="test.jpeg")
        content_type = _detect_content_type(attachment)
        assert content_type == "image/jpeg"

    def test_detect_jpg(self):
        """Test JPG content type detection."""
        attachment = FileAttachment(source="test.jpg")
        content_type = _detect_content_type(attachment)
        assert content_type == "image/jpeg"

    def test_detect_pdf(self):
        """Test PDF content type detection."""
        attachment = FileAttachment(source="test.pdf")
        content_type = _detect_content_type(attachment)
        assert content_type == "application/pdf"

    def test_detect_html(self):
        """Test HTML content type detection."""
        attachment = FileAttachment(source="test.html")
        content_type = _detect_content_type(attachment)
        assert content_type == "text/html"

    def test_detect_markdown(self):
        """Test Markdown content type detection."""
        attachment = FileAttachment(source="test.md")
        content_type = _detect_content_type(attachment)
        assert content_type == "text/markdown"

    def test_explicit_content_type_valid(self):
        """Test explicit valid content type."""
        attachment = FileAttachment(source=b"content", filename="test", content_type="image/png")
        content_type = _detect_content_type(attachment)
        assert content_type == "image/png"

    def test_explicit_content_type_invalid(self):
        """Test explicit invalid content type raises error."""
        attachment = FileAttachment(source=b"content", filename="test", content_type="text/plain")
        with pytest.raises(FileValidationError, match="not allowed"):
            _detect_content_type(attachment)

    def test_unknown_extension_without_explicit(self):
        """Test unknown extension without explicit content type raises error."""
        attachment = FileAttachment(source="test.xyz")
        with pytest.raises(FileValidationError, match="Could not determine"):
            _detect_content_type(attachment)

    def test_no_extension_no_explicit(self):
        """Test file without extension and no explicit content type."""
        attachment = FileAttachment(source="testfile")
        with pytest.raises(FileValidationError, match="Could not determine"):
            _detect_content_type(attachment)


# File Reading Tests


class TestFileReading:
    """Tests for _read_file_bytes function."""

    def test_read_from_path_string(self, temp_file):
        """Test reading from string path."""
        content = _read_file_bytes(str(temp_file))
        assert content == b"fake png content"

    def test_read_from_path_object(self, temp_file):
        """Test reading from Path object."""
        content = _read_file_bytes(temp_file)
        assert content == b"fake png content"

    def test_read_from_bytes(self):
        """Test reading from bytes."""
        data = b"test data"
        content = _read_file_bytes(data)
        assert content == data

    def test_read_from_file_object(self, temp_file):
        """Test reading from file object."""
        with open(temp_file, "rb") as f:
            content = _read_file_bytes(f)
            assert content == b"fake png content"

    def test_read_from_bytesio(self):
        """Test reading from BytesIO."""
        bio = io.BytesIO(b"bytesio content")
        content = _read_file_bytes(bio)
        assert content == b"bytesio content"

    def test_read_nonexistent_file(self):
        """Test reading nonexistent file raises error."""
        with pytest.raises(FileValidationError, match="File not found"):
            _read_file_bytes("/nonexistent/file.png")

    def test_read_directory(self, tmp_path):
        """Test reading directory raises error."""
        with pytest.raises(FileValidationError, match="not a file"):
            _read_file_bytes(tmp_path)

    def test_read_preserves_file_position(self, temp_file):
        """Test that reading preserves file object position."""
        with open(temp_file, "rb") as f:
            f.read(5)  # Read first 5 bytes
            initial_pos = f.tell()
            _read_file_bytes(f)
            final_pos = f.tell()
            assert final_pos == initial_pos  # Position restored


# File Size Validation Tests


class TestFileSizeValidation:
    """Tests for _validate_file_size function."""

    def test_valid_small_file(self):
        """Test validation passes for small file."""
        content = b"x" * 1000  # 1KB
        _validate_file_size(content)  # Should not raise

    def test_valid_max_size_file(self):
        """Test validation passes for file at max size."""
        content = b"x" * (10 * 1024 * 1024)  # Exactly 10MB
        _validate_file_size(content)  # Should not raise

    def test_oversized_file(self):
        """Test validation fails for oversized file."""
        content = b"x" * (11 * 1024 * 1024)  # 11MB
        with pytest.raises(FileValidationError, match="exceeds maximum"):
            _validate_file_size(content)

    def test_empty_file(self):
        """Test validation fails for empty file."""
        with pytest.raises(FileValidationError, match="File is empty"):
            _validate_file_size(b"")


# PresignedUploadResponse Tests


class TestPresignedUploadResponse:
    """Tests for PresignedUploadResponse dataclass."""

    def test_from_dict(self):
        """Test creating PresignedUploadResponse from dict."""
        data = {
            "uploadUrl": "https://s3.amazonaws.com/bucket/key?signature=abc",
            "fileKey": "datasets/workspace-id/uuid.jpg",
            "expiresAt": "2025-12-18T10:45:00Z",
            "maxSizeBytes": 10485760,
        }

        response = PresignedUploadResponse.from_dict(data)

        assert response.upload_url == data["uploadUrl"]
        assert response.file_key == data["fileKey"]
        assert response.expires_at == data["expiresAt"]
        assert response.max_size_bytes == data["maxSizeBytes"]


# FileUploadHandler Tests


class TestFileUploadHandler:
    """Tests for FileUploadHandler class."""

    def test_validate_file_success(self, file_upload_handler, temp_file):
        """Test successful file validation."""
        attachment = FileAttachment(source=temp_file, content_type="image/png")

        file_bytes, content_type, filename = file_upload_handler.validate_file(attachment)

        assert file_bytes == b"fake png content"
        assert content_type == "image/png"
        assert filename == "test.png"

    def test_validate_file_invalid_type(self, file_upload_handler):
        """Test validation fails for invalid content type."""
        attachment = FileAttachment(
            source=b"content", filename="test.txt", content_type="text/plain"
        )

        with pytest.raises(FileValidationError):
            file_upload_handler.validate_file(attachment)

    def test_validate_file_too_large(self, file_upload_handler):
        """Test validation fails for oversized file."""
        large_content = b"x" * (11 * 1024 * 1024)
        attachment = FileAttachment(
            source=large_content, filename="large.png", content_type="image/png"
        )

        with pytest.raises(FileValidationError, match="exceeds maximum"):
            file_upload_handler.validate_file(attachment)

    @pytest.mark.asyncio
    async def test_request_presigned_url_async(self, file_upload_handler, http_client):
        """Test async presigned URL request."""
        mock_response = HTTPResponse(
            status_code=200,
            data={
                "uploadUrl": "https://s3.amazonaws.com/test",
                "fileKey": "datasets/ws/uuid.jpg",
                "expiresAt": "2025-12-18T10:45:00Z",
                "maxSizeBytes": 10485760,
            },
            headers={},
        )
        http_client.fetch = AsyncMock(return_value=mock_response)

        result = await file_upload_handler.request_presigned_url("test.jpg", "image/jpeg")

        assert isinstance(result, PresignedUploadResponse)
        assert result.file_key == "datasets/ws/uuid.jpg"
        http_client.fetch.assert_called_once()

    def test_request_presigned_url_sync(self, file_upload_handler, http_client):
        """Test sync presigned URL request."""
        mock_response = HTTPResponse(
            status_code=200,
            data={
                "uploadUrl": "https://s3.amazonaws.com/test",
                "fileKey": "datasets/ws/uuid.jpg",
                "expiresAt": "2025-12-18T10:45:00Z",
                "maxSizeBytes": 10485760,
            },
            headers={},
        )
        http_client.fetch_sync = Mock(return_value=mock_response)

        result = file_upload_handler.request_presigned_url_sync("test.jpg", "image/jpeg")

        assert isinstance(result, PresignedUploadResponse)
        assert result.file_key == "datasets/ws/uuid.jpg"
        http_client.fetch_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_to_s3_async_success(self, file_upload_handler, http_client):
        """Test successful async S3 upload."""
        mock_httpx_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_httpx_client.put = AsyncMock(return_value=mock_response)
        http_client._ensure_async_client = Mock(return_value=mock_httpx_client)

        await file_upload_handler.upload_to_s3(
            "https://s3.amazonaws.com/test", b"content", "image/jpeg"
        )

        mock_httpx_client.put.assert_called_once()
        call_args = mock_httpx_client.put.call_args
        assert call_args[0][0] == "https://s3.amazonaws.com/test"
        assert call_args[1]["content"] == b"content"
        assert call_args[1]["headers"]["Content-Type"] == "image/jpeg"

    def test_upload_to_s3_sync_success(self, file_upload_handler, http_client):
        """Test successful sync S3 upload."""
        mock_httpx_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_httpx_client.put = Mock(return_value=mock_response)
        http_client._ensure_sync_client = Mock(return_value=mock_httpx_client)

        file_upload_handler.upload_to_s3_sync(
            "https://s3.amazonaws.com/test", b"content", "image/jpeg"
        )

        mock_httpx_client.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_to_s3_failure(self, file_upload_handler, http_client):
        """Test S3 upload failure."""
        mock_httpx_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.text = "Access Denied"
        mock_httpx_client.put = AsyncMock(return_value=mock_response)
        http_client._ensure_async_client = Mock(return_value=mock_httpx_client)

        with pytest.raises(FileUploadError, match="S3 upload failed"):
            await file_upload_handler.upload_to_s3(
                "https://s3.amazonaws.com/test", b"content", "image/jpeg"
            )

    @pytest.mark.asyncio
    async def test_upload_to_s3_timeout(self, file_upload_handler, http_client):
        """Test S3 upload timeout."""
        mock_httpx_client = AsyncMock()
        mock_httpx_client.put = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        http_client._ensure_async_client = Mock(return_value=mock_httpx_client)

        with pytest.raises(FileUploadError, match="timed out"):
            await file_upload_handler.upload_to_s3(
                "https://s3.amazonaws.com/test", b"content", "image/jpeg"
            )

    @pytest.mark.asyncio
    async def test_upload_file_complete_workflow(self, file_upload_handler, http_client, temp_file):
        """Test complete file upload workflow."""
        # Mock presigned URL request
        presigned_response = HTTPResponse(
            status_code=200,
            data={
                "uploadUrl": "https://s3.amazonaws.com/test",
                "fileKey": "datasets/ws/uuid.png",
                "expiresAt": "2025-12-18T10:45:00Z",
                "maxSizeBytes": 10485760,
            },
            headers={},
        )
        http_client.fetch = AsyncMock(return_value=presigned_response)

        # Mock S3 upload
        mock_httpx_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_httpx_client.put = AsyncMock(return_value=mock_response)
        http_client._ensure_async_client = Mock(return_value=mock_httpx_client)

        # Execute
        attachment = FileAttachment(source=temp_file, content_type="image/png")
        s3_key = await file_upload_handler.upload_file(attachment)

        assert s3_key == "datasets/ws/uuid.png"
        http_client.fetch.assert_called_once()
        mock_httpx_client.put.assert_called_once()


# DatasetsClient Integration Tests


class TestDatasetsClientFileUpload:
    """Tests for DatasetsClient file upload integration."""

    @pytest.fixture
    def client(self):
        """Create a DatasetsClient instance."""
        return DatasetsClient(api_key="test-api-key")

    @pytest.mark.asyncio
    async def test_add_row_with_file_async(self, client, temp_file):
        """Test add_row with FileAttachment."""
        with patch.object(
            client._file_upload_handler, "upload_file", new_callable=AsyncMock
        ) as mock_upload:
            mock_upload.return_value = "datasets/ws/uuid.png"

            with patch.object(client._http_client, "fetch", new_callable=AsyncMock) as mock_fetch:
                mock_fetch.return_value = HTTPResponse(
                    status_code=200,
                    data={
                        "datasetRow": {
                            "values": {"image": "datasets/ws/uuid.png"},
                            "name": "Test Row",
                        }
                    },
                    headers={},
                )

                attachment = FileAttachment(source=temp_file)
                row = await client.add_row(
                    slug="test-dataset",
                    values={"image": attachment},
                    name="Test Row",
                )

                # Verify file was uploaded
                mock_upload.assert_called_once()
                assert mock_upload.call_args[0][0] == attachment

                # Verify dataset row was created with S3 key
                assert row.values["image"] == "datasets/ws/uuid.png"

    def test_add_row_with_file_sync(self, client, temp_file):
        """Test add_row_sync with FileAttachment."""
        with patch.object(client._file_upload_handler, "upload_file_sync") as mock_upload:
            mock_upload.return_value = "datasets/ws/uuid.png"

            with patch.object(client._http_client, "fetch_sync") as mock_fetch:
                mock_fetch.return_value = HTTPResponse(
                    status_code=200,
                    data={
                        "datasetRow": {
                            "values": {"image": "datasets/ws/uuid.png"},
                            "name": "Test Row",
                        }
                    },
                    headers={},
                )

                attachment = FileAttachment(source=temp_file)
                row = client.add_row_sync(
                    slug="test-dataset",
                    values={"image": attachment},
                    name="Test Row",
                )

                mock_upload.assert_called_once()
                assert row.values["image"] == "datasets/ws/uuid.png"

    def test_add_row_with_mixed_values(self, client, temp_file):
        """Test add_row with both string and file values."""
        with patch.object(client._file_upload_handler, "upload_file_sync") as mock_upload:
            mock_upload.return_value = "datasets/ws/uuid.png"

            with patch.object(client._http_client, "fetch_sync") as mock_fetch:
                mock_fetch.return_value = HTTPResponse(
                    status_code=200,
                    data={
                        "datasetRow": {
                            "values": {
                                "text": "description",
                                "image": "datasets/ws/uuid.png",
                            },
                            "name": "Test Row",
                        }
                    },
                    headers={},
                )

                attachment = FileAttachment(source=temp_file)
                row = client.add_row_sync(
                    slug="test-dataset",
                    values={"text": "description", "image": attachment},
                    name="Test Row",
                )

                # Only file should be uploaded
                mock_upload.assert_called_once()

                # Both values should be in result
                assert row.values["text"] == "description"
                assert row.values["image"] == "datasets/ws/uuid.png"

    def test_add_row_with_multiple_files(self, client, tmp_path):
        """Test add_row with multiple file attachments."""
        file1 = tmp_path / "file1.png"
        file1.write_bytes(b"content1")
        file2 = tmp_path / "file2.jpg"
        file2.write_bytes(b"content2")

        with patch.object(client._file_upload_handler, "upload_file_sync") as mock_upload:
            mock_upload.side_effect = ["datasets/ws/uuid1.png", "datasets/ws/uuid2.jpg"]

            with patch.object(client._http_client, "fetch_sync") as mock_fetch:
                mock_fetch.return_value = HTTPResponse(
                    status_code=200,
                    data={
                        "datasetRow": {
                            "values": {
                                "image1": "datasets/ws/uuid1.png",
                                "image2": "datasets/ws/uuid2.jpg",
                            }
                        }
                    },
                    headers={},
                )

                row = client.add_row_sync(
                    slug="test-dataset",
                    values={
                        "image1": FileAttachment(source=file1),
                        "image2": FileAttachment(source=file2),
                    },
                )

                # Both files should be uploaded
                assert mock_upload.call_count == 2
                assert row.values["image1"] == "datasets/ws/uuid1.png"
                assert row.values["image2"] == "datasets/ws/uuid2.jpg"

    def test_add_row_file_upload_error_propagates(self, client, temp_file):
        """Test that file upload errors are propagated."""
        with patch.object(client._file_upload_handler, "upload_file_sync") as mock_upload:
            mock_upload.side_effect = FileUploadError("S3 upload failed")

            with pytest.raises(FileUploadError, match="S3 upload failed"):
                client.add_row_sync(
                    slug="test-dataset",
                    values={"image": FileAttachment(source=temp_file)},
                )

    def test_process_file_uploads_only_processes_attachments(self, client):
        """Test that _process_file_uploads only processes FileAttachment objects."""
        values = {"text": "hello", "number": "42", "other": "value"}

        processed = client._process_file_uploads_sync(values)

        # No files, so values should be unchanged
        assert processed == values

    @pytest.mark.asyncio
    async def test_process_file_uploads_async(self, client, temp_file):
        """Test async file processing."""
        with patch.object(
            client._file_upload_handler, "upload_file", new_callable=AsyncMock
        ) as mock_upload:
            mock_upload.return_value = "datasets/ws/uuid.png"

            values = {
                "text": "description",
                "image": FileAttachment(source=temp_file),
            }

            processed = await client._process_file_uploads(values)

            assert processed["text"] == "description"
            assert processed["image"] == "datasets/ws/uuid.png"
            mock_upload.assert_called_once()
