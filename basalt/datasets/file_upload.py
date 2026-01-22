"""File upload utilities for dataset client."""

from __future__ import annotations

import io
import logging
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, cast

import httpx

if TYPE_CHECKING:
    from .._internal.http import HTTPClient

from ..types.exceptions import FileUploadError, FileValidationError

logger = logging.getLogger(__name__)

# Constants
ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "application/pdf",
    "text/html",
    "text/markdown",
}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB


@dataclass
class FileAttachment:
    """
    Represents a file to be uploaded to a dataset.

    Attributes:
        source: File source - can be a file path (str or Path), bytes, or file-like object
        content_type: Optional MIME type (auto-detected if not provided)
        filename: Optional filename (required for bytes, derived from path otherwise)

    Example:
        # From file path
        FileAttachment(source="photo.jpg")

        # From bytes with explicit metadata
        FileAttachment(
            source=image_bytes,
            filename="photo.jpg",
            content_type="image/jpeg"
        )
    """

    source: str | Path | bytes | BinaryIO
    content_type: str | None = None
    filename: str | None = None

    def __post_init__(self) -> None:
        """Validate the attachment at construction time."""
        # Validate source type
        if not isinstance(self.source, (str, Path, bytes, io.BytesIO, io.BufferedReader)):
            raise FileValidationError(
                f"source must be str, Path, bytes, or file-like object, got {type(self.source)}"
            )

        # Derive filename from path if not provided
        if self.filename is None:
            if isinstance(self.source, (str, Path)):
                self.filename = os.path.basename(str(self.source))
            elif isinstance(self.source, bytes):
                raise FileValidationError("filename is required when source is bytes")
            elif hasattr(self.source, "name"):
                # File-like object with name attribute
                self.filename = os.path.basename(str(self.source.name))
            else:
                raise FileValidationError(
                    "filename is required for file-like objects without a name attribute"
                )


@dataclass
class PresignedUploadResponse:
    """Response from the presign-upload endpoint."""

    upload_url: str
    file_key: str
    expires_at: str
    max_size_bytes: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PresignedUploadResponse:
        """Parse API response into PresignedUploadResponse."""
        return cls(
            upload_url=data["uploadUrl"],
            file_key=data["fileKey"],
            expires_at=data["expiresAt"],
            max_size_bytes=data["maxSizeBytes"],
        )


def _detect_content_type(attachment: FileAttachment) -> str:
    """
    Detect and validate content type.

    Args:
        attachment: The file attachment

    Returns:
        Validated content type

    Raises:
        FileValidationError: If content type cannot be determined or is not allowed
    """
    # Use explicit content_type if provided
    if attachment.content_type:
        if attachment.content_type not in ALLOWED_CONTENT_TYPES:
            raise FileValidationError(
                f"Content type '{attachment.content_type}' is not allowed. "
                f"Allowed types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            )
        return attachment.content_type

    # Try to guess from filename
    if attachment.filename:
        guessed, _ = mimetypes.guess_type(attachment.filename)
        if guessed and guessed in ALLOWED_CONTENT_TYPES:
            return guessed

    # Fail if can't determine
    raise FileValidationError(
        "Could not determine content type. Please specify content_type explicitly. "
        f"Allowed types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
    )


def _read_file_bytes(source: str | Path | bytes | BinaryIO) -> bytes:
    """
    Read file bytes from source.

    Args:
        source: The file source

    Returns:
        File contents as bytes

    Raises:
        FileValidationError: If file cannot be read or doesn't exist
    """
    try:
        if isinstance(source, bytes):
            return source
        elif isinstance(source, (str, Path)):
            path = Path(source)
            if not path.exists():
                raise FileValidationError(f"File not found: {source}")
            if not path.is_file():
                raise FileValidationError(f"Path is not a file: {source}")
            with open(path, "rb") as f:
                return f.read()
        else:
            # File-like object
            if hasattr(source, "read"):
                # Cast to BinaryIO for type checker - we've verified it has read()
                file_obj = cast(BinaryIO, source)
                # Save current position
                current_pos = file_obj.tell() if hasattr(file_obj, "tell") else None
                # Seek to beginning if possible
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
                data = file_obj.read()
                # Restore position if possible
                if current_pos is not None and hasattr(file_obj, "seek"):
                    file_obj.seek(current_pos)
                if isinstance(data, bytes):
                    return data
                else:
                    raise FileValidationError("File-like object must return bytes from read()")
            else:
                raise FileValidationError("File-like object must have a read() method")
    except FileValidationError:
        raise
    except Exception as e:
        raise FileValidationError(f"Failed to read file: {e}") from e


def _validate_file_size(file_bytes: bytes) -> None:
    """
    Validate file size is within limits.

    Args:
        file_bytes: The file content

    Raises:
        FileValidationError: If file size is invalid
    """
    size = len(file_bytes)
    if size == 0:
        raise FileValidationError("File is empty")
    if size > MAX_FILE_SIZE_BYTES:
        size_mb = size / (1024 * 1024)
        limit_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise FileValidationError(f"File size {size_mb:.2f}MB exceeds maximum {limit_mb}MB")


class FileUploadHandler:
    """Handles file validation, presigned URL requests, and S3 uploads."""

    def __init__(self, http_client: HTTPClient, base_url: str, api_key: str) -> None:
        """
        Initialize FileUploadHandler.

        Args:
            http_client: The HTTP client for API requests
            base_url: Base URL for the Basalt API
            api_key: API key for authentication
        """
        self._http_client = http_client
        self._base_url = base_url
        self._api_key = api_key
        self._logger = logging.getLogger("basalt.datasets.file_upload")

    def validate_file(self, attachment: FileAttachment) -> tuple[bytes, str, str]:
        """
        Validate file and extract metadata.

        Args:
            attachment: The file attachment to validate

        Returns:
            Tuple of (file_bytes, content_type, filename)

        Raises:
            FileValidationError: If validation fails
        """
        # Detect content type
        content_type = _detect_content_type(attachment)

        # Read file bytes
        file_bytes = _read_file_bytes(attachment.source)

        # Validate size
        _validate_file_size(file_bytes)

        # Filename should exist after __post_init__
        filename = attachment.filename
        if not filename:
            raise FileValidationError("filename is required")

        self._logger.debug(
            "File validated successfully",
            extra={
                "filename": filename,
                "content_type": content_type,
                "size_bytes": len(file_bytes),
            },
        )

        return file_bytes, content_type, filename

    async def request_presigned_url(
        self, filename: str, content_type: str
    ) -> PresignedUploadResponse:
        """
        Request presigned URL from Basalt API.

        Args:
            filename: The filename for the upload
            content_type: The MIME type of the file

        Returns:
            Presigned upload response with URL and metadata

        Raises:
            BasaltAPIError: If the API request fails
        """
        url = f"{self._base_url}/files/generate-upload-url"
        body: dict[str, Any] = {"fileName": filename, "contentType": content_type}
        headers = {"Authorization": f"Bearer {self._api_key}"}

        self._logger.debug(
            "Requesting presigned URL",
            extra={"filename": filename, "content_type": content_type},
        )

        response = await self._http_client.fetch(url, "POST", body=body, headers=headers)

        if not response or not response.data:
            raise FileUploadError("Failed to get presigned URL: empty response")

        return PresignedUploadResponse.from_dict(response.data)

    def request_presigned_url_sync(
        self, filename: str, content_type: str
    ) -> PresignedUploadResponse:
        """
        Request presigned URL from Basalt API (synchronous version).

        Args:
            filename: The filename for the upload
            content_type: The MIME type of the file

        Returns:
            Presigned upload response with URL and metadata

        Raises:
            BasaltAPIError: If the API request fails
        """
        url = f"{self._base_url}/files/generate-upload-url"
        body: dict[str, Any] = {"fileName": filename, "contentType": content_type}
        headers = {"Authorization": f"Bearer {self._api_key}"}

        self._logger.debug(
            "Requesting presigned URL",
            extra={"filename": filename, "content_type": content_type},
        )

        response = self._http_client.fetch_sync(url, "POST", body=body, headers=headers)

        if not response or not response.data:
            raise FileUploadError("Failed to get presigned URL: empty response")

        return PresignedUploadResponse.from_dict(response.data)

    async def upload_to_s3(self, presigned_url: str, file_bytes: bytes, content_type: str) -> None:
        """
        Upload file to S3 using presigned URL.

        Args:
            presigned_url: The presigned S3 upload URL
            file_bytes: The file content to upload
            content_type: The MIME type of the file

        Raises:
            FileUploadError: If the upload fails
        """
        headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(file_bytes)),
        }

        self._logger.debug(
            "Uploading to S3",
            extra={"size_bytes": len(file_bytes), "content_type": content_type},
        )

        try:
            # Use the HTTPClient's underlying httpx client
            client = self._http_client._ensure_async_client()
            response = await client.put(
                presigned_url,
                content=file_bytes,
                headers=headers,
                timeout=30.0,
            )

            if response.status_code not in (200, 204):
                error_text = response.text if hasattr(response, "text") else ""
                raise FileUploadError(
                    f"S3 upload failed with status {response.status_code}: {error_text}"
                )

            self._logger.debug("S3 upload successful")

        except FileUploadError:
            raise
        except httpx.TimeoutException as e:
            raise FileUploadError(f"Upload timed out: {e}") from e
        except httpx.TransportError as e:
            raise FileUploadError(f"Network error during upload: {e}") from e
        except Exception as e:
            raise FileUploadError(f"Unexpected error during upload: {e}") from e

    def upload_to_s3_sync(self, presigned_url: str, file_bytes: bytes, content_type: str) -> None:
        """
        Upload file to S3 using presigned URL (synchronous version).

        Args:
            presigned_url: The presigned S3 upload URL
            file_bytes: The file content to upload
            content_type: The MIME type of the file

        Raises:
            FileUploadError: If the upload fails
        """
        headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(file_bytes)),
        }

        self._logger.debug(
            "Uploading to S3",
            extra={"size_bytes": len(file_bytes), "content_type": content_type},
        )

        try:
            # Use the HTTPClient's underlying httpx client
            client = self._http_client._ensure_sync_client()
            response = client.put(
                presigned_url,
                content=file_bytes,
                headers=headers,
                timeout=30.0,
            )

            if response.status_code not in (200, 204):
                error_text = response.text if hasattr(response, "text") else ""
                raise FileUploadError(
                    f"S3 upload failed with status {response.status_code}: {error_text}"
                )

            self._logger.debug("S3 upload successful")

        except FileUploadError:
            raise
        except httpx.TimeoutException as e:
            raise FileUploadError(f"Upload timed out: {e}") from e
        except httpx.TransportError as e:
            raise FileUploadError(f"Network error during upload: {e}") from e
        except Exception as e:
            raise FileUploadError(f"Unexpected error during upload: {e}") from e

    async def upload_file(self, attachment: FileAttachment) -> str:
        """
        Complete upload workflow for a single file.

        This orchestrates the entire upload process:
        1. Validate file and extract metadata
        2. Request presigned URL from API
        3. Upload file to S3
        4. Return S3 key

        Args:
            attachment: The file attachment to upload

        Returns:
            S3 file key (e.g., "datasets/workspace-id/uuid.jpg")

        Raises:
            FileValidationError: If file validation fails
            FileUploadError: If upload fails
            BasaltAPIError: If API request fails
        """
        self._logger.info("Starting file upload", extra={"filename": attachment.filename})

        # 1. Validate and extract
        file_bytes, content_type, filename = self.validate_file(attachment)

        # 2. Request presigned URL
        presigned = await self.request_presigned_url(filename, content_type)

        # 3. Upload to S3
        await self.upload_to_s3(presigned.upload_url, file_bytes, content_type)

        # 4. Return S3 key
        self._logger.info(
            "File uploaded successfully",
            extra={"file_key": presigned.file_key, "filename": filename},
        )

        return presigned.file_key

    def upload_file_sync(self, attachment: FileAttachment) -> str:
        """
        Complete upload workflow for a single file (synchronous version).

        This orchestrates the entire upload process:
        1. Validate file and extract metadata
        2. Request presigned URL from API
        3. Upload file to S3
        4. Return S3 key

        Args:
            attachment: The file attachment to upload

        Returns:
            S3 file key (e.g., "datasets/workspace-id/uuid.jpg")

        Raises:
            FileValidationError: If file validation fails
            FileUploadError: If upload fails
            BasaltAPIError: If API request fails
        """
        self._logger.info("Starting file upload", extra={"filename": attachment.filename})

        # 1. Validate and extract
        file_bytes, content_type, filename = self.validate_file(attachment)

        # 2. Request presigned URL
        presigned = self.request_presigned_url_sync(filename, content_type)

        # 3. Upload to S3
        self.upload_to_s3_sync(presigned.upload_url, file_bytes, content_type)

        # 4. Return S3 key
        self._logger.info(
            "File uploaded successfully",
            extra={"file_key": presigned.file_key, "filename": filename},
        )

        return presigned.file_key
