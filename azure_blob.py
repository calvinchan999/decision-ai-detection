import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlsplit
from urllib.request import Request, urlopen


class AzureUploadError(Exception):
    pass


# Blob folder used for each detection category. The faucet detector was
# historically called "boilwater"; set that value here only if the container
# folder has to stay continuous with the older data set.
CATEGORY_PREFIXES = {
    "drinking": "drinking",
    "faucet": "faucet",
}


class AzureBlobUploader:
    def __init__(self) -> None:
        container_url = os.environ.get("AZURE_BLOB_CONTAINER_URL", "").strip()
        sas_token = os.environ.get("AZURE_BLOB_SAS_TOKEN", "").strip().lstrip("?")
        if not container_url or container_url.startswith("https://REPLACE_"):
            raise AzureUploadError("Set AZURE_BLOB_CONTAINER_URL in .env.")
        if not sas_token or sas_token.startswith("REPLACE_"):
            raise AzureUploadError("Set AZURE_BLOB_SAS_TOKEN in .env.")

        parsed_url = urlsplit(container_url)
        if (
            parsed_url.scheme.lower() != "https"
            or not parsed_url.netloc
            or not parsed_url.path.strip("/")
            or parsed_url.query
            or parsed_url.fragment
        ):
            raise AzureUploadError(
                "AZURE_BLOB_CONTAINER_URL must be an HTTPS container URL "
                "without a query or fragment."
            )

        permissions = parse_qs(sas_token).get("sp", [""])[0]
        if not any(permission in permissions for permission in ("c", "w")):
            raise AzureUploadError(
                "AZURE_BLOB_SAS_TOKEN must include create or write permission."
            )

        self.container_url = container_url.rstrip("/")
        self.sas_token = sas_token

    def upload(self, image_path: Path, category: str) -> None:
        if category not in CATEGORY_PREFIXES:
            raise AzureUploadError(f"Unknown detection category: {category}")
        try:
            image = image_path.read_bytes()
        except OSError as error:
            raise AzureUploadError(f"Could not read captured image: {error}") from None

        blob_url = (
            f"{self.container_url}/{CATEGORY_PREFIXES[category]}/"
            f"{quote(image_path.name, safe='')}?{self.sas_token}"
        )
        request = Request(
            blob_url,
            data=image,
            headers={
                "Content-Type": "image/jpeg",
                "x-ms-blob-type": "BlockBlob",
                "x-ms-version": "2023-11-03",
            },
            method="PUT",
        )
        try:
            with urlopen(request, timeout=30) as response:
                if response.status != 201:
                    raise AzureUploadError(
                        f"Azure Blob upload returned HTTP {response.status}."
                    )
        except HTTPError as error:
            raise AzureUploadError(
                f"Azure Blob upload returned HTTP {error.code}."
            ) from None
        except (URLError, TimeoutError, OSError) as error:
            reason = error.reason if isinstance(error, URLError) else error
            raise AzureUploadError(
                f"Azure Blob upload connection failed: {reason}"
            ) from None

    def upload_then_delete(self, image_path: Path, categories: list[str]) -> None:
        if not categories:
            raise AzureUploadError("At least one detection category is required.")
        errors = []
        for category in categories:
            try:
                self.upload(image_path, category)
            except AzureUploadError as error:
                errors.append(f"{category}: {error}")
        if errors:
            raise AzureUploadError("; ".join(errors))
        try:
            image_path.unlink()
        except OSError as error:
            raise AzureUploadError(
                f"Azure upload succeeded, but the local image could not be deleted: "
                f"{error}"
            ) from None
