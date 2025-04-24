from typing import Optional
import aiobotocore.client
from loguru import logger
import os

class S3StorageService:
    """
    Service for interacting with S3-compatible storage using aiobotocore.
    """
    def __init__(self, s3_client: Optional[aiobotocore.client.BaseClient], bucket_name: Optional[str]):
        """
        Initializes the storage service.

        Args:
            s3_client: An initialized aiobotocore S3 client instance. Can be None if S3 disabled.
            bucket_name: The name of the S3 bucket to use. Can be None if S3 disabled.
        """
        self.s3_client = s3_client
        self.bucket_name = bucket_name
        if self.s3_client and self.bucket_name:
            logger.info(f"S3StorageService initialized for bucket: {self.bucket_name}")
        else:
             logger.warning("S3StorageService initialized but S3 client or bucket name is missing/disabled.")

    async def upload_file(self, file_path: str, s3_key: str, content_type: Optional[str] = None) -> bool:
        """
        Uploads a local file to S3.

        Args:
            file_path: The path to the local file to upload.
            s3_key: The desired key (path) for the file in the S3 bucket.
            content_type: Optional content type for the S3 object.

        Returns:
            True if upload was successful, False otherwise.
        """
        if not self.s3_client or not self.bucket_name:
            logger.error(f"Cannot upload file '{s3_key}'. S3 client or bucket not configured.")
            return False
        if not os.path.exists(file_path):
             logger.error(f"Cannot upload file '{s3_key}'. Local file not found: {file_path}")
             return False

        extra_args = {}
        if content_type:
            extra_args['ContentType'] = content_type

        logger.info(f"Uploading '{file_path}' to S3 bucket '{self.bucket_name}' with key '{s3_key}'")
        try:
            with open(file_path, "rb") as f:
                await self.s3_client.put_object(
                    Bucket=self.bucket_name,
                    Key=s3_key,
                    Body=f,
                    **extra_args
                )
            logger.info(f"Successfully uploaded '{s3_key}' to S3.")
            return True
        except aiobotocore.client.ClientError as e:
            logger.error(f"S3 ClientError during upload of '{s3_key}': {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error during S3 upload of '{s3_key}': {e}", exc_info=True)
            return False

    async def download_file(self, s3_key: str, destination_path: str) -> bool:
        """
        Downloads a file from S3 to a local path.

        Args:
            s3_key: The key (path) of the file in the S3 bucket.
            destination_path: The local path where the file should be saved.

        Returns:
            True if download was successful, False otherwise.
        """
        if not self.s3_client or not self.bucket_name:
            logger.error(f"Cannot download file '{s3_key}'. S3 client or bucket not configured.")
            return False

        # Ensure destination directory exists
        dest_dir = os.path.dirname(destination_path)
        if dest_dir and not os.path.exists(dest_dir):
            try:
                os.makedirs(dest_dir)
                logger.debug(f"Created destination directory: {dest_dir}")
            except OSError as e:
                 logger.error(f"Failed to create directory '{dest_dir}' for download: {e}")
                 return False

        logger.info(f"Downloading S3 key '{s3_key}' from bucket '{self.bucket_name}' to '{destination_path}'")
        try:
            response = await self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            # Stream the download to handle potentially large files
            with open(destination_path, "wb") as f:
                async for chunk in response['Body'].iter_chunks():
                    f.write(chunk)
            logger.info(f"Successfully downloaded '{s3_key}' to '{destination_path}'.")
            return True
        except aiobotocore.client.ClientError as e:
            # Handle specific errors like 'NoSuchKey'
            if e.response['Error']['Code'] == 'NoSuchKey':
                 logger.error(f"S3 key '{s3_key}' not found in bucket '{self.bucket_name}'.")
            else:
                 logger.error(f"S3 ClientError during download of '{s3_key}': {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error during S3 download of '{s3_key}': {e}", exc_info=True)
            return False

    async def delete_file(self, s3_key: str) -> bool:
        """
        Deletes a file from S3.

        Args:
            s3_key: The key (path) of the file in the S3 bucket.

        Returns:
            True if deletion was successful (or file didn't exist), False on error.
        """
        if not self.s3_client or not self.bucket_name:
            logger.error(f"Cannot delete file '{s3_key}'. S3 client or bucket not configured.")
            return False

        logger.info(f"Deleting S3 key '{s3_key}' from bucket '{self.bucket_name}'")
        try:
            await self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            logger.info(f"Successfully initiated deletion of '{s3_key}' from S3 (or key did not exist).")
            # Note: delete_object doesn't error if the key doesn't exist
            return True
        except aiobotocore.client.ClientError as e:
            logger.error(f"S3 ClientError during deletion of '{s3_key}': {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error during S3 deletion of '{s3_key}': {e}", exc_info=True)
            return False

# --- Dependency function (can live here or in dependencies.py) ---
# from app.config import settings
# from app.dependencies import get_s3_client # Assuming get_s3_client is in dependencies

# async def get_storage_service(
#     s3_client = Depends(get_s3_client)
# ) -> S3StorageService:
#     """Provides an instance of the S3StorageService."""
#     return S3StorageService(s3_client, settings.S3_BUCKET_NAME)