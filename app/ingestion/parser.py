import time
from pathlib import Path
from typing import List

from llama_cloud.client import LlamaCloud
from llama_index.core import Document

from app.utils.logger import get_logger

logger = get_logger(__name__)

class LlamaParse:
    """
    A robust wrapper for the llama-cloud SDK to replace the deprecated 
    llama-index-readers-llama-parse package.
    """
    def __init__(self, api_key: str, result_type: str = "markdown", verbose: bool = False, language: str = "en"):
        self.api_key = api_key
        self.result_type = result_type
        self.verbose = verbose
        self.language = language
        self.client = LlamaCloud(api_key=api_key)

    def load_data(self, file_path: str | Path) -> List[Document]:
        """
        Parses a file using the llama-cloud SDK and returns a list of LlamaIndex Documents.
        
        Args:
            file_path: Path to the file to parse.
            
        Returns:
            List of LlamaIndex Document objects.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Uploading file for parsing: {file_path.name}")
        with open(file_path, "rb") as f:
            file_obj = self.client.files.create(file=f)

        logger.info(f"Creating parsing job for file ID: {file_obj.id}")
        job = self.client.parsing.create(
            file_id=file_obj.id,
            result_type=self.result_type
        )

        # Polling for completion
        logger.info(f"Waiting for parsing job {job.id} to complete...")
        while True:
            status = self.client.parsing.get_status(job_id=job.id)
            if status.status == "SUCCESS":
                logger.info(f"Parsing job {job.id} completed successfully.")
                break
            elif status.status == "FAILURE":
                logger.error(f"Parsing job {job.id} failed.")
                raise RuntimeError(f"LlamaParse job failed for {file_path.name}")
            
            if self.verbose:
                logger.info(f"Job status: {status.status}...")
            time.sleep(2)

        result = self.client.parsing.get_result(job_id=job.id)
        
        # Determine content type based on result_type
        content = result.markdown if self.result_type == "markdown" else result.text
        
        # Return as LlamaIndex Document
        return [Document(text=content)]
