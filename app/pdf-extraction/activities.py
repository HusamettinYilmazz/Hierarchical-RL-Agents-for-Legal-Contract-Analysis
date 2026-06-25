
import os
from pathlib import Path
from dataclasses import dataclass

from temporalio import activity
import pymupdf4llm

from utils.helper import parse_s3_path, get_s3_client

from dotenv import load_dotenv
load_dotenv()

AWS_ACCESS_KEY_ID=os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY=os.environ['AWS_SECRET_ACCESS_KEY']
AWS_REGION=os.environ['AWS_REGION']
AWS_S3_ENDPOINT_URL=os.environ['AWS_S3_ENDPOINT_URL']
S3_BUCKET=os.environ['S3_BUCKET']
TEMP_DIR=os.environ['TEMP_DIR']

@dataclass
class DownloadPdfInput:
    s3_path: str

@dataclass
class DownloadPdfOutput:
    file_local_path: str


@activity.defn
def download_pdf(params: DownloadPdfInput):
    bucket, key = parse_s3_path(s3_path=params.s3_path)
    file_name = Path(key).name
    local_path = str(Path(TEMP_DIR) / file_name)

    os.makedirs(TEMP_DIR, exist_ok=True)
    activity.logger.info(f"Downloading s3://{bucket}/{key} to {local_path}")

    s3_client = get_s3_client()
    s3_client.download_file(
        bucket,
        key,
        local_path
    )

    activity.logger.info(f"Downloading is completed: {local_path}")
    return DownloadPdfOutput(file_local_path=local_path)
 