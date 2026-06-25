
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

@dataclass
class ExtractMarkdownInput:
    file_path: str

@dataclass
class ExtractMarkdownOutput:
    markdown_text: str

@dataclass
class UploadMarkdownInput:
    markdown_text: str
    original_s3_path: str

@dataclass
class UploadMarkdownOutput:
    output_s3_path: str

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

@activity.defn
def extract_markdown(params: ExtractMarkdownInput):
    activity.logger.info(f"Extracting text from {params.file_path}")
    markdown_text = pymupdf4llm.to_markdown(params.file_path)

    activity.logger.info(f"Extraction completed: {len(markdown_text)} characters extracted")
    return ExtractMarkdownOutput(markdown_text=markdown_text)

@activity.defn
def upload_markdown(params: UploadMarkdownInput):
    bucket, key = parse_s3_path(params.original_s3_path)
    md_key = key.replace(".pdf", ".md")

    activity.logger.info(f"Uploading markdown to s3://{bucket}/{md_key}")
    s3_client = get_s3_client()

    s3_client.put_object(
        Bucket=bucket,
        Key=md_key,
        Body=params.markdown_text.encode("utf-8", errors="ignore"),
        ContentType="text/markdown"
    )

    output_path = f"s3://{bucket}/{md_key}"
    activity.logger.info(f"Uploading completed: {output_path}")

    return UploadMarkdownOutput(output_s3_path=output_path)
