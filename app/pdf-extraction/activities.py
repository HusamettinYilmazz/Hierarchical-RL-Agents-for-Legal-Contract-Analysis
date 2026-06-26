
import os
from pathlib import Path
from dataclasses import dataclass
import tempfile
import math

from temporalio import activity
import pymupdf4llm
import fitz

from openai import OpenAI

from utils.helper import parse_s3_path, get_s3_client

from dotenv import load_dotenv
load_dotenv()

AWS_ACCESS_KEY_ID=os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY=os.environ['AWS_SECRET_ACCESS_KEY']
AWS_REGION=os.environ['AWS_REGION']
AWS_S3_ENDPOINT_URL=os.environ['AWS_S3_ENDPOINT_URL']
S3_BUCKET=os.environ['S3_BUCKET']
TEMP_DIR=os.environ['TEMP_DIR']

OPENROUTER_API_KEKY=os.environ['OPENROUTER_API_KEKY']
OPENROUTER_MODEL=os.environ['OPENROUTER_MODEL']
OPENROUTER_BASE_URL=os.environ['OPENROUTER_BASE_URL']
MAX_TOKENS=os.environ['MAX_TOKENS']

@dataclass
class DownloadPdfInput:
    s3_file_path: str

@dataclass
class DownloadPdfOutput:
    file_local_path: str

@dataclass
class ExtractMarkdownInput:
    file_path: str
    batch_size: int = 2

@dataclass
class ExtractMarkdownOutput:
    markdown_text: str
    page_count: int

@dataclass
class UploadMarkdownInput:
    markdown_text: str
    original_s3_path: str

@dataclass
class UploadMarkdownOutput:
    output_s3_path: str

@dataclass
class CallLLMInput:
    prompt: str

@dataclass
class CallLLMOutput:
    content: str

@activity.defn
async def download_pdf(params: DownloadPdfInput) -> DownloadPdfOutput:
    activity.heartbeat({
            "stage": "Downloading",
            "status": "Starting",
            "working_file": params.s3_file_path,
        })
    
    bucket, key = parse_s3_path(s3_path=params.s3_file_path)
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
    activity.heartbeat({
            "stage": "Downloading",
            "status": "Finished",
            "working_file": params.s3_file_path,
        })
    
    return DownloadPdfOutput(file_local_path=local_path)

@activity.defn
async def extract_markdown(params: ExtractMarkdownInput) -> ExtractMarkdownOutput:
    activity.logger.info(f"Extracting text from {params.file_path}")
    
    doc = fitz.open(params.file_path)
    page_count = doc.page_count

    all_text_chunks = []
    total_char_num = 0
    num_batches = math.ceil(page_count / params.batch_size)

    activity.heartbeat({
            "stage": "Extracting",
            "status": "Starting",
            "working_file": params.file_path,
            "extracted_pages": f"{0}/{page_count}",
            "extracted_chars": total_char_num
        })
    
    for batch_idx in range(num_batches):
        start_page = batch_idx * params.batch_size
        end_page = min(start_page + params.batch_size, page_count)
        
        batch_md = pymupdf4llm.to_markdown(
            params.file_path,
            pages=list(range(start_page, end_page))
        )
        all_text_chunks.append(batch_md)
        total_char_num += len(batch_md)

        activity.heartbeat({
            "stage": "Extracting",
            "status": "In-Progress",
            "working_file": params.file_path,
            "extracted_pages": f"{end_page}/{page_count}",
            "extracted_chars": total_char_num
        })

    markdown_text = "\n".join(all_text_chunks)
    activity.logger.info(f"Extraction completed: on total {page_count} pages and {total_char_num} characters extracted")
    activity.heartbeat({
            "stage": "Extracting",
            "status": "Finished",
            "working_file": params.file_path,
            "extracted_pages": f"{end_page}/{page_count}",
            "extracted_chars": total_char_num
        })
    return ExtractMarkdownOutput(
        markdown_text=markdown_text,
        page_count=page_count
        )

@activity.defn
async def upload_markdown(params: UploadMarkdownInput)-> UploadMarkdownOutput:
    bucket, key = parse_s3_path(params.original_s3_path)
    md_key = key.replace(".pdf", ".md")
    uploading_path = f"s3://{bucket}/{md_key}"

    activity.logger.info(f"Uploading markdown to {uploading_path}")
    activity.heartbeat({
            "stage": "Uploading",
            "status": "Starting",
            "working_file": uploading_path,
        })
    
    s3_client = get_s3_client()
    s3_client.put_object(
        Bucket=bucket,
        Key=md_key,
        Body=params.markdown_text.encode("utf-8", errors="ignore"),
        ContentType="text/markdown"
    )
    activity.logger.info(f"Uploading completed: {uploading_path}")
    activity.heartbeat({
            "stage": "Uploading",
            "status": "Finished",
            "working_file": uploading_path,
        })
    
    return UploadMarkdownOutput(output_s3_path=uploading_path)

@activity.defn
async def call_llm(params: CallLLMInput) -> CallLLMOutput:
    activity.logger.info(f"Calling LLM: {params.llm_model}")
    activity.heartbeat({
            "stage": "Calling-llm",
            "status": "Starting",
            "prompt_chars": len(params.prompt),
        })
    
    llm_client = OpenAI(
        api_key=OPENROUTER_API_KEKY,
        base_url=OPENROUTER_BASE_URL,
    )

    response = llm_client.chat.completions.create(
        model= OPENROUTER_MODEL,
        messages=[{
            "role": "user",
            "content": params.prompt
            }],
        max_tokens=int(MAX_TOKENS),
    )

    content = response.choices[0].message.content

    return CallLLMOutput(llm_response=content)
    
