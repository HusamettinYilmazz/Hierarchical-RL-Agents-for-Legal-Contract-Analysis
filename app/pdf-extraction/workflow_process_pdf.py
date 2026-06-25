"""
Run the process using
$ python app/pdf-extraction/process_pdf.py s3://temporal-dev/math_files/grade_4/math_g4_test.pdf
"""

import os
import sys
from datetime import timedelta
from dataclasses import dataclass

from temporalio import workflow
from temporalio.common import RetryPolicy

from dotenv import load_dotenv
load_dotenv()

with workflow.unsafe.imports_passed_through():
    from activities import (
        download_pdf, DownloadPdfInput,
        extract_markdown, ExtractMarkdownInput,
        upload_markdown, UploadMarkdownInput,
    )


AWS_ACCESS_KEY_ID=os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY=os.environ['AWS_SECRET_ACCESS_KEY']
AWS_REGION=os.environ['AWS_REGION']
AWS_S3_ENDPOINT_URL=os.environ['AWS_S3_ENDPOINT_URL']
S3_BUCKET=os.environ['S3_BUCKET']
TEMP_DIR=os.environ['TEMP_DIR']

@dataclass
class ProcessPdfPipelineInput:
    s3_file_path: str

@dataclass
class ProcessPdfPipelineOutput:
    output_s3_path: str


DEFAULT_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=3),
    backoff_coefficient= 2.0,
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=5,
)
@workflow.defn
class ProcessPdfPipeline:

    @workflow.run
    async def run(self, params: ProcessPdfPipelineInput) -> ProcessPdfPipelineOutput:
        workflow.logger.info(f"Starting pipeline for: {params.s3_file_path}")

        temp_local_path = await workflow.execute_activity(
            activity=download_pdf,
            args=DownloadPdfInput(s3_path=params.s3_file_path),
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=DEFAULT_RETRY_POLICY,
        )

        markdown_text = await workflow.execute_activity(
            activity=extract_markdown,
            args=ExtractMarkdownInput(file_path=temp_local_path),
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=DEFAULT_RETRY_POLICY,
        )

        output_s3_path = await workflow.execute_activity(
            activity=upload_markdown,
            arg=UploadMarkdownInput(markdown_text=markdown_text),
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=DEFAULT_RETRY_POLICY,
        )

        os.remove(temp_local_path)
        workflow.logger.info(f"Pipeline completed.")
        return ProcessPdfPipelineOutput(output_s3_path=output_s3_path)
