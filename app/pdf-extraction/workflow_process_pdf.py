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

@workflow.defn
class ProcessPdfPipeline:

    @workflow.run
    async def run(self, params: ProcessPdfPipelineInput) -> ProcessPdfPipelineOutput:
        workflow.logger.info(f"Starting pipeline for: {params.s3_file_path}")

        temp_local_path = await workflow.execute_activity(
            download_pdf,
            DownloadPdfInput(s3_path=params.s3_file_path),
            
        )

        markdown_text = await workflow.execute_activity(
            extract_markdown,
            ExtractMarkdownInput(file_path=temp_local_path),

        )

        output_s3_path = await workflow.execute_activity(
            upload_markdown,
            UploadMarkdownInput(markdown_text=markdown_text),

        )

        os.remove(temp_local_path)
        workflow.logger.info(f"Pipeline completed.")
        return ProcessPdfPipelineOutput(output_s3_path=output_s3_path)


if __name__ == "__main__":

    input_params = ProcessPdfPipelineInput(s3_file_path=sys.argv[1])
    # _ = process_pdf(params=input_params)
