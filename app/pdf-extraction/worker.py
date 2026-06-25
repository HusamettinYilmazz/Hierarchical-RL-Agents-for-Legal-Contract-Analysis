
import os
import asyncio
import logging

from temporalio.client import Client
from temporalio.worker import Worker

from dotenv import load_dotenv
load_dotenv()

from workflow_process_pdf import ProcessPdfPipeline
from activities import (
    download_pdf, DownloadPdfInput,
    extract_markdown, ExtractMarkdownInput,
    upload_markdown, UploadMarkdownInput,
)

TEMPORAL_HOST=os.environ['TEMPORAL_HOST']
TEMPORAL_NAMESPACE=os.environ['TEMPORAL_NAMESPACE']
TEMPORAL_PDF_PROCESS_TASK_QUEUE=os.environ['TEMPORAL_PDF_PROCESS_TASK_QUEUE']


async def main():
    temporal_client = await Client.connect(
        target_host=TEMPORAL_HOST,
        namespace=TEMPORAL_NAMESPACE,
    )

    worker_pdf_process = Worker(
        client=temporal_client,
        task_queue=TEMPORAL_PDF_PROCESS_TASK_QUEUE,
        activities=[download_pdf, extract_markdown, upload_markdown],
        workflows=[ProcessPdfPipeline],
    )
    print(f"Worker started and pooling task queue: {TEMPORAL_PDF_PROCESS_TASK_QUEUE}")
    await worker_pdf_process.run()



if __name__ == "__main__":
    asyncio.run(main())