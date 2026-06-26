
import textwrap
from dataclasses import dataclass
from datetime import timedelta

import json_repair
from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from activities import(
        download_pdf, DownloadPdfInput,
        extract_markdown, ExtractMarkdownInput,
        call_llm, CallLLMInput,
    )

from utils.prompt import _SUMMARY_PROMPT


@dataclass
class SummarizePDFInput:
    s3_file_path: str

@dataclass
class SummarizePDFOutput:
    s3_file_path: str
    summary: str
    risks: str

DEFAULT_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=60),
    maximum_attempts=3
)

workflow.defn
class SummarizePDF:
    
    @workflow.run
    async def run(self, params: SummarizePDFInput) -> SummarizePDFOutput:
        workflow.logger.info(f"Starting Summarizing PDF workflow for: {params.s3_file_path}")
        
        downloaded_pdf = await workflow.execute_activity(
            download_pdf,
            DownloadPdfInput(s3_file_path=params.s3_file_path),
            retry_policy=DEFAULT_RETRY_POLICY,
            start_to_close_timeout=timedelta(minutes=2),
            heartbeat_timeout=timedelta(seconds=30),
        )

        extracted_md = await workflow.execute_activity(
            extract_markdown,
            ExtractMarkdownInput(file_path=downloaded_pdf.file_local_path),
            retry_policy=DEFAULT_RETRY_POLICY,
            start_to_close_timeout=timedelta(minutes=4),
            heartbeat_timeout=timedelta(seconds=30),
        )

        prompt = _SUMMARY_PROMPT.format(text=extracted_md.markdown_text)
        llm_response = await workflow.execute_activity(
            call_llm,
            CallLLMInput(prompt=prompt),
            retry_policy=DEFAULT_RETRY_POLICY,
            start_to_close_timeout=timedelta(minutes=5),
            heartbeat_timeout=timedelta(minutes=2),
        )

        parsed_output = json_repair.loads(llm_response.content)

        return SummarizePDFOutput(
            s3_file_path=params.s3_file_path,
            summary=parsed_output.get("summary", ""),
            risks=parsed_output.get("risks", "")
        )
