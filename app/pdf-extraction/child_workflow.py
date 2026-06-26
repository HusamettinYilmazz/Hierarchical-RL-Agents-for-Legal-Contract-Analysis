
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

workflow.defn
class SummarizePDF:
    
    @workflow.run
    async def run(self, params: SummarizePDFInput) -> SummarizePDFOutput:
        workflow.logger.info(f"Starting Summarizing PDF workflow for: {params.s3_file_path}")
        
        downloaded_pdf = await workflow.execute_activity(
            download_pdf,
            DownloadPdfInput(s3_file_path=params.s3_file_path),    
        )

        extracted_md = await workflow.execute_activity(
            extract_markdown,
            ExtractMarkdownInput(file_path=downloaded_pdf.file_local_path),
        )

        prompt = _SUMMARY_PROMPT.format(text=extracted_md.markdown_text)
        llm_response = await workflow.execute_activity(
            call_llm,
            CallLLMInput(prompt=prompt)
        )
