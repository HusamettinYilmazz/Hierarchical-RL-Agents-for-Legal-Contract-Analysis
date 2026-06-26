
import asyncio
import json_repair
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError
from temporalio.workflow import ParentClosePolicy

from utils.prompt import _SYNTHESIS_PROMPT, _REVISION_PROMPT

with workflow.unsafe.imports_passed_through():
    from activities import (
        call_llm, CallLLMInput,
    )
    from child_workflow import (
        SummarizePDF, SummarizePDFInput,
    )

@dataclass
class ContractReviewInput:
    s3_file_paths: list[str]
    max_revisions: int = 2

@dataclass
class ContractReviewOutput:
    report: str
    sources: list
    approved_by: str

DEFAULT_RETRY_POLICY = RetryPolicy(
    initial_interval=timedelta(seconds=3),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=60),
    maximum_attempts=4,
)

@workflow.defn
class ContractReviewWorkflow:
    def __init__(self):
        self.status: str = "processing"
        self._summaries: list = []
        self._report: str = ""

    @workflow.run
    async def run(self, params: ContractReviewInput) -> ContractReviewOutput:
        self.status = "extracting"
        workflow.logger.info(f"Fanning out to {len(params.s3_file_paths)} child workflows")

        workflow_id = workflow.info().workflow_id
        workflow_task_queue = workflow.info().task_queue

        handles = await asyncio.gather(
            *[
                workflow.start_child_workflow(
                    SummarizePDF.run,
                    SummarizePDFInput(s3_file_path=cur_s3_file_path),
                    id=f"{workflow_id}-pdf-{idx+1}",
                    task_queue=workflow_task_queue,
                    parent_close_policy=ParentClosePolicy.ABANDON
                )

                for idx, cur_s3_file_path in enumerate(params.s3_file_paths)
            ]
        )

        raw_results = await asyncio.gather(
            *handles,
            return_exceptions=True,   
        )

        for idx, res in enumerate(raw_results):
            if isinstance(res, Exception):
                workflow.logger.warning(f"PDF {idx} failed: {res}")
            else:
                self._summaries.append({
                    "s3_file_path": res.s3_file_path,
                    "summary":      res.summary,
                    "risks":        res.risks,
                })

        if len(self._summaries) == 0:
            raise ApplicationError("All PDFs failed to process.")

        self.status = "analyzing"
        workflow.logger.info(f"Syntheszing {len(self._summaries)}summaries")

        combined_summaries = "\n\n".join([
            f"**Contract {i+1}** (`{summary['s3_file_path']}`):\n"
            f"**Summary** {summary['summary']}\n"
            f"**Risks** {summary['risks']}\n"

            for i, summary in enumerate(self._summaries)
        ])

        llm_prompt = _SYNTHESIS_PROMPT.format(
            summaries=combined_summaries,
            n=len(self._summaries)
        )

        llm_result = await workflow.execute_activity(
            call_llm,
            CallLLMInput(prompt=llm_prompt),
            start_to_close_timeout=timedelta(minutes=4),
            heartbeat_timeout=timedelta(minutes=3),
            retry_policy=DEFAULT_RETRY_POLICY,
        )

        self._report = json_repair.loads(llm_result.content)

        return ContractReviewOutput(
            report=self._report,
            sources=params.s3_file_paths,
            approved_by=""
        )