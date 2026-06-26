
import asyncio
import textwrap
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError
from temporalio.workflow import ParentClosePolicy

with workflow.unsafe.imports_passed_through():
    from activities import (
        call_llm, CallLLMInput,
    )
    from child_workflow import (
        SummarizePDF, SummarizePDFInput,
    )

@dataclass
class ContractReviewInput:
    s3_file_paths: str
    max_revisions: int = 2

@dataclass
class ContractReviewOutput:
    report: str
    sources: list
    approved_by: str

@workflow.defn
class ContractReviewWorkflow:
    def __init__(self):
        self.status: str = "Processing"
        self._summaries: list = []
        self._report: str = ""

    @workflow.run
    async def run(self, params: ContractReviewInput) -> ContractReviewOutput:
        self.status = "Extracting"
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
            *[
                h.result()
                for h in handles            
            ],
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


        