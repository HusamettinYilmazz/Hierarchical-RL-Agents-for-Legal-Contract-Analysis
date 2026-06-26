
import asyncio
import json_repair
import json
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

        self._review_decision: Optional[str] = None
        self._review_feedback: str = ""
        self._approved_by: str = ""

    @workflow.query
    def get_status(self) -> dict:
        return {
            "status":           self.status,
            "pdfs_processed":   len(self._summaries),
            "report_preview":   self._report[:500] if self._report else None,
            "approved_by":      self._approved_by
        }
    
    @workflow.query
    def get_report(self) -> dict:
        return {
            "status":           self.status,
            "report":           self._report,
            "approved_by":      self._approved_by,
            "sources":          [s["s3_file_path"] for s in self._summaries],
        }

    @workflow.signal
    async def assign_reviewer(self, name: str) -> None:
        self._approved_by = name

    @workflow.update
    async def submit_decision(self, decision: str, feedback: str = "") -> str:
        self._review_decision = decision
        self._review_feedback = feedback

        return f"Decision '{decision}' recorded."

    @submit_decision.validator
    def validate_decision(self, decision: str, feedback: str = "") -> None:
        if decision not in ("approve", "revise"):
            raise ValueError(f"Must be 'approve' or 'revise', got: '{decision}'")
        
        if decision == "revise" and not feedback.strip():
            raise ValueError("Feedback is required when requesting a revision.")

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

        
        for revision_no in range(params.max_revisions + 1):
            self._status = "awaiting-review"
            workflow.logger.info(f"waiting for human review cycle_no: {revision_no}")

            self._review_decision = None

            timed_out = not await workflow.wait_condition(
                lambda: self._review_decision is not None,
                timeout= timedelta(minutes=2),
            )

            if timed_out:
                workflow.logger.warning(f"Review timed out -- auto-completing")
                break

            if self._review_decision == "approve":
                workflow.logger.info(f"Approved by: {self._approved_by}")
                break

            self._status = "revising"
            workflow.logger.info(f"Revising feedback: {self._review_feedback}")

            llm_prompt = _REVISION_PROMPT.format(
                report=json.dumps(
                    self._report, ensure_ascii=False, indent=2
                ),
                feedback=self._review_feedback,
            )

            revised_report = await workflow.execute_activity(
                call_llm,
                CallLLMInput(prompt=llm_prompt),
                start_to_close_timeout=timedelta(minutes=3),
                heartbeat_timeout=timedelta(seconds=100),
                retry_policy=DEFAULT_RETRY_POLICY,
            )

            self._report = json_repair.loads(revised_report.content)
        
        self._status = "completed"

        return ContractReviewOutput(
            report=self._report,
            sources=[s['s3_file_path'] for s in self._summaries],
            approved_by=self._approved_by,
        )
