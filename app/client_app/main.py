
import os
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from temporalio.client import Client


from dotenv import load_dotenv
load_dotenv()

TEMPORAL_HOST=os.environ['TEMPORAL_HOST']
TEMPORAL_NAMESPACE=os.environ['TEMPORAL_NAMESPACE']
TEMPORAL_PDF_PROCESS_TASK_QUEUE=os.environ['TEMPORAL_PDF_PROCESS_TASK_QUEUE']

app = FastAPI(
    title="PDF Extraction Client",
    version="1.0"
)

class ProcessPdfRequest(BaseModel):
    s3_file_path: str

class ProcessPdfExecuteResponse(BaseModel):
    workflow_id: str
    results: dict


async def get_temporal_client() -> Client:
    return await Client.connect(
        target_host=TEMPORAL_HOST,
        namespace=TEMPORAL_NAMESPACE,
    )

@app.get("/health")
async def health():
    return {"status": "amazing :)"}

@app.post("/process-pdf/execute", response_model=ProcessPdfExecuteResponse)
async def process_pdf(request: ProcessPdfRequest):

    workflow_id = f"pdf-pipeline-{uuid.uuid4()}"

    temp_client = await get_temporal_client()

    results = await temp_client.execute_workflow(
        workflow="ProcessPdfPipeline",
        args=[
            {
                "s3_file_path": request.s3_file_path,
            }
        ],
        id=workflow_id,
        task_queue=TEMPORAL_PDF_PROCESS_TASK_QUEUE,
        result_type=dict,
    )

    return ProcessPdfExecuteResponse(
        workflow_id=workflow_id,
        results=results
    )



"""
uvicorn main:app --reload --port 5000
curl -X POST http://localhost:5000/process-pdf/execute \
    -H "Content-Type: application/json" \
    -d '{"s3_file_path": "s3://temporal-dev/math_files/grade_4/math_g4_test.pdf"}'

curl -X POST http://localhost:5000/process-pdf/start \
    -H "Content-Type: application/json" \
    -d '{"s3_file_path": "s3://temporal-dev/math_files/grade_4/math_g4_test.pdf"}'    
"""