
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

class ProcessPdfStartResponse(BaseModel):
    workflow_id: str

class StartReviewRequest(BaseModel):
    s3_file_paths: list[str]
    max_revisions: int = 2

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

@app.post("/process-pdf/start", response_model=ProcessPdfStartResponse)
async def process_pdf(request: ProcessPdfRequest):

    workflow_id = f"pdf-pipeline-{uuid.uuid4()}"

    temp_client = await get_temporal_client()

    results = await temp_client.start_workflow(
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

    return ProcessPdfStartResponse(
        workflow_id=workflow_id
    )

@app.get("/process-pdf/status/{workflow_id}")
async def get_workflow_status(workflow_id: str):

    client = await get_temporal_client()
    
    handle = client.get_workflow_handle(
        workflow_id,
        result_type=dict
    )
    desc = await handle.describe()

    try:
        result = await handle.result()
    except:
        result = None

    workflow_status = desc.status

    return {
        "workflow_id": workflow_id,
        "workflow_status": workflow_status.name,
        "workflow_result": result
    }

@app.post("/contract-review/start", response_model=ProcessPdfStartResponse)
async def start_contract_review(request: StartReviewRequest):
    workflow_id = f"contract-review-{uuid.uuid4()}"

    client = await get_temporal_client()

    await client.start_workflow(
        "ContractReviewWorkflow",
        args=[{
            "s3_file_paths": request.s3_file_paths,
            "max_revisions": request.max_revisions
        }],
        id=workflow_id,
        task_queue=TEMPORAL_PDF_PROCESS_TASK_QUEUE
    )

    return {"workflow_id": workflow_id}

"""
uvicorn main:app --reload --port 5000

### Process PDF

curl -X POST http://localhost:5000/process-pdf/execute \
    -H "Content-Type: application/json" \
    -d '{"s3_file_path": "s3://temporal-dev/math_files/grade_4/math_g4_test.pdf"}'

curl -X POST http://localhost:5000/process-pdf/start \
    -H "Content-Type: application/json" \
    -d '{"s3_file_path": "s3://temporal-dev/math_files/grade_4/math_g4_test.pdf"}'    

    
### Contract Review

curl -X POST http://localhost:5000/contract-review/start \
    -H "Content-Type: application/json" \
    -d '{
    "s3_file_paths": ["s3://temporal-dev/math_files/grade_4/math_g4_test.pdf",
         "s3://temporal-dev/math_files/grade_4/math_g4_test1.pdf",
         "s3://temporal-dev/math_files/grade_4/math_g4_test2.pdf"
        ]
    }'    


curl -X POST http://localhost:5000/contract-review/start \
    -H "Content-Type: application/json" \
    -d '{
    "s3_file_paths": ["s3://temporal-dev/legal_documents/nda-innovate-consultpro.pdf",
         "s3://temporal-dev/legal_documents/software-license-globalsoft.pdf",
         "s3://temporal-dev/legal_documents/vendor-service-agreement.pdf"
        ]
    }'    
 
"""