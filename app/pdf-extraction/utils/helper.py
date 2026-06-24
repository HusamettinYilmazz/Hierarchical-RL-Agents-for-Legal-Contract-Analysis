
import os
from pathlib import Path

import boto3
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID=os.environ['AWS_ACCESS_KEY_ID']
AWS_SECRET_ACCESS_KEY=os.environ['AWS_SECRET_ACCESS_KEY']
AWS_REGION=os.environ['AWS_REGION']
AWS_S3_ENDPOINT_URL=os.environ['AWS_S3_ENDPOINT_URL']
S3_BUCKET=os.environ['S3_BUCKET']
TEMP_DIR=os.environ['TEMP_DIR']


def get_s3_client():

    return boto3.client(
        service_name="s3",
        region_name= AWS_REGION,
        endpoint_url=AWS_S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,

    )

def parse_s3_path(s3_path: str):
    s3_path = s3_path.replace("s3://", "")
    buket, _, key = s3_path.partition("/")

    return buket, key
