import sagemaker
from sagemaker.model import Model
from sagemaker import serializers, deserializers
from sagemaker import image_uris
import boto3
import os
import shutil
import time
import json
import jinja2
import tarfile
from tqdm import tqdm
from pathlib import Path
import boto3
import sagemaker
from sagemaker import get_execution_role
from PIL import Image
import numpy as np
import argparse
import yaml
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--cfg", help="config file for model deployment", required=True, type=str
)
args = parser.parse_args()
with open(args.cfg, "r") as f:
    cfg = yaml.safe_load(f)


# login to aws
try:
    # role = sagemaker.get_execution_role()  # execution role for the endpoint
    # hardcode the role
    role = cfg["AWS_ROLE"]
    sess = (
        sagemaker.session.Session()
    )  # sagemaker session for interacting with different AWS APIs
    bucket = sess.default_bucket()  # bucket to house artifacts

    region = (
        sess._region_name
    )  # region name of the current SageMaker Studio environment
    account_id = (
        sess.account_id()
    )  # account_id of the current SageMaker Studio environment
except:
    raise Exception(
        "Cannot get aws configs, please config aws before deploying sagemaker endpoints"
    )

logging.info(f"Log into aws\nRole: {role}\n")


# use boto3 session
session = boto3.Session()
boto3_kwargs = {}

boto3_sm_run_client = boto3.client(
    "sagemaker-runtime", region_name=region, **boto3_kwargs
)

boto3_sm_client = boto3.client("sagemaker", region_name=region, **boto3_kwargs)

boto3_s3_client = boto3.client("s3", region_name=region, **boto3_kwargs)

boto3_iam_client = boto3.client("iam", region_name=region, **boto3_kwargs)


# prepare `model artifact`
# remove `artifact` folder and make a new one
try:
    shutil.rmtree(cfg["MODEL_ARTIFACT_PATH"])
    logging.info("The model artifact folder has been deleted")
except OSError as e:
    if e.errno == 2:
        logging.warning("The model artifact folder does not exist")
    else:
        logging.error(f"Error occurred: {e}")
os.mkdir(cfg["MODEL_ARTIFACT_PATH"])
logging.info("Make a new `model artifact` folder")

# move serving.properties to artifact folder
shutil.copy(cfg["SERVING_PROPERTIES_PATH"], cfg["MODEL_ARTIFACT_PATH"])

# move requirements.txt to artifact folder
shutil.copy(cfg["REQUIREMENTS_PATH"], cfg["MODEL_ARTIFACT_PATH"])

# move models to artifact folder: with weights and model.py
try:
    for f in os.listdir(cfg["MODEL_PATH"]):
        p = os.path.join(cfg["MODEL_PATH"], f)
        shutil.copy(p, cfg["MODEL_ARTIFACT_PATH"])
except IsADirectoryError:
    logging.error(f"Put all files in {cfg['MODEL_PATH']}, no sub-folders!")
logging.info("Copied all files in `model path` into model artifact folder")

# compress artifact folder as model.tar.gz
file_list = []
for root, dirs, files in os.walk(cfg["MODEL_ARTIFACT_PATH"]):
    for file in files:
        fp = os.path.join(cfg["MODEL_ARTIFACT_PATH"], file)
        file_list.append(fp)

with tarfile.open("model.tar.gz", "w:gz") as tar:
    progress_bar = tqdm(total=len(file_list), unit="file", desc="Creating model.tar.gz")
    for file_name in file_list:
        tar.add(file_name)
        progress_bar.update(1)
    progress_bar.close()

# upload model.tar.gz to s3
logging.info("Uploading model.tar.gz to aws S3...")
boto3_s3_client.upload_file(
    "model.tar.gz", cfg["S3_BUCKET"], f"{cfg['S3_ARTIFACT_PREFIX']}/model.tar.gz"
)
logging.info("model.tar.gz uploaded:")
logging.info(
    boto3_s3_client.list_objects(
        Bucket=cfg["S3_BUCKET"], Prefix=cfg["S3_ARTIFACT_PREFIX"]
    ).get("Contents", [])
)

# create model
model_name = f"{cfg['DEPLOYED_MODEL_NAME']}-v{cfg['VERSION']}"
s3_code_artifact = f"s3://{cfg['S3_BUCKET']}/{cfg['S3_ARTIFACT_PREFIX']}/model.tar.gz"
logging.info(f"ModelDataUrl: {s3_code_artifact}")
create_model_response = boto3_sm_client.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role,
    PrimaryContainer={
        "Image": cfg["INFERENCE_IMAGE_URI"],
        "ModelDataUrl": s3_code_artifact,
    },
)
model_arn = create_model_response["ModelArn"]
logging.info(f"Created Model: {model_arn}")

# create endpoint config
endpoint_config_name = f"{model_name}-config"
endpoint_config_response = boto3_sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "VariantName": "sd2-inf2-newuri",
            "ModelName": model_name,
            "InstanceType": "ml.inf2.xlarge",  # -
            "InitialInstanceCount": 1,
            "ContainerStartupHealthCheckTimeoutInSeconds": 360,
            "VolumeSizeInGB": 400,
        },
    ],
)
logging.info(
    f"Created Endpoint Config: {endpoint_config_response['EndpointConfigArn']}"
)

# deploy endpoint
endpoint_name = f"{model_name}-endpoint"
create_endpoint_response = boto3_sm_client.create_endpoint(
    EndpointName=f"{endpoint_name}", EndpointConfigName=endpoint_config_name
)
logging.info(f"Created Endpoint: {create_endpoint_response['EndpointArn']}")

# check endpoint deploy status
resp = boto3_sm_client.describe_endpoint(EndpointName=endpoint_name)
status = resp["EndpointStatus"]
logging.info(f"Status: {status}")

while status == "Creating":
    time.sleep(60)
    resp = boto3_sm_client.describe_endpoint(EndpointName=endpoint_name)
    status = resp["EndpointStatus"]
    logging.info(f"Status: {status}")
logging.info(f"Arn: {resp['EndpointArn']}")
logging.info(f"Status: {status}")
