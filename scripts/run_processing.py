"""Submit a SageMaker Processing job that runs one eval config in the cloud.

Runs LOCALLY (laptop or Studio). Uses the SageMaker SDK to launch a Processing
job on a prebuilt sklearn container — no Dockerfile, no ECR. SageMaker:
  1. uploads our code (src/rag_eval + the entry script) to S3,
  2. provisions the instance, installs requirements,
  3. copies input channels S3 -> local, runs processing_entry.py,
  4. copies the output channel local -> S3,
  5. tears the instance down.

Example:
  AWS_PROFILE=dev .venv/bin/python scripts/run_processing.py \
      --config configs/bm25.yaml \
      --role-arn arn:aws:iam::875699230934:role/rag-arch-eval-sagemaker-exec

Data is read from s3://<bucket>/data/, results land under
s3://<bucket>/results/processing/<job-name>/.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from sagemaker.processing import (
    FrameworkProcessor,
    ProcessingInput,
    ProcessingOutput,
)
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.session import Session
import boto3

ROOT = Path(__file__).resolve().parents[1]
# source_dir is src/: it contains the rag_eval package, the entry script, and
# requirements.txt (which SageMaker auto-installs inside the job).
SRC = ROOT / "src"
ENTRY_FILENAME = "processing_entry.py"

DEFAULT_BUCKET = "rag-arch-eval-data-875699230934"

# PyTorch Deep Learning Container (Python 3.10, CPU), resolved via
# sagemaker.image_uris for us-east-1. Chosen because it is:
#   - modern Python (3.10+) — satisfies this harness's syntax/deps, and
#   - ROOT-based — so FrameworkProcessor's code-staging (tar extract into the
#     working dir) works. The SageMaker Distribution image is non-root and its
#     hardening breaks that staging ("tar: Cannot mkdir: Permission denied").
DEFAULT_IMAGE_URI = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.2-cpu-py310"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True, help="local path to a config YAML")
    ap.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--profile", default="dev")
    ap.add_argument("--instance-type", default="ml.t3.large")
    ap.add_argument("--image-uri", default=DEFAULT_IMAGE_URI,
                    help="container image for the job (defaults to SageMaker Distribution, Py3.11+)")
    a = ap.parse_args()

    cfg_name = a.config.stem
    job_name = f"rag-eval-{cfg_name.replace('_', '-')}-{int(time.time())}"[:63]

    boto_sess = boto3.Session(profile_name=a.profile, region_name=a.region)
    sm_session = Session(boto_session=boto_sess, default_bucket=a.bucket)

    # FrameworkProcessor keeps the convenient source_dir + auto-install
    # requirements.txt + entry-script machinery, but we OVERRIDE image_uri to
    # the SageMaker Distribution image (Python 3.11+). estimator_cls=SKLearn is
    # only used by the SDK for packaging conventions; the actual runtime is the
    # image we pass. framework_version is still required by the constructor even
    # with image_uri set.
    processor = FrameworkProcessor(
        estimator_cls=SKLearn,
        framework_version="1.2-1",
        image_uri=a.image_uri,
        role=a.role_arn,
        instance_type=a.instance_type,
        instance_count=1,
        sagemaker_session=sm_session,
        base_job_name="rag-eval",
    )

    inputs = [
        # Dataset parquet: s3://bucket/data/ -> /opt/ml/processing/input/data/
        ProcessingInput(
            source=f"s3://{a.bucket}/data/",
            destination="/opt/ml/processing/input/data",
            input_name="data",
        ),
        # The chosen config, uploaded fresh from local each run.
        ProcessingInput(
            source=str(a.config.resolve()),
            destination="/opt/ml/processing/input/config",
            input_name="config",
        ),
    ]
    outputs = [
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{a.bucket}/results/processing/{job_name}/",
            output_name="results",
        ),
    ]

    print(f"[launch] job={job_name} config={cfg_name} instance={a.instance_type}")
    processor.run(
        code=ENTRY_FILENAME,            # relative to source_dir
        source_dir=str(SRC),            # src/: rag_eval package + requirements.txt
        inputs=inputs,
        outputs=outputs,
        arguments=["--config-name", a.config.name],
        job_name=job_name,
        wait=True,
        logs=True,
    )
    print(f"[launch] results -> s3://{a.bucket}/results/processing/{job_name}/")


if __name__ == "__main__":
    main()
