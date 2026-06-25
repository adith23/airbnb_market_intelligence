import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from kubernetes.client import models as k8s

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "default_project")
GCP_REGION = os.environ.get("GCP_REGION", "us-central1")
ENV = os.environ.get("ENV", "prod")
CITY = os.environ.get("TARGET_CITY", "new-york-city")

# GCS FUSE Volume configuration to share data across isolated Pods natively
DATA_BUCKET = f"airbnb-processed-data-{GCP_PROJECT_ID}-{ENV}"
gcs_volume = k8s.V1Volume(
    name="gcs-fuse-csi-vol",
    csi=k8s.V1CSIVolumeSource(
        driver="gcsfuse.csi.storage.gke.io", volume_attributes={"bucketName": DATA_BUCKET}
    ),
)
gcs_volume_mount = k8s.V1VolumeMount(
    name="gcs-fuse-csi-vol", mount_path="/app/shared_data", read_only=False
)

default_args = {
    "owner": "ml_engineering_team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def create_k8s_task(task_id: str, cmd_args: list) -> KubernetesPodOperator:
    return KubernetesPodOperator(
        task_id=task_id,
        name=f"pod-{task_id}",
        namespace="composer-user-workloads",
        image=f"{GCP_REGION}-docker.pkg.dev/{GCP_PROJECT_ID}/airbnb-ml-models-{ENV}/pipeline:latest",
        cmds=["python", "main.py"] + cmd_args,
        annotations={"gke-gcsfuse/volumes": "true"},
        volumes=[gcs_volume],
        volume_mounts=[gcs_volume_mount],
        env_vars={"AIRFLOW_DATA_DIR": "/app/shared_data"},
        get_logs=True,
        is_delete_operator_pod=True,
    )


with DAG(
    "ml_pipeline_airbnb",
    default_args=default_args,
    description="Airbnb ML Training and Evaluation Pipeline (Dockerized)",
    schedule_interval="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["ml_engineering", "airbnb", "kubernetes"],
) as dag:
    feature_eng_task = create_k8s_task("feature_engineering", ["feature-engineer", "--city", CITY])
    train_model_task = create_k8s_task("train_model", ["train", "--city", CITY])
    evaluate_model_task = create_k8s_task("evaluate_model", ["evaluate", "--city", CITY])
    explain_task = create_k8s_task("explain_model", ["explain", "--city", CITY])
    bias_audit_task = create_k8s_task("bias_audit", ["bias-audit", "--city", CITY])

    feature_eng_task >> train_model_task >> evaluate_model_task >> explain_task >> bias_audit_task
