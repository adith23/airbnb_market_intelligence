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
    name='gcs-fuse-csi-vol',
    csi=k8s.V1CSIVolumeSource(
        driver="gcsfuse.csi.storage.gke.io",
        volume_attributes={"bucketName": DATA_BUCKET}
    )
)
gcs_volume_mount = k8s.V1VolumeMount(
    name='gcs-fuse-csi-vol',
    mount_path='/app/shared_data',
    read_only=False
)

default_args = {
    'owner': 'data_engineering_team',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def create_k8s_task(task_id: str, cmd_args: list) -> KubernetesPodOperator:
    return KubernetesPodOperator(
        task_id=task_id,
        name=f"pod-{task_id}",
        namespace='composer-user-workloads',
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
    'data_pipeline_airbnb',
    default_args=default_args,
    description='End-to-End Airbnb Data Pipeline (Dockerized)',
    schedule_interval='@daily',
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['data_engineering', 'airbnb', 'kubernetes'],
) as dag:

    ingest_task = create_k8s_task('ingest_data', ["ingest", "--city", CITY])
    profile_task = create_k8s_task('profile_data', ["profile", "--city", CITY])
    quality_task = create_k8s_task('quality_report', ["quality-report", "--city", CITY])
    validate_task = create_k8s_task('validate_data', ["validate", "--city", CITY])
    clean_task = create_k8s_task('clean_data', ["clean", "--city", CITY])
    enrich_task = create_k8s_task('enrich_data', ["enrich", "--city", CITY])
    model_task = create_k8s_task('model_data', ["model", "--cities", CITY])

    ingest_task >> profile_task >> quality_task >> validate_task >> clean_task >> enrich_task >> model_task
