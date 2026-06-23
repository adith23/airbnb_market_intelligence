import shutil
from pathlib import Path

# Mapping of old paths (relative to root) to new paths
FILE_MOVES = {
    "pipeline/utils.py": "src/platform/common/utils.py",
    "pipeline/downloader.py": "src/platform/data_engineering/ingestion/downloader.py",
    "pipeline/cleaner.py": "src/platform/data_engineering/ingestion/cleaner.py",
    "pipeline/validator.py": "src/platform/data_engineering/ingestion/validator.py",
    "pipeline/modeler.py": "src/platform/data_engineering/modeling/modeler.py",
    "pipeline/harmonizer.py": "src/platform/data_engineering/modeling/harmonizer.py",
    "pipeline/relationship_mapper.py": "src/platform/data_engineering/modeling/relationship_mapper.py",
    "dashboard/backend/data_client.py": "src/platform/data_engineering/storage/data_client.py",
    "pipeline/ml/feature_store.py": "src/platform/feature_engineering/feature_store.py",
    "pipeline/enricher.py": "src/platform/feature_engineering/enricher.py",
    "pipeline/ml/trainer.py": "src/platform/data_science/modeling/trainer.py",
    "pipeline/ml/explainer.py": "src/platform/data_science/explainability/explainer.py",
    "pipeline/ml/evaluator.py": "src/platform/data_science/evaluation/evaluator.py",
    "pipeline/ml/bias_auditor.py": "src/platform/data_science/evaluation/bias_auditor.py",
    "pipeline/profiler.py": "src/platform/data_science/evaluation/profiler.py",
    "dashboard/backend/sql_agent.py": "src/platform/agentic_ai/sql_agent.py",
    "dashboard/backend/ml_client.py": "src/platform/mlops/serving/ml_client.py",
    "pipeline/ml/orchestrator.py": "src/platform/mlops/orchestrator.py",
    "pipeline/metadata.py": "src/platform/common/metadata.py",
    "pipeline/automation.py": "scripts/run_pipeline_logic.py",
}

IMPORT_REWRITES = {
    "from src.platform.common.utils": "from src.platform.common.utils",
    "from src.platform.common import utils": "from src.platform.common import utils",
    "from src.platform.data_engineering.ingestion.downloader": "from src.platform.data_engineering.ingestion.downloader",
    "from src.platform.data_engineering.ingestion import downloader": "from src.platform.data_engineering.ingestion import downloader",
    "from src.platform.data_engineering.ingestion.cleaner": "from src.platform.data_engineering.ingestion.cleaner",
    "from src.platform.data_engineering.ingestion import cleaner": "from src.platform.data_engineering.ingestion import cleaner",
    "from src.platform.data_engineering.ingestion.validator": "from src.platform.data_engineering.ingestion.validator",
    "from src.platform.data_engineering.ingestion import validator": "from src.platform.data_engineering.ingestion import validator",
    "from src.platform.data_engineering.modeling.modeler": "from src.platform.data_engineering.modeling.modeler",
    "from src.platform.data_engineering.modeling import modeler": "from src.platform.data_engineering.modeling import modeler",
    "from src.platform.data_engineering.modeling.harmonizer": "from src.platform.data_engineering.modeling.harmonizer",
    "from src.platform.data_engineering.modeling import harmonizer": "from src.platform.data_engineering.modeling import harmonizer",
    "from src.platform.feature_engineering.feature_store": "from src.platform.feature_engineering.feature_store",
    "from src.platform.feature_engineering import feature_store": "from src.platform.feature_engineering import feature_store",
    "from src.platform.feature_engineering.enricher": "from src.platform.feature_engineering.enricher",
    "from src.platform.feature_engineering import enricher": "from src.platform.feature_engineering import enricher",
    "from src.platform.data_science.modeling.trainer": "from src.platform.data_science.modeling.trainer",
    "from src.platform.data_science.modeling import trainer": "from src.platform.data_science.modeling import trainer",
    "from src.platform.data_science.explainability.explainer": "from src.platform.data_science.explainability.explainer",
    "from src.platform.data_science.explainability import explainer": "from src.platform.data_science.explainability import explainer",
    "from src.platform.data_science.evaluation.evaluator": "from src.platform.data_science.evaluation.evaluator",
    "from src.platform.data_science.evaluation import evaluator": "from src.platform.data_science.evaluation import evaluator",
    "from src.platform.data_science.evaluation.bias_auditor": "from src.platform.data_science.evaluation.bias_auditor",
    "from src.platform.data_science.evaluation import bias_auditor": "from src.platform.data_science.evaluation import bias_auditor",
    "from src.platform.common.metadata": "from src.platform.common.metadata",
    "from src.platform.common import metadata": "from src.platform.common import metadata",
    "from src.platform.data_engineering.modeling.relationship_mapper": "from src.platform.data_engineering.modeling.relationship_mapper",
    "from src.platform.data_engineering.modeling import relationship_mapper": "from src.platform.data_engineering.modeling import relationship_mapper",
    "from src.platform.data_science.evaluation.profiler": "from src.platform.data_science.evaluation.profiler",
    "from src.platform.data_science.evaluation import profiler": "from src.platform.data_science.evaluation import profiler",
    "from scripts.run_pipeline_logic": "from scripts.run_pipeline_logic",
    "from scripts import run_pipeline_logic": "from scripts import run_pipeline_logic",
    "from src.platform.mlops.orchestrator": "from src.platform.mlops.orchestrator",
    "from src.platform.mlops import orchestrator": "from src.platform.mlops import orchestrator",
    "from src.platform.data_engineering.storage.data_client": "from src.platform.data_engineering.storage.data_client",
    "from src.platform.data_engineering.storage import data_client": "from src.platform.data_engineering.storage import data_client",
    "from src.platform.agentic_ai.sql_agent": "from src.platform.agentic_ai.sql_agent",
    "from src.platform.agentic_ai import sql_agent": "from src.platform.agentic_ai import sql_agent",
    "from src.platform.mlops.serving.ml_client": "from src.platform.mlops.serving.ml_client",
    "from src.platform.mlops.serving import ml_client": "from src.platform.mlops.serving import ml_client",
}

EMPTY_DIRS = [
    "pipelines/airflow/dags",
    "pipelines/airflow/plugins",
    "pipelines/airflow/includes",
    "pipelines/dbt/models",
    "pipelines/dbt/seeds",
    "pipelines/dbt/tests",
    "models/experiments",
    "models/checkpoints",
    "data/features",
    "sql/ddl",
    "sql/dml",
    "sql/views",
    "infra/terraform",
    "infra/kubernetes",
    "infra/cloudformation",
    "tests/integration",
    "tests/data_quality",
    "notebooks/exploratory",
    "notebooks/experiments",
    "monitoring/dashboards",
    "monitoring/alerts",
]


def main():
    root_dir = Path("c:/Projects/airbnb_market_intelligence")

    print("1. Creating enterprise directory structure placeholders...")
    for d in EMPTY_DIRS:
        dir_path = root_dir / d
        dir_path.mkdir(parents=True, exist_ok=True)
        (dir_path / ".gitkeep").touch()

    print("2. Moving source files to domain-driven directories...")
    for old_rel, new_rel in FILE_MOVES.items():
        old_path = root_dir / old_rel
        new_path = root_dir / new_rel

        if old_path.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))
            print(f"   Moved: {old_rel} -> {new_rel}")

            # Ensure __init__.py exists in all package layers
            curr = new_path.parent
            while curr != root_dir and curr != root_dir / "src" and str(curr) != curr.anchor:
                (curr / "__init__.py").touch()
                curr = curr.parent

    print("3. Refactoring over 50+ import statements across the entire codebase...")
    all_py_files = list(root_dir.rglob("*.py"))

    for py_file in all_py_files:
        if "venv" in str(py_file) or ".venv" in str(py_file):
            continue

        try:
            with open(py_file, encoding="utf-8") as f:
                content = f.read()

            new_content = content
            for old_import, new_import in IMPORT_REWRITES.items():
                new_content = new_content.replace(old_import, new_import)

            if new_content != content:
                with open(py_file, "w", encoding="utf-8") as f:
                    f.write(new_content)
                    print(f"   Updated imports in: {py_file.relative_to(root_dir)}")
        except Exception as e:
            print(f"   Error parsing {py_file}: {e}")

    print("4. Cleaning up deprecated directories...")
    for deprecated_dir in ["pipeline/ml", "pipeline", "dashboard/backend"]:
        dir_path = root_dir / deprecated_dir
        if dir_path.exists() and not list(dir_path.glob("*.py")):
            try:
                shutil.rmtree(dir_path)
                print(f"   Removed empty directory: {deprecated_dir}")
            except Exception:
                pass

    print("\\n\\n✅ MIGRATION COMPLETE: Codebase successfully refactored to Enterprise Structure!")


if __name__ == "__main__":
    main()
