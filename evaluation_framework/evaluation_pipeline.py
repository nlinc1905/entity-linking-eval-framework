from dagster import RunConfig

from dagster_components.assets import GenerateDataConfig
from dagster_components.ops import (
    EngineerFeaturesConfig, TrainModelConfig, ScorePredictionsConfig,
    LogMlflowMetricsConfig, SaveDataConfig,
)
from dagster_components.jobs import evaluate


run_config = RunConfig({
    "generate_data": GenerateDataConfig(
        dataset_size=1000,
        perc_exact_match=0.1,
        perc_non_match=0.5,
        fuzzy_match_corruption_perc=0.5,
        fuzzy_match_mingle_perc=0.25,
        use_pareto_dist_for_degrees=True,
        raw_file_path="eval_data/raw-1000.parquet",
    ),
    "engineer_features": EngineerFeaturesConfig(
        raw_file_path="eval_data/raw-1000.parquet",
        train_test_ratio=0.7,
        train_file_path="eval_data/train-1000.parquet",
        test_file_path="eval_data/test-1000.parquet",
    ),
    "train_model": TrainModelConfig(
        model_key="lr",
    ),
    "score_predictions": ScorePredictionsConfig(
        classifier_result_col="model_score_category",
        id_colname_prefix="index_",
    ),
    "log_mlflow_metrics": LogMlflowMetricsConfig(
        track_mlflow_experiment=True,
    ),
    "save_data_for_dashboard": SaveDataConfig(
        dashboard_data_path="dashboard/data/",
    ),
})

evaluate.execute_in_process(run_config=run_config)
