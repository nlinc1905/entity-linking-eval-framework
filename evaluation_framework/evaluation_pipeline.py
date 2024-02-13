import argparse
import os
from dagster import RunConfig
import pandas as pd
import plotly.express as px

from dagster_components.assets import GenerateDataConfig
from dagster_components.ops import (
    EngineerFeaturesConfig, TrainModelConfig,
    LogMlflowMetricsConfig, SaveDataConfig,
)
from dagster_components.jobs import evaluate


default_config = {
    "generate_data": GenerateDataConfig(
        dataset_size=1000,
        perc_exact_match=0.1,
        perc_non_match=0.1,
        fuzzy_match_corruption_perc=0.5,
        fuzzy_match_mingle_perc=0.5,
        use_pareto_dist_for_degrees=True,
        raw_file_path="eval_data/raw-1000.parquet",
    ),
    "engineer_features": EngineerFeaturesConfig(
        raw_file_path="eval_data/raw-1000.parquet",
        train_test_ratio=0.7,
    ),
    "train_model": TrainModelConfig(
        model_name="lr",
    ),
    "log_mlflow_metrics": LogMlflowMetricsConfig(
        track_mlflow_experiment=True,
    ),
    "save_data_for_dashboard": SaveDataConfig(
        dashboard_data_path="dashboard/data/",
    ),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs the evaluation pipeline from the command line.")
    parser.add_argument(
        "--run_type",
        help=(
            "'single' for single runs with default config, or "
            "'multi' for running several times for different combos of corruption and mingling."
        )
    )
    args = parser.parse_args()

    if args.run_type == "multi":

        # run several evaluations for various combinations of corruption and mingling
        corruption_percentages = [i / 10 for i in range(1, 10)]
        mingle_percentages = [i / 10 for i in range(1, 10)]
        score_params = []  # will be list of tuples
        for cp in corruption_percentages:
            for mp in mingle_percentages:
                score_params.append(tuple([cp, mp]))
                # default_config['generate_data'] = GenerateDataConfig(
                #     dataset_size=1000,
                #     perc_exact_match=0.1,
                #     perc_non_match=0.0,
                #     fuzzy_match_corruption_perc=cp,
                #     fuzzy_match_mingle_perc=mp,
                #     use_pareto_dist_for_degrees=True,
                #     raw_file_path="eval_data/raw-1000.parquet",
                # )
                # run_config = RunConfig(default_config)
                # evaluate.execute_in_process(run_config=run_config)

        # collect scores to visualize
        metric_to_plot = "average_precision"
        score_files = [
            os.path.join(root, name)
            for root, dirs, files in os.walk("mlruns")
            for name in files if name == metric_to_plot
        ]
        score_data = []  # will be list of tuples
        for f in score_files:
            with open(f, "r") as file:
                first_line = file.readline()
                score_data.append(tuple(first_line.split(" ")[:2]))
        score_data = pd.DataFrame(score_data, columns=["time", metric_to_plot]).sort_values("time")
        score_data['corruption_perc'] = [t[0] for t in score_params]
        score_data['mingle_perc'] = [t[1] for t in score_params]
        score_data.sort_values('time', ascending=False, inplace=True)

        # plot results
        fig = px.scatter_3d(score_data, x='corruption_perc', y='mingle_perc', z=metric_to_plot)
        fig.update_traces(marker={'size': 5})
        fig.show()

    else:
        # do a single run
        run_config = RunConfig(default_config)
        evaluate.execute_in_process(run_config=run_config)
