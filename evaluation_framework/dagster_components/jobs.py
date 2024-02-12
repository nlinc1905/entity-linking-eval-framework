from dagster import job

from dagster_components.assets import generate_data
from dagster_components.ops import (
    engineer_features, train_model, get_model_predictions, score_predictions,
    log_mlflow_metrics, save_data_for_dashboard,
)


@job
def evaluate():
    generated_data = generate_data()
    train_x, train_y, test_x, test_y = engineer_features(generated_data=generated_data)
    model = train_model(train_x=train_x, train_y=train_y)
    preds, pred_probs = get_model_predictions(model=model, test_x=test_x)
    scores, conf_matrix, test_x, rank_table_df = score_predictions(
        preds=preds,
        pred_probs=pred_probs,
        test_x=test_x,
        test_y=test_y,
    )
    log_mlflow_metrics(model=model, scores=scores)
    save_data_for_dashboard(
        scores=scores,
        conf_matrix=conf_matrix,
        test_x=test_x,
        test_y=test_y,
        preds=preds,
        rank_table_df=rank_table_df,
    )
