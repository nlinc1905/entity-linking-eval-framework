from dagster import op, Config, Out
import pandas as pd
import polars as pl
import recordlinkage as rl
import typing as t
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, average_precision_score,
    ndcg_score
)
import numpy as np
import networkx as nx
import mlflow
import json

from dagster_components.feature_engineering.make_features import make_features
from dagster_components.entity_data_generator.generate_data import (
    generate_and_corrupt, combine_datasets_and_save_to_parquet,
    get_graph, compare_graph_attributes, gini,
)
from dagster_components.metrics.classifier_metrics import (
    get_true_positives, get_false_positives, get_true_negatives, get_false_negatives
)
from dagster_components.metrics.ranker_metrics import map_at_k


class EngineerFeaturesConfig(Config):
    raw_file_path: str
    train_test_ratio: float
    train_file_path: str
    test_file_path: str


class TrainModelConfig(Config):
    model_key: str


class ScorePredictionsConfig(Config):
    classifier_result_col: str
    id_colname_prefix: str


class LogMlflowMetricsConfig(Config):
    track_mlflow_experiment: bool


class SaveDataConfig(Config):
    dashboard_data_path: str


@op(out={'train_x': Out(), 'train_y': Out(), 'test_x': Out(), 'test_y': Out()})
def engineer_features(
    generated_data: pl.DataFrame,
    config: EngineerFeaturesConfig
) -> t.Tuple[pd.DataFrame, pd.MultiIndex, pd.DataFrame, pd.MultiIndex]:
    train, test = make_features(
        df=generated_data,
        train_test_ratio=config.train_test_ratio,
        output_file_path_train=config.train_file_path,
        output_file_path_test=config.test_file_path,
    )

    # recordlinkage expects labels to be a pandas MultiIndex, and features to be a dataframe
    train_x = train.drop('true_label', axis=1)
    train_y = train.loc[train['true_label'] == 1, 'true_label'].index
    test_x = test.drop('true_label', axis=1)
    test_y = test.loc[test['true_label'] == 1, 'true_label'].index

    return train_x, train_y, test_x, test_y


@op
def train_model(config: TrainModelConfig, train_x: pd.DataFrame, train_y: pd.MultiIndex):
    # TODO: make this better
    models = {
        "lr": rl.LogisticRegressionClassifier(),
        "nb": rl.NaiveBayesClassifier(binarize=0.3, alpha=1e-4),
        "svm": rl.SVMClassifier(),  # uses linear SVC
        "km": rl.KMeansClassifier(),  # uses k=2
        "em": rl.ECMClassifier(binarize=0.8, max_iter=100, atol=1e-4),
    }
    model = models[config.model_key]
    model.fit(train_x, train_y)

    return model


@op(out={'preds': Out(), 'pred_probs': Out()})
def get_model_predictions(model, test_x: pd.DataFrame) -> t.Tuple[pd.MultiIndex, pd.Series]:
    preds = model.predict(test_x)
    pred_probs = model.prob(test_x)
    return preds, pred_probs


@op(out={'scores': Out(), 'conf_matrix': Out(), 'test_x': Out(), 'rank_table_df': Out()})
def score_predictions(
    config: ScorePredictionsConfig,
    preds: pd.MultiIndex, 
    pred_probs: pd.Series,
    test_x: pd.DataFrame, 
    test_y: pd.MultiIndex,
) -> t.Tuple[dict, np.ndarray, pd.DataFrame, pd.DataFrame]:
    # preds and test_y are both MultiIndexes and must be converted to labels (1, 0)
    # get true labels and predicted labels
    test_preds = test_x.index.isin(preds).astype(int)
    test_labels = test_x.index.isin(test_y).astype(int)

    # determine which rows are TP, FP, TN, FN
    tps = get_true_positives(test_y, preds)
    fps = get_false_positives(test_y, preds)
    tns = get_true_negatives(test_x.index, test_y, preds)
    fns = get_false_negatives(test_y, preds)

    # add a model score category column
    test_x[config.classifier_result_col] = 'UNK'
    test_x.loc[tps, config.classifier_result_col] = 'TP'
    test_x.loc[fps, config.classifier_result_col] = 'FP'
    test_x.loc[tns, config.classifier_result_col] = 'TN'
    test_x.loc[fns, config.classifier_result_col] = 'FN'

    # sort the matches by descending predicted probabilities for each ID
    pred_probs = pred_probs.reset_index(drop=False).rename(columns={0: 'prob'})
    pred_probs.sort_values([f'{config.id_colname_prefix}1', 'prob'], ascending=[True, False], inplace=True)

    # add the binary ground truth label to the predicted probability dataframe
    test_y_dict = {k: 1 for k in test_y}
    pred_probs['key'] = list(zip(pred_probs[f'{config.id_colname_prefix}1'], pred_probs[f'{config.id_colname_prefix}2']))
    pred_probs['actual'] = pred_probs['key'].map(test_y_dict).fillna(0).astype(int)

    # convert predicted probabilities into a ranking dataframe to evaluation information retrieval metrics
    link_to_score_category_map = dict(zip(test_x.index, test_x[config.classifier_result_col]))
    rank_table_df = pred_probs.copy()
    rank_table_df['predicted_rank'] = rank_table_df.groupby(f'{config.id_colname_prefix}1').transform('cumcount') + 1
    rank_table_df[config.classifier_result_col] = rank_table_df['key'].map(link_to_score_category_map)
    rank_table_df['prob'] = rank_table_df['prob'].apply(lambda x: round(x, 4))
    rank_table_df.rename(columns={"prob": "link_probability_score", "actual": "true_link"}, inplace=True)
    rank_table_df.drop('key', axis=1, inplace=True)

    # for IR, want to score everything that actually had links or the model predicted had links
    ranks_to_score = rank_table_df[
        (rank_table_df['true_link'] == 1)
        | (rank_table_df['link_probability_score'] >= 0.5)
        ].reset_index(drop=True)

    # create list of lists to feed to MAP scoring function
    # lists should be [[doc_id1, doc_id2], [], ...] where an empty list is when the doc was not a true link or not predicted
    y_true, y_pred = [], []
    for row in ranks_to_score.iterrows():
        row_list_true, row_list_pred = [], []
        # row is a tuple of (index, data), where data is indexed with the following order:
        # [id1, id2, link_probability_score, true_link, predicted_rank, model_score_category]
        if row[1][3] == 1:
            row_list_true.append(row[1][1])
        if row[1][2] >= 0.5:
            row_list_pred.append(row[1][1])
        y_true.append(row_list_true)
        y_pred.append(row_list_pred)

    # calculate the connected components Gini to measure how well-formed the predicted graph is
    g_pred = get_graph(nodes=[], edges=preds.tolist(), include_singletons=False)
    g_pred_cc = np.array([len(c) for c in sorted(nx.connected_components(g_pred), key=len, reverse=True)])

    conf_matrix = confusion_matrix(y_true=test_labels, y_pred=test_preds)
    scores = {
        # classification metrics
        'accuracy': accuracy_score(y_true=test_labels, y_pred=test_preds),
        'f1': f1_score(y_true=test_labels, y_pred=test_preds),
        'roc_auc': roc_auc_score(
            y_true=rank_table_df['true_link'].values,
            y_score=rank_table_df['link_probability_score']
        ),
        'precision': precision_score(y_true=test_labels, y_pred=test_preds),
        'recall': recall_score(y_true=test_labels, y_pred=test_preds),
        'average_precision': average_precision_score(
            y_true=rank_table_df['true_link'].values,
            y_score=rank_table_df['link_probability_score']
        ),
        # ranking metrics
        'map_at_k': map_at_k(y_true=y_true, y_pred=y_pred, k=40),
        'ndcg': ndcg_score(
            y_true=[ranks_to_score['true_link'].tolist()],
            y_score=[ranks_to_score['link_probability_score'].tolist()]
        ),
        # graph metrics
        'cc_gini': gini(g_pred_cc),
    }
    scores = {k: round(v, 4) for k, v in scores.items()}

    return scores, conf_matrix, test_x, rank_table_df


@op
def log_mlflow_metrics(config: LogMlflowMetricsConfig, model, scores: dict) -> None:
    if config.track_mlflow_experiment:
        with mlflow.start_run() as run:
            mlflow.log_params(model.params)
            mlflow.log_metrics(scores)


@op
def save_data_for_dashboard(
    config: SaveDataConfig,
    scores: dict,
    conf_matrix: np.ndarray,
    test_x: pd.DataFrame,
    test_y: pd.MultiIndex,
    preds: pd.MultiIndex,
    rank_table_df: pd.DataFrame
) -> None:
    test_x.to_parquet(f"{config.dashboard_data_path}scored_data.parquet", index=True)
    test_y.to_frame().to_parquet(f"{config.dashboard_data_path}labels.parquet", index=False)
    preds.to_frame().to_parquet(f"{config.dashboard_data_path}preds.parquet", index=False)
    rank_table_df.to_parquet(f"{config.dashboard_data_path}ranks.parquet", index=False)
    with open(f"{config.dashboard_data_path}scores.json", 'w') as f:
        json.dump(scores, f)
    np.save(f"{config.dashboard_data_path}conf_matrix.npy", conf_matrix)
