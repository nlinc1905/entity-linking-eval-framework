import os
import recordlinkage as rl
from recordlinkage.datasets import load_krebsregister
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score, average_precision_score,
    ndcg_score, label_ranking_average_precision_score
)
import numpy as np
import polars as pl
import pandas as pd
import json
import mlflow

from entity_data_generator.generate_data import (
    generate_and_corrupt, combine_datasets_and_save_to_parquet,
    get_graph, compare_graph_attributes,
)
from feature_engineering.make_features import make_features
from metrics.classifier_metrics import (
    get_true_positives, get_false_positives, get_true_negatives, get_false_negatives
)
from metrics.ranker_metrics import map_at_k


DATASET_SIZE = 500
MATCHING_SCHEME = {
    'exact': 0.2,
    'fuzzy': 0.8,
    'non': 0.0,
}
FUZZY_MATCH_CORRUPTION_AMOUNT = 0.75  # this percentage of columns will be corrupted
USE_PARETO_DIST = True  # whether to sample degrees (nbr of corruptions) from a Pareto distribution
# filename will have dataset size, match perc, corruption perc
OUTPUT_FILE_PATH = (
    f"eval_data/raw-{int(DATASET_SIZE*2)}-{str(1 - MATCHING_SCHEME['non']).replace('.', '_')}"
    f"-{str(FUZZY_MATCH_CORRUPTION_AMOUNT).replace('.', '_')}.parquet"
)

INPUT_FILE_PATH = "eval_data/raw-1000-1_0-0_5.parquet"
TRAIN_TEST_RATIO = 0.7
OUTPUT_FILE_PATH_TRAIN = INPUT_FILE_PATH.replace("raw", "train").replace("parquet", "csv")
OUTPUT_FILE_PATH_TEST = INPUT_FILE_PATH.replace("raw", "test").replace("parquet", "csv")

INPUT_DATA_PATH_TRAIN = "eval_data/train-1000-1_0-0_5.csv"
INPUT_DATA_PATH_TEST = "eval_data/test-1000-1_0-0_5.csv"
TRACK_MLFLOW_EXPERIMENT = True
DASHBOARD_DATA_PATH = "dashboard/data/"
CLASSIFIER_RESULT_COL = "model_score_category"


# TODO: wrap this section in Dagster generate_data op
a, b, g = generate_and_corrupt(
    dataset_size=DATASET_SIZE,
    perc_exact_match=MATCHING_SCHEME['exact'],
    perc_non_match=MATCHING_SCHEME['non'],
    fuzzy_match_corruption_perc=FUZZY_MATCH_CORRUPTION_AMOUNT,
    use_pareto_dist_for_degrees=USE_PARETO_DIST,
)

# combine the datasets and save the result
generated_graph_data = combine_datasets_and_save_to_parquet(ds=a + b, out_path=OUTPUT_FILE_PATH)

if g is not None:
    # compare network attributes between the generated data and a randomly generated one
    nodelist = generated_graph_data.get_column('index').to_list()
    edgelist = generated_graph_data.group_by('id').agg(pl.col('index')).get_column('index').to_list()
    g_gen = get_graph(nodes=nodelist, edges=edgelist, include_singletons=False)
    compare_graph_attributes(g1=g, g2=g_gen, plot=True)
# TODO: end wrapper for Dagster generate_data op

# TODO: wrap this in a Dagster make_features op
make_features(
    df=INPUT_FILE_PATH,
    train_test_ratio=TRAIN_TEST_RATIO,
    output_file_path_train=OUTPUT_FILE_PATH_TRAIN,
    output_file_path_test=OUTPUT_FILE_PATH_TEST,
)
# TODO: end wrapper for Dagster make_features op


if os.path.isfile(INPUT_DATA_PATH_TRAIN) and os.path.isfile(INPUT_DATA_PATH_TEST):
    ID_COLNAME_PREFIX = "index_"
    train = pd.read_csv(INPUT_DATA_PATH_TRAIN).set_index(['index_1', 'index_2'])
    test = pd.read_csv(INPUT_DATA_PATH_TEST).set_index(['index_1', 'index_2'])
    # recordlinkage expects labels to be a pandas MultiIndex, and features to be a dataframe
    train_x = train.drop('true_label', axis=1)
    train_y = train.loc[train['true_label'] == 1, 'true_label'].index
    test_x = test.drop('true_label', axis=1)
    test_y = test.loc[test['true_label'] == 1, 'true_label'].index
else:
    ID_COLNAME_PREFIX = "id"
    # TODO: assumes all features have been created already
    # https://archive.ics.uci.edu/dataset/210/record+linkage+comparison+patterns
    x, true_links = load_krebsregister(
        block=[*range(1, 11)],  # which blocks (out of 10) to load
        missing_values=0,  # which value to fill NaNs?
        shuffle=True,
    )

    # train/test split
    train_test_split_ratio = 0.8
    split_idx = int(train_test_split_ratio * len(x))
    train_x, test_x = x[:split_idx], x[split_idx:]
    train_y, test_y = train_x.index.intersection(true_links), test_x.index.intersection(true_links)

# train model - these are all wrappers of sklearn
models = {
    "lr": rl.LogisticRegressionClassifier(),
    "nb": rl.NaiveBayesClassifier(binarize=0.3, alpha=1e-4),
    "svm": rl.SVMClassifier(),  # uses linear SVC
    "km": rl.KMeansClassifier(),  # uses k=2
    "em": rl.ECMClassifier(binarize=0.8, max_iter=100, atol=1e-4),
}
model = models['lr']
model.fit(train_x, train_y)

# make preds
preds = model.predict(test_x)
pred_probs = model.prob(test_x)

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
test_x[CLASSIFIER_RESULT_COL] = 'UNK'
test_x.loc[tps, CLASSIFIER_RESULT_COL] = 'TP'
test_x.loc[fps, CLASSIFIER_RESULT_COL] = 'FP'
test_x.loc[tns, CLASSIFIER_RESULT_COL] = 'TN'
test_x.loc[fns, CLASSIFIER_RESULT_COL] = 'FN'

# sort the matches by descending predicted probabilities for each ID
pred_probs = pred_probs.reset_index(drop=False).rename(columns={0: 'prob'})
pred_probs.sort_values([f'{ID_COLNAME_PREFIX}1', 'prob'], ascending=[True, False], inplace=True)

# add the binary ground truth label to the predicted probability dataframe
test_y_dict = {k: 1 for k in test_y}
pred_probs['key'] = list(zip(pred_probs[f'{ID_COLNAME_PREFIX}1'], pred_probs[f'{ID_COLNAME_PREFIX}2']))
pred_probs['actual'] = pred_probs['key'].map(test_y_dict).fillna(0).astype(int)

# convert predicted probabilities into a ranking dataframe to evaluation information retrieval metrics
link_to_score_category_map = dict(zip(test_x.index, test_x[CLASSIFIER_RESULT_COL]))
rank_table_df = pred_probs.copy()
rank_table_df['predicted_rank'] = rank_table_df.groupby(f'{ID_COLNAME_PREFIX}1').transform('cumcount') + 1
rank_table_df[CLASSIFIER_RESULT_COL] = rank_table_df['key'].map(link_to_score_category_map)
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

conf_matrix = confusion_matrix(y_true=test_labels, y_pred=test_preds)
scores = {
    # classification metrics
    'accuracy': accuracy_score(y_true=test_labels, y_pred=test_preds),
    'f1': f1_score(y_true=test_labels, y_pred=test_preds),
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
}
scores = {k: round(v, 4) for k, v in scores.items()}
print(conf_matrix, "\n", scores)

# log to MLFlow
if TRACK_MLFLOW_EXPERIMENT:
    with mlflow.start_run() as run:
        mlflow.log_params(model.params)
        mlflow.log_metrics(scores)

# save data for dashboard
test_x.to_parquet(f"{DASHBOARD_DATA_PATH}scored_data.parquet", index=True)
test_y.to_frame().to_parquet(f"{DASHBOARD_DATA_PATH}labels.parquet", index=False)
preds.to_frame().to_parquet(f"{DASHBOARD_DATA_PATH}preds.parquet", index=False)
rank_table_df.to_parquet(f"{DASHBOARD_DATA_PATH}ranks.parquet", index=False)
with open(f"{DASHBOARD_DATA_PATH}scores.json", 'w') as f:
    json.dump(scores, f)
np.save(f"{DASHBOARD_DATA_PATH}conf_matrix.npy", conf_matrix)
