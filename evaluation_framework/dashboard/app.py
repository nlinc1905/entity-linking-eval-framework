from dash import Dash, Input, Output, callback
import dash_bootstrap_components as dbc
import json
import polars as pl
import numpy as np

from graph.primitives.entities import Node, Link
from graph.manipulators import (
    create_links_dict, remove_nodes_that_are_not_in_links,
    filter_to_false_positives, filter_to_false_negatives, filter_to_errors,
)
from components.stylesheet import style_confusion_matrix, style_graph, style_rank_data_table
from components.plots import make_paracoords, make_boxplot
from components.layout import make_layout


DASHBOARD_DATA_PATH = "data/"
ID_COLNAME_PREFIX = "index_"
CLASSIFIER_RESULT_COL = "model_score_category"
CATEGORY_TO_COLOR_MAP = {
    'TP': '#228B22',  # '#90EE90',
    'FP': '#B22222',  # '#F08080',
    'TN': '#1E90FF',  # '#ADD8E6',
    'FN': '#DAA520',  # '#FFA07A',
}


# ingest the source data for this dashboard, which is the output of the evaluation_pipeline.py
# this data should be for the test set only
test_x = pl.read_parquet(f"{DASHBOARD_DATA_PATH}scored_data.parquet")
test_y = pl.read_parquet(f"{DASHBOARD_DATA_PATH}labels.parquet")
preds = pl.read_parquet(f"{DASHBOARD_DATA_PATH}preds.parquet")
rank_table_df = pl.read_parquet(f"{DASHBOARD_DATA_PATH}ranks.parquet")
with open(f"{DASHBOARD_DATA_PATH}scores.json", "r") as f:
    scores = json.load(f)
conf_matrix = np.load(f"{DASHBOARD_DATA_PATH}conf_matrix.npy")

# convert polars dataframes to pandas MultiIndex where needed
tps = test_x.filter(pl.col(CLASSIFIER_RESULT_COL) == 'TP').select([CLASSIFIER_RESULT_COL, f'{ID_COLNAME_PREFIX}1', f'{ID_COLNAME_PREFIX}2']).to_pandas().set_index([f'{ID_COLNAME_PREFIX}1', f'{ID_COLNAME_PREFIX}2']).index
fps = test_x.filter(pl.col(CLASSIFIER_RESULT_COL) == 'FP').select([CLASSIFIER_RESULT_COL, f'{ID_COLNAME_PREFIX}1', f'{ID_COLNAME_PREFIX}2']).to_pandas().set_index([f'{ID_COLNAME_PREFIX}1', f'{ID_COLNAME_PREFIX}2']).index
fns = test_x.filter(pl.col(CLASSIFIER_RESULT_COL) == 'FN').select([CLASSIFIER_RESULT_COL, f'{ID_COLNAME_PREFIX}1', f'{ID_COLNAME_PREFIX}2']).to_pandas().set_index([f'{ID_COLNAME_PREFIX}1', f'{ID_COLNAME_PREFIX}2']).index
test_x = test_x.to_pandas().set_index([f'{ID_COLNAME_PREFIX}1', f'{ID_COLNAME_PREFIX}2'])
test_y = test_y.to_pandas().set_index([f'{ID_COLNAME_PREFIX}1', f'{ID_COLNAME_PREFIX}2']).index
preds = preds.to_pandas().set_index([f'{ID_COLNAME_PREFIX}1', f'{ID_COLNAME_PREFIX}2']).index
rank_table_df = rank_table_df.to_pandas()

# create node and links objects for the graph
nodes = {rr: Node(_id=rr, _label=str(rr), _classes=None, some_other_node_feature="test") for r in set(test_x.index) for rr in r}  # {id: N}
links = create_links_dict(idx=test_y, nodes_dict=nodes)  # {(node_id1, node_id2): L}

# create predicted links for the graph
pred_links = {r: Link(source=nodes[r[0]], target=nodes[r[1]], weight=1, classes="tp") for r in tps}
pred_links.update({r: Link(source=nodes[r[0]], target=nodes[r[1]], weight=1, classes="fp") for r in fps})
pred_links.update({r: Link(source=nodes[r[0]], target=nodes[r[1]], weight=1, classes="fn") for r in fns})

# remove nodes that would have no links in the graph
nodes = remove_nodes_that_are_not_in_links(nodes_dict=nodes, links_dict=pred_links)

# filter to errors
# elements = build_graph_elements(nodes_dict=nodes, links_dict=pred_links)
# elements = filter_to_false_positives(nodes_dict=nodes, fp_set=set(fps), fp_links_only=False, pred_links_dict=pred_links)
# elements = filter_to_false_negatives(nodes_dict=nodes, fn_set=set(fns), fn_links_only=False, pred_links_dict=pred_links)
elements = filter_to_errors(nodes_dict=nodes, fp_set=set(fps), fn_set=set(fns), pred_links_dict=pred_links)
ids_w_errors = rank_table_df.loc[rank_table_df[CLASSIFIER_RESULT_COL].isin(['FP', 'FN']), f'{ID_COLNAME_PREFIX}1'].unique()
rank_table_df = rank_table_df[rank_table_df[f'{ID_COLNAME_PREFIX}1'].isin(ids_w_errors)]

# make plots to inspect feature distributions by error category
paracoords_fig = make_paracoords(df=test_x, comparison_col=CLASSIFIER_RESULT_COL, category_to_color_map=CATEGORY_TO_COLOR_MAP)
boxplots = [
    make_boxplot(
        df=test_x,
        col_to_plot=col,
        comparison_col=CLASSIFIER_RESULT_COL,
        category_to_color_map=CATEGORY_TO_COLOR_MAP
    )
    for col in test_x.select_dtypes(include=[np.float32, np.float64])
]


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


# prepare CSS
conf_matrix_styles = style_confusion_matrix(CATEGORY_TO_COLOR_MAP)
graph_styles = style_graph(CATEGORY_TO_COLOR_MAP)
rank_table_styles = style_rank_data_table(CATEGORY_TO_COLOR_MAP)

# make the HTML layout
app.layout = make_layout(
    conf_matrix=conf_matrix,
    conf_matrix_styles=conf_matrix_styles,
    scores=scores,
    graph_elements=elements,
    graph_styles=graph_styles,
    rank_table_df=rank_table_df,
    rank_table_styles=rank_table_styles,
    paracoords_fig=paracoords_fig,
    boxplot_figs=boxplots,
)


@callback(
    Output('node-data', 'children'),
    Input('cytoscape-layout-9', 'tapNodeData'),
)
def display_selected_node_data(data):
    return json.dumps(data, indent=2)


app.run(debug=False)
