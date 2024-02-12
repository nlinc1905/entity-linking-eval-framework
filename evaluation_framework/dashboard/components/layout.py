from dash import html, dash_table, dcc
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def make_layout(
        conf_matrix: np.ndarray,
        conf_matrix_styles: dict,
        scores: dict,
        roc_curve: go.Figure,
        pr_curve: go.Figure,
        graph_elements: list,
        graph_styles: list,
        rank_table_df: pd.DataFrame,
        rank_table_styles: list,
        paracoords_fig: go.Figure,
        boxplot_figs: list,
):

    classification_scores_layout = html.Table(
        children=[
            html.Th(["Classification Metrics"]),
            html.Tr([
                html.Td(['Accuracy']),
                html.Td(id='accuracy-score', children=[scores['accuracy']], style={'padding-left': '8px'}),
            ]),
            html.Tr([
                html.Td(['F1 Score']),
                html.Td(id='f1-score', children=[scores['f1']], style={'padding-left': '8px'}),
            ]),
            html.Tr([
                html.Td(['Precision']),
                html.Td(id='precision-score', children=[scores['precision']], style={'padding-left': '8px'}),
            ]),
            html.Tr([
                html.Td(['Recall (Sensitivity)']),
                html.Td(id='recall-score', children=[scores['recall']], style={'padding-left': '8px'}),
            ]),
            html.Tr([
                html.Td(['PR AUC (Average Precision)']),
                html.Td(id='pr-score', children=[scores['average_precision']], style={'padding-left': '8px'}),
            ]),
        ],
        style={
            'border': '2px solid gray',
            'border-collapse': 'separate',
            'border-radius': '10px',
            'padding': '8px',
        },
    )

    confusion_matrix_layout = html.Table(
        id='confusion-matrix',
        children=[
            html.Tr(
                children=[
                    html.Td(
                        id='tn',
                        children=[f"{conf_matrix[0][0]:,} True Negatives"],
                        style=conf_matrix_styles['tn'],
                    ),
                    html.Td(
                        id='fp',
                        children=[f"{conf_matrix[0][1]:,} False Positives"],
                        style=conf_matrix_styles['fp'],
                    ),
                ],
                style={'border-bottom': '2px solid gray'}
            ),
            html.Tr([
                html.Td(
                    id='fn',
                    children=[f"{conf_matrix[1][0]:,} False Negatives"],
                    style=conf_matrix_styles['fn'],
                ),
                html.Td(
                    id='tp',
                    children=[f"{conf_matrix[1][1]:,} True Positives"],
                    style=conf_matrix_styles['tp'],
                ),
            ]),
        ],
    )

    ranking_scores_layout = html.Table(
        children=[
            html.Th(["Ranking Metrics"]),
            html.Tr([
                html.Td(['Mean Average Precision at k = 40']),
                html.Td(id='mapk-score', children=[scores['map_at_k']], style={'padding-left': '8px'}),
            ]),
            html.Tr([
                html.Td(['Normalized Discounted Cumulative Gain (NDCG)']),
                html.Td(id='ndcg-score', children=[scores['ndcg']], style={'padding-left': '8px'}),
            ]),
        ],
        style={
            'border': '2px solid gray',
            'border-collapse': 'separate',
            'border-radius': '10px',
            'padding': '8px',
        },
    )

    graph_layout = cyto.Cytoscape(
        id='cytoscape-layout-9',
        elements=graph_elements,
        style={'width': '100%', 'height': '450px', 'background-color': '#FFFFF0'},
        layout={
            'name': 'cose',  # cose or circle work best
            'mouseoverEdgeData': {'weight': 'data(weight)'}
        },
        stylesheet=graph_styles,
    )

    graph_data_layout = html.Pre(
        id='node-data',
        style={'border': 'thin lightgrey solid', 'overflowX': 'scroll'},
    )

    rank_table_layout = dash_table.DataTable(
        id='rank-table',
        data=rank_table_df.to_dict('records'),
        columns=[{"name": i.replace("_", " ").title(), "id": i} for i in rank_table_df.columns],
        style_data_conditional=rank_table_styles,
        page_size=50,
        filter_action='native',
    )

    layout = dbc.Container([

        html.Br(),

        dbc.Row(
            id='scores',
            children=[
                dbc.Col(
                    children=[classification_scores_layout],
                    width=4,
                ),
                dbc.Col(
                    children=[confusion_matrix_layout],
                    width=4,
                ),
                dbc.Col(
                    children=[ranking_scores_layout],
                    width=4,
                )
            ],
        ),

        dbc.Row(
            id="curves",
            children=[
                dbc.Col(
                    children=[dcc.Graph(figure=roc_curve)],
                    width=6,
                ),
                dbc.Col(
                    children=[dcc.Graph(figure=pr_curve)],
                    width=6,
                ),
            ],
        ),

        html.Br(),
        html.Hr(),

        html.H5("Graph of Entities with Incorrectly Predicted Links"),
        html.P("(true negatives are not shown because they are not links)"),
        dbc.Row(
            id='graph',
            children=[
                dbc.Col(
                    children=[graph_layout],
                    width=9,
                ),
                dbc.Col(
                    children=[graph_data_layout],
                    width=3,
                ),
            ],
        ),

        html.Br(),
        html.Hr(),

        html.H5("Ranked Matches per Entity for Entities with Incorrectly Predicted Links"),
        html.P("(the order of probable matches that the user would see)"),
        dbc.Row(
            id='rankings',
            children=[
                dbc.Col(
                    children=[rank_table_layout],
                    width=12,
                ),
            ],
        ),

        html.Br(),
        html.Hr(),

        html.H5("Plots to Compare the Entity-Pair Feature Distributions by Classifier Score Category"),
        dcc.Graph(figure=paracoords_fig),
        html.Div([dcc.Graph(figure=f) for f in boxplot_figs]),

    ])

    return layout
