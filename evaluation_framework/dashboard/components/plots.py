import typing as t
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def min_max_normalizer(arr: np.ndarray):
    """Normalize values in the array to a range of 0-1"""
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def categories_to_continuous_colorscale(category_to_color_map: dict) -> t.List[list]:
    """
    Converts a list of categories into evenly spaced values on a continuous scale,
    and maps the categories to the provided colors.
    """
    val_scale = np.linspace(0, len(category_to_color_map), len(category_to_color_map))
    val_scale = min_max_normalizer(arr=val_scale)
    resulting_colorscale = [[val_scale[i], c] for i, c in enumerate(category_to_color_map.values())]
    return resulting_colorscale


def get_label_positions(category_to_color_map: dict, category: str) -> float:
    """Gets the x coordinate for a category label."""
    all_pos = np.linspace(0, 1, len(category_to_color_map) + 2)
    all_pos = all_pos[1: -1]
    pos = all_pos[list(category_to_color_map).index(category)]
    return pos


def reduce_data_if_needed(df: pd.DataFrame, comparison_col: str, true_negative_limit: int = 1_000) -> pd.DataFrame:
    """
    If the number of rows is sufficiently large, reduce the data to be plotted by sampling
    from the true negatives, since they will always be the largest category.  This function's
    only purpose is to speed up plot creation in the browser.
    """
    if len(df[df[comparison_col] == 'TN']) > true_negative_limit:
        df = pd.concat(
            objs=[
                df[df[comparison_col] != 'TN'],
                df[df[comparison_col] == 'TN'].sample(n=true_negative_limit, replace=False, random_state=14)
            ],
            ignore_index=True
        )
    return df


def make_paracoords(df: pd.DataFrame, comparison_col: str, category_to_color_map: dict) -> go.Figure:
    """Creates a parallel coordinates figure."""

    # make an integer column from the comparison_col that plotly can use for color mapping
    category_to_int_map = {c: i for i, c in enumerate(list(category_to_color_map))}
    df[f'{comparison_col}_int'] = df[comparison_col].map(category_to_int_map)

    # if the number of rows is sufficiently large, reduce by sampling from true negatives,
    # since they will always be the largest category
    df = reduce_data_if_needed(df, comparison_col)

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=df[f'{comparison_col}_int'],
                colorscale=categories_to_continuous_colorscale(category_to_color_map=category_to_color_map)
            ),
            dimensions=list([
                dict(label=col, values=df[col]) for col in df.columns if
                col not in [comparison_col, f'{comparison_col}_int']
            ]),
        )
    )

    fig.update_layout(
        plot_bgcolor='#FFFFF0',
        paper_bgcolor='#FFFFF0',
        annotations=[
            dict(
                xref='paper',
                yref='paper',
                x=get_label_positions(category_to_color_map, category),
                y=-0.085,
                xanchor='center',
                yanchor='top',
                text=f"<b>{category}</b>",
                font=dict(
                    size=14,
                    color=color
                ),
                showarrow=False
            )
            for i, (category, color) in enumerate(category_to_color_map.items())
        ]
    )

    return fig


def make_boxplot(df: pd.DataFrame, col_to_plot: str, comparison_col: str, category_to_color_map: dict) -> go.Figure:
    """Creates a boxplot figure for the given column, grouped by the given comparison column."""
    fig = go.Figure()

    # if the number of rows is sufficiently large, reduce by sampling from true negatives,
    # since they will always be the largest category
    df = reduce_data_if_needed(df, comparison_col)

    for category, color in category_to_color_map.items():
        df_sub = df[df[comparison_col] == category]
        fig.add_trace(
            go.Box(y=df_sub[col_to_plot], name=category, line=dict(color=color), boxpoints="all")
        )

    fig.update_layout(
        title=f'Distribution of {col_to_plot} by {comparison_col}'
    )

    return fig


def make_curve(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str) -> go.Figure:
    """Creates ROC and PR curves"""
    fig = px.area(
        x=x,
        y=y,
        title=title,
        labels=dict(x=xlabel, y=ylabel),
        height=300,
    )
    if "ROC" in title:
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    else:
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0.5, y1=0.5)
    return fig
