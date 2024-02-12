import polars as pl
import json
from dagster import asset, Config, Out

from dagster_components.entity_data_generator.generate_data import (
    generate_and_corrupt, combine_datasets_and_save_to_parquet,
    get_graph, compare_graph_attributes, gini,
)


class GenerateDataConfig(Config):
    dataset_size: int
    perc_exact_match: float
    perc_non_match: float
    fuzzy_match_corruption_perc: float
    fuzzy_match_mingle_perc: float
    use_pareto_dist_for_degrees: bool
    raw_file_path: str


@asset
def generate_data(config: GenerateDataConfig) -> pl.DataFrame:

    # the second half of the dataset will be derived from the first half
    half_dataset_size = int(config.dataset_size / 2)

    a, b, g = generate_and_corrupt(
        dataset_size=half_dataset_size,
        perc_exact_match=config.perc_exact_match,
        perc_non_match=config.perc_non_match,
        fuzzy_match_corruption_perc=config.fuzzy_match_corruption_perc,
        fuzzy_match_mingle_perc=config.fuzzy_match_mingle_perc,
        use_pareto_dist_for_degrees=config.use_pareto_dist_for_degrees,
    )

    # combine the datasets and save the result
    generated_graph_data = combine_datasets_and_save_to_parquet(ds=a + b, out_path=config.raw_file_path)

    # save node data for the dashboard
    node_data_for_dashboard = dict(zip(
        generated_graph_data.get_column('index'),
        generated_graph_data.with_columns(
            pl.col(pl.Date).dt.strftime('%Y-%m-%d')
        ).drop(['index', 'id']).to_dicts()
    ))
    with open("dashboard/data/node_data.json", "w") as f:
        json.dump(node_data_for_dashboard, f)

    if g is not None:
        # compare network attributes between the generated data and a randomly generated one
        nodelist = generated_graph_data.get_column('index').to_list()
        edgelist = generated_graph_data.group_by('id').agg(pl.col('index')).get_column('index').to_list()
        g_gen = get_graph(nodes=nodelist, edges=edgelist, include_singletons=False)
        compare_graph_attributes(g1=g, g2=g_gen, plot=True)

    return generated_graph_data
