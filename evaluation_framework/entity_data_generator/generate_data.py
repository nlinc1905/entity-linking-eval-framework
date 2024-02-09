import typing as t
import random
import datetime
import polars as pl
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import itertools

from entity_data_generator.entities.entity import Entity
from entity_data_generator.entities.get_random import random_name, random_date, random_email
from entity_data_generator.entities.mutators import StringMutations, DateMutations


def get_degree_distribution(
    alpha: float = 70.0,
    x_m: float = 1.0,
    nbr_samples: int = 100,
    max_degree: int = 10,
    use_degree_offset: bool = True,
    generate_random_graph: bool = False,
    verbose: bool = False,
) -> t.Tuple[np.ndarray, t.Any]:
    """
    Generates a Pareto shaped degree distribution, where the shape is controlled
    by parameters alpha and x_m.

    :param alpha: Controls the shape or 'inequality' of the distribution.  The larger alpha is,
        the more concentrated links will be for a smaller portion of nodes.  At the
        extremes, alpha = 0 means 1 node will have all the links, and alpha = inf means all
        nodes will have uniform links.
    :param x_m: This is the smallest possible value for the distribution.  This parameter
        does not control the smallest number of links a node can have - it controls the shape of the
        continuous Pareto distribution that the degrees are sampled from.  Unless there is
        good reason to change it, leave it at the default of 1.
    :param nbr_samples: How many nodes to generate
    :param max_degree: The most links any single entity/node can have.  This is dataset
        specific and should be changed to reflect the dataset you wish to model.
    :param use_degree_offset: If True (default), the resulting degree distribution will
        range from 0-max_degree inclusive.  If False, the resulting degree distribution
        will range from 1-(max_degree+1) inclusive.  Set to False if the degrees are being
        generated only for nodes/entities with links.  Leave as True if the degrees are
        being generated for all nodes/entities, including the ones that will have 0 links.
    :param generate_random_graph: If True, also returns a networkx MultiGraph with randomly
        generated links, following the degree distribution
    :param verbose: If True, information about the resulting distribution will be displayed
    """
    # the sum of degrees must be even
    condition = True
    while condition:
        samples = (np.random.default_rng().pareto(alpha, nbr_samples) + 1) * x_m

        # leverage matplotlib's binning to discretize the samples into bins
        # the bins will be the degrees, so by mapping samples from the distribution to the bins,
        # we map to degrees
        count, bins, _ = plt.hist(samples, bins=max_degree, density=True)
        degrees = np.digitize(samples, bins) - int(use_degree_offset)
        condition = degrees.sum() % 2 != 0

    if verbose:
        fit = alpha*x_m**alpha / bins**(alpha+1)
        plt.plot(bins, max(count)*fit/max(fit), linewidth=2, color='r')
        plt.title("Degree Distribution Will Look Like This")
        plt.show()
    plt.close()

    g = None
    if generate_random_graph:
        g = nx.configuration_model(deg_sequence=degrees, seed=14)
        g.remove_edges_from(nx.selfloop_edges(g))  # sometimes randomly generated graph have self-loops

    return degrees, g


def entity_generator(nbr_entities: int, id_offset: int = 0):
    """
    Generates Entity instances with fake identity data properties.

    :param nbr_entities: How many to generate
    :param id_offset: The entity IDs that are generated will begin with this value + 1
    """
    current = 0
    while current < nbr_entities:

        fname, lname = random_name()
        bdate = random_date().strftime('%Y-%m-%d')
        email = random_email()

        yield Entity(
            id_number=str(current + id_offset + 1),
            first_name=fname,
            last_name=lname,
            birth_date=bdate,
            email=email,
        )

        current += 1


def corrupt_entities(entities: t.List[Entity], perc_to_corrupt: float) -> t.List[Entity]:
    """
    Corrupts the provided entities by randomly changing perc_to_corrupt percentage of the entities'
    properties.  Possible corruptions are defined in entities.mutators.py.

    :param entities: The entities to corrupt
    :param perc_to_corrupt: Which percentage of attributes to corrupt
    """
    string_mutation_options = StringMutations
    date_mutation_options = DateMutations

    mutable_properties = [p for p in list(entities[0].__dict__) if p[0] != '_']
    nbr_properties_to_corrupt = int(perc_to_corrupt * len(mutable_properties))
    # print(f"Corrupting {nbr_properties_to_corrupt} properties ({perc_to_corrupt}%) per entity.")

    corrupted = []
    for ent in entities:

        # randomly choose which properties to mutate and how to mutate them
        properties_to_mutate = random.sample(mutable_properties, k=nbr_properties_to_corrupt)
        string_mutation_types = random.sample(
            list(string_mutation_options.__dict__['_value2member_map_']),
            k=nbr_properties_to_corrupt
        )
        date_mutation_types = random.sample(
            list(date_mutation_options.__dict__['_value2member_map_']),
            k=nbr_properties_to_corrupt
        )

        # the object is modified in place
        corrupted_ent = copy(ent)
        for i in range(len(properties_to_mutate)):
            if isinstance(corrupted_ent.__dict__[properties_to_mutate[i]], datetime.date):
                corrupted_ent.mutate_property(
                    property_name=properties_to_mutate[i],
                    date_mutation=date_mutation_types[i],
                )
            else:
                corrupted_ent.mutate_property(
                    property_name=properties_to_mutate[i],
                    string_mutation=string_mutation_types[i],
                )
        corrupted.append(corrupted_ent)

    return corrupted


def generate_and_corrupt(
    dataset_size: int, 
    perc_exact_match: float, 
    perc_non_match: float, 
    fuzzy_match_corruption_perc: float,
    use_pareto_dist_for_degrees: bool,
) -> t.Tuple[list, list, t.Union[nx.MultiGraph, None]]:
    """
    Generates fake identity records with various attributes/properties: dataset A.
    Spawns a second dataset from the first with corruptions to the data: dataset B.  
    The corruptions will allow entity linking algorithms to be tested and evaluated. 
    The records IDs will serve as ground truth links between the datasets.

    Note that percentage of fuzzy match records is no an argument, as it will be 
    inferred to be the remainder after accounting for exact and non-matches.

    :param dataset_size: How many records to generate.  Datasets A and B 
        will both have this many records.
    :param perc_exact_match: Percentage of records in dataset B that are 
        exact matches of records from dataset A
    :param perc_non_match: Percentage of records in dataset B that are 
        not in dataset A
    :param fuzzy_match_corruption_perc: Percentage of columns
        (attributes or properties of the records) to corrupt
    :param use_pareto_dist_for_degrees: If True, corruptions are made such that the
        resulting nodes have degrees (numbers of corruptions) that follow a Pareto
        distribution, i.e. most nodes have 1 random corruption, fewer nodes have
        2 random corruptions, even fewer nodes have 3 random corruptions, etc.  If
        False, corruptions are made by randomly sampling nodes to corrupt.  It is
        possible for nodes to have > 1 corruption if this argument is False, but
        not guaranteed (most will have only 1).

    :return: tuple of dataset A and B, where each dataset is a list of Entity instances
    """
    if (
        perc_exact_match < 0.0 or perc_exact_match > 1.0 
        or perc_non_match < 0.0 or perc_non_match > 1.0 
        or fuzzy_match_corruption_perc < 0.0 or fuzzy_match_corruption_perc > 1.0
    ):
        raise ValueError("At least one argument has an invalid percentage.")
    if perc_exact_match + perc_non_match > 1.0:
        raise ValueError("The sum of the percentages of exact matches and non matches exceeds 100%")

    nbr_non_matches = int(perc_non_match * dataset_size)
    nbr_matches = dataset_size - nbr_non_matches
    nbr_exact_matches = int(perc_exact_match * nbr_matches)
    nbr_fuzzy_matches = dataset_size - nbr_exact_matches - nbr_non_matches

    # generate dataset A
    entity_gen = entity_generator(nbr_entities=dataset_size)
    ds_a = [e for e in entity_gen]

    # generate dataset B
    entity_gen = entity_generator(nbr_entities=nbr_non_matches, id_offset=len(ds_a))
    if use_pareto_dist_for_degrees:
        ds_b = [e for e in entity_gen]
        entities_to_duplicate = random.sample(ds_a, k=nbr_matches)
        degree_dist, g = get_degree_distribution(
            nbr_samples=nbr_matches,
            max_degree=7,
            use_degree_offset=False,
            generate_random_graph=use_pareto_dist_for_degrees,
            verbose=False,
        )
        for idx, ent in enumerate(entities_to_duplicate):
            # create as many corruptions for each node as there are degrees for that node in the degree distribution
            for i in range(degree_dist[idx]):
                # stop duplicating records when the dataset size limit has been reached
                if len(ds_b) == dataset_size:
                    break
                is_exact_match = random.random() <= perc_exact_match
                if is_exact_match:
                    ds_b.append(ent)
                else:
                    corr = corrupt_entities([ent], perc_to_corrupt=fuzzy_match_corruption_perc)
                    ds_b += corr
    else:
        g = None
        ds_b = random.sample(ds_a, k=nbr_exact_matches)
        ds_b = ds_b + [e for e in entity_gen]
        # sample with replacement, so use random.choices here instead of random.sample
        ents_to_corrupt = random.choices(ds_a, k=nbr_fuzzy_matches)
        corr = corrupt_entities(ents_to_corrupt, perc_to_corrupt=fuzzy_match_corruption_perc)
        ds_b = ds_b + corr

    return ds_a, ds_b, g


def combine_datasets_and_save_to_parquet(ds: t.List[Entity], out_path: str) -> pl.DataFrame:
    """
    Converts a dataset to a Polars dataframe and dumps to parquet.
    Compression should be 'snappy' for compatibility with Spark.  Note
    that this uses pyarrow (C++) instead of the Rust backend.

    :param ds: List of Entity instances
    :param out_path: Where to save the parquet file

    :return: polars dataframe version of the dataset
    """
    ds_ent_properties = [ent.__dict__ for ent in ds]
    df = (
        pl.DataFrame(ds_ent_properties)
        .rename({"_id_number": "id"})
        .with_row_index(name='index', offset=0)
    )
    df.write_parquet(out_path, compression="snappy")
    return df


def list_to_pair_combos(iterable):
    """
    Converts a list of edges to a list of edge tuples, by creating a powerset from
    the list and removing any that are not length 2.
    """
    s = list(iterable)
    # powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    powerset = list(itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)
    ))
    pairs = [p for p in powerset if len(p) == 2]
    return pairs


def get_graph(nodes: list, edges: list, include_singletons: bool = True) -> nx.Graph:
    """
    Converts a list of unique nodes, and a list of clusters to a networkx graph.
    Edges is a list of clusters like [[node1, node2, node3], [node1], [node1, node2], ...]
    and gets converted to a list of tuples containing every pair from the clusters,
    like [(node1, node2), (node1, node3), (node2, node3), ...]
    """
    # convert from [[node1, node2], [node1], ...] to [(node1, node2), ...]
    # and eliminate the singletons/self-links
    edges = [
        el for edges in edges
        for el in [
            pair for pair in list_to_pair_combos(edges)
        ] if len(el) > 0
    ]

    # create the network
    g = nx.Graph()
    if include_singletons:
        g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    return g


def gini(x: np.ndarray):
    """Compute Gini coefficient of array of values"""
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return diffsum / (len(x)**2 * np.mean(x))


def compare_graph_attributes(g1: nx.MultiGraph, g2: nx.Graph, plot: bool) -> None:
    """Compares graph attributes between 2 graphs"""
    # compare the distribution of the lengths of the top n connected components
    top_n = 10
    g1_cc = np.array([
        len(c) for c in sorted(nx.connected_components(g1), key=len, reverse=True)[:top_n]
    ])
    g2_cc = np.array([
        len(c) for c in sorted(nx.connected_components(g2), key=len, reverse=True)[:top_n]
    ])

    if plot:
        fig = plt.figure(figsize=(12, 8))
        plt.title("Random Graph\nLinks are Randomly Predicted\n(nodes with 0 links are not shown)")
        nx.draw(g1, node_size=20)
        plt.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(12, 8))
        plt.title("Perfect Graph\nLinks are Perfectly Predicted\n(nodes with 0 links are not shown)")
        nx.draw(g2, node_size=20)
        plt.tight_layout()
        plt.show()

        # plt.hist(g1_cc, bins=4, label="Random Graph")
        # plt.hist(g2_cc, bins=4, label="Perfect Graph")
        # plt.title("Connected Component Size Distributions")
        # plt.xlabel("Connected Component Size")
        # plt.legend()
        # plt.show()

    print(
        f"Top {top_n} connected component sizes and Gini coefficients for\n"
        f"(random graph, perfect graph)\n{g1_cc.tolist(), g2_cc.tolist()}"
        f"\n{gini(g1_cc), gini(g2_cc)}"
    )


if __name__ == "__main__":
    DATASET_SIZE = 500
    MATCHING_SCHEME = {
        'exact': 0.2,
        'fuzzy': 0.8,
        'non': 0.0,
    }
    FUZZY_MATCH_CORRUPTION_AMOUNT = 0.5  # this percentage of columns will be corrupted
    USE_PARETO_DIST = True  # whether to sample degrees (nbr of corruptions) from a Pareto distribution
    # filename will have dataset size, match perc, corruption perc
    OUTPUT_FILE_PATH = (
        f"eval_data/raw-{int(DATASET_SIZE * 2)}-{str(1 - MATCHING_SCHEME['non']).replace('.', '_')}"
        f"-{str(FUZZY_MATCH_CORRUPTION_AMOUNT).replace('.', '_')}.parquet"
    )

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
