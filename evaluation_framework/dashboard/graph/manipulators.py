import pandas as pd
from tqdm import tqdm

from .primitives.entities import Node, Link


def create_nodes_dict(df: pd.DataFrame, extra_features_to_include_in_data: tuple = ()) -> dict:
    """
    Creates a node dictionary from a dataframe of features.  The resulting dict will be
    {node_id: Node instance}
    TODO: this function should use node features, not pair features like it does now

    :param df: Pandas dataframe with links as a MultiIndex (node_id1, node_id2).  This dataframe
        contains a row for every pair of nodes, and the columns are features that describe the pair.
    :param extra_features_to_include_in_data: If provided, these will be pair features to include
        in the node data.  This parameter would be more useful if the nodes dict were constructed
        from node data, instead of pairs data.
    """
    if (
        len(extra_features_to_include_in_data) > 0 
        and not set(extra_features_to_include_in_data).issubset(df.columns)
    ):
        raise ValueError(
            f"One or more {extra_features_to_include_in_data} "
            f"could not be found in columns: {df.columns}"
        )

    nodes_dict = dict()
    for node_id_tuple in tqdm(set(df.index)):
        for node_id in node_id_tuple:

            # map feature names to their values for this specific node
            keyword_params = {
                feature: df.loc[node_id_tuple, feature]
                for feature in extra_features_to_include_in_data
            }

            # populate the dict entry for this node
            nodes_dict[node_id] = Node(
                _id=node_id,
                _label=str(node_id),
                _classes=None,
                **keyword_params
            )

    return nodes_dict


def create_links_dict(idx: pd.MultiIndex, nodes_dict: dict, classes: str = "") -> dict:
    """
    Creates a link dictionary from a Pandas MultiIndex of (node_id1, node_id2).
    The resulting dict will be {(node_id1, node_id2): Link instance}.
    TODO: weights are assumed to be uniform, it would be nice if they could vary

    :param idx: Index of links (node_id1, node_id2)
    :param nodes_dict: Dictionary of node IDs mapped to Node instances
    :param classes: CSS style selectors for Cytoscape for these links.
    """
    links_dict = {
        r: Link(source=nodes_dict[r[0]], target=nodes_dict[r[1]], weight=1, classes=classes)
        for r in set(idx)
    }
    return links_dict


def remove_nodes_that_are_not_in_links(nodes_dict: dict, links_dict: dict) -> dict:
    """
    Removes nodes that are not in the links, so they will not show up in the graph.

    :param nodes_dict: Dictionary of node IDs mapped to Node instances
    :param links_dict: Dictionary of link IDs (tuples) mapped to Link instances
    """
    linked_node_ids = set([ll for l in list(links_dict) for ll in l])
    nodes_dict = {n: nodes_dict[n] for n in linked_node_ids}
    return nodes_dict


def build_graph_elements(nodes_dict: dict, links_dict: dict) -> list:
    """
    Creates graph elements for Dash Cytoscape.  Cytoscape's elements must be a list of
    dictionaries with a data property.  The classes property helps with styling but is
    optional.

    :param nodes_dict: Dictionary of node IDs mapped to Node instances
    :param links_dict: Dictionary of link IDs (tuples) mapped to Link instances
    """
    graph_nodes = [{'data': n.data} for n in nodes_dict.values()]
    edges = [{'data': e.data, 'classes': e.classes} for e in links_dict.values()]
    return graph_nodes + edges


def filter_to_false_positives(
    nodes_dict: dict,
    fp_set: set,
    fp_links_only: bool = False,
    pred_links_dict: dict = None,
) -> list:
    """
    Filters graph elements to nodes with false positives.

    :param nodes_dict: Dictionary of node IDs mapped to Node instances
    :param fp_set: Set of tuples of false positive links
    :param fp_links_only: if True, return false positive links only.  If False (default),
        return all links of nodes with false positive links, so that the FP links can be
        compared to other link types for the effected nodes.
    :param pred_links_dict: Dictionary of link IDs (tuples) mapped to Link instances
    """
    if not fp_links_only and pred_links_dict is None:
        raise ValueError("pred_links_dict cannot be None if fp_links_only is False")

    if fp_links_only:
        links_dict = {
            r: Link(source=nodes_dict[r[0]], target=nodes_dict[r[1]], weight=1, classes="fp")
            for r in fp_set
        }
    else:
        # flatten the FP set of tuples to get a list of nodes with incorrectly predicted links
        nodes_w_fps = [n for fp in fp_set for n in fp]

        # filter the predicted links to the ones with errors
        links_dict = {
            k: v for k, v in pred_links_dict.items()
            if (v.source.id in nodes_w_fps or v.target.id in nodes_w_fps)
        }

    # remove any nodes that are not linked, so that they will not appear in the graph
    nodes_dict = remove_nodes_that_are_not_in_links(links_dict=links_dict, nodes_dict=nodes_dict)

    return build_graph_elements(nodes_dict=nodes_dict, links_dict=links_dict)


def filter_to_false_negatives(
    nodes_dict: dict,
    fn_set: set,
    fn_links_only: bool = False,
    pred_links_dict: dict = None,
) -> list:
    """
    Filters graph elements to nodes with false negatives.

    :param nodes_dict: Dictionary of node IDs mapped to Node instances
    :param fn_set: Set of tuples of false negative links
    :param fn_links_only: if True, return false negative links only.  If False (default),
        return all links of nodes with false negative links, so that the FN links can be
        compared to other link types for the effected nodes.
    :param pred_links_dict: Dictionary of link IDs (tuples) mapped to Link instances
    """
    if not fn_links_only and pred_links_dict is None:
        raise ValueError("pred_links_dict cannot be None if fn_links_only is False")

    if fn_links_only:
        links_dict = {
            r: Link(source=nodes_dict[r[0]], target=nodes_dict[r[1]], weight=1, classes="fn")
            for r in fn_set
        }
    else:
        # flatten the FN set of tuples to get a list of nodes with incorrectly predicted links
        nodes_w_fns = [n for fn in fn_set for n in fn]

        # filter the predicted links to the ones with errors
        links_dict = {
            k: v for k, v in pred_links_dict.items()
            if (v.source.id in nodes_w_fns or v.target.id in nodes_w_fns)
        }

    # remove any nodes that are not linked, so that they will not appear in the graph
    nodes_dict = remove_nodes_that_are_not_in_links(links_dict=links_dict, nodes_dict=nodes_dict)

    return build_graph_elements(nodes_dict=nodes_dict, links_dict=links_dict)


def filter_to_errors(nodes_dict: dict, fp_set: set, fn_set: set, pred_links_dict: dict = None) -> list:
    """
    Filters graph elements to nodes with errors (FP or FN).

    :param nodes_dict: Dictionary of node IDs mapped to Node instances
    :param fp_set: Set of tuples of false positive links
    :param fn_set: Set of tuples of false negative links
    :param pred_links_dict: Dictionary of link IDs (tuples) mapped to Link instances
    """
    # combine FP and FN sets and flatten to get a list of nodes with incorrectly predicted links
    nodes_w_errors = [n for err in fp_set.union(fn_set) for n in err]

    # filter the predicted links to the ones with errors
    links_dict = {
        k: v for k, v in pred_links_dict.items()
        if (v.source.id in nodes_w_errors or v.target.id in nodes_w_errors)
    }

    # remove any nodes that are not linked, so that they will not appear in the graph
    nodes_dict = remove_nodes_that_are_not_in_links(links_dict=links_dict, nodes_dict=nodes_dict)

    return build_graph_elements(nodes_dict=nodes_dict, links_dict=links_dict)


'''
# create node and links objects for the test set graph
nodes = {rr: Node(_id=rr, _label=str(rr), _classes=None, some_other_node_feature="test") for r in set(test_x.index) for rr in r}  # {id: N}
links = create_links_dict(idx=test_y)  # {(node_id1, node_id2): L}

# create predicted links for the graph
pred_links = {r: Link(source=nodes[r[0]], target=nodes[r[1]], weight=1, classes="tp") for r in tps}
pred_links.update({r: Link(source=nodes[r[0]], target=nodes[r[1]], weight=1, classes="fp") for r in fps})
pred_links.update({r: Link(source=nodes[r[0]], target=nodes[r[1]], weight=1, classes="fn") for r in fns})

nodes = remove_nodes_that_are_not_in_links(nodes_dict=nodes, links_dict=pred_links)

# elements = build_graph_elements(nodes_dict=nodes, links_dict=pred_links)
# elements = filter_to_false_positives(nodes_dict=nodes, fp_set=set(fps), fp_links_only=False, pred_links_dict=pred_links)
# elements = filter_to_false_negatives(nodes_dict=nodes, fn_set=set(fns), fn_links_only=False, pred_links_dict=pred_links)
elements = filter_to_errors(nodes_dict=nodes, fp_set=set(fps), fn_set=set(fns), pred_links_dict=pred_links)
'''
