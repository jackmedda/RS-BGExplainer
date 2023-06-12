import numba
import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
import scipy.sparse as sp

import biga.utils as utils


def get_nx_adj_matrix(dataset):
    uid_field = dataset.uid_field
    iid_field = dataset.iid_field
    n_users = dataset.num(uid_field)
    n_items = dataset.num(iid_field)
    inter_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
    num_all = n_users + n_items
    A = get_adj_from_inter_matrix(inter_matrix, num_all, n_users)

    return nx.Graph(A)


def get_nx_biadj_matrix(dataset, remove_first_row_col=False):
    inter_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
    if remove_first_row_col:
        inter_matrix = inter_matrix[1:, 1:]

    return nx.bipartite.from_biadjacency_matrix(inter_matrix)


def get_bipartite_igraph(dataset, remove_first_row_col=False):
    inter_matrix = dataset.inter_matrix(form='csr').astype(np.float32)
    if remove_first_row_col:
        inter_matrix = inter_matrix[1:, 1:]

    incid_adj = ig.Graph.Incidence(inter_matrix.todense().tolist())
    bip_info = np.concatenate([np.zeros(inter_matrix.shape[0], dtype=int), np.ones(inter_matrix.shape[1], dtype=int)])

    return ig.Graph.Bipartite(bip_info, incid_adj.get_edgelist())


def _igraph_distances(graph):
    import joblib
    vcount = graph.vcount()
    distances = np.zeros((vcount, vcount), dtype=int)

    def _add_sp(i):
        distances[i] = [len(sp) for sp in graph.get_shortest_paths(i)]

    joblib.Parallel(n_jobs=-1, require='sharedmem')(
        joblib.delayed(_add_sp)(i) for i in range(vcount)
    )

    return distances


def extract_graph_metrics_per_node(dataset, remove_first_row_col=False, metrics="all", **sp_kwargs):
    metrics = ["Degree", "Sparsity", "Reachability", "Sharing Potentiality"] if metrics == "all" else metrics
    sp_kwargs = sp_kwargs or dict(length=2, depth=2)

    item_hist, _, item_hist_len = dataset.history_user_matrix()
    user_hist, _, user_hist_len = dataset.history_item_matrix()

    G_df = None
    node_col = 'Node'

    igg = get_bipartite_igraph(dataset, remove_first_row_col=remove_first_row_col)
    for metr in metrics:
        if metr == "Degree":
            df = pd.DataFrame(dict(zip(igg.vs.indices, igg.degree())).items(), columns=[node_col, metr])
        elif metr == "Reachability":
            last_user_id = dataset.user_num - (1 + remove_first_row_col)
            user_reach = get_user_reachability(igg, last_user_id=last_user_id)

            first_item_id = last_user_id + 1
            item_reach = get_item_reachability(igg, first_item_id=first_item_id)

            df = pd.DataFrame({**user_reach, **item_reach}.items(), columns=[node_col, metr])
        elif metr == "Sparsity" or metr == "Density":
            if remove_first_row_col:
                item_hist = item_hist[1:].where(item_hist[1:] == 0, item_hist[1:] - 1)
                item_pop = item_hist_len
                user_hist = user_hist[1:].where(user_hist[1:] == 0, user_hist[1:] - 1)
                user_hist_len = user_hist_len

            user_density = np.nan_to_num(
                ((item_hist_len[user_hist] / user_hist.shape[0]).sum(dim=1) / user_hist_len[1:]).numpy(),
                nan=0
            )
            user_metric = 1 - user_density if metr == "Sparsity" else user_density

            # item density represents the activity of the users that interact with an item.
            # If only a user interact with item X and this user interacted with all the items in the catalog, then
            # the density of X is maximum. A low density then means a high sparsity, which means the users that interact
            # with that item interact with a few others
            item_density = np.nan_to_num(
                ((user_hist_len[item_hist] / item_hist.shape[0]).sum(dim=1) / item_pop[1:]).numpy(),
                nan=0
            )
            item_metric = 1 - item_density if metr == "Sparsity" else item_density

            df = pd.DataFrame(
                zip(igg.vs.indices, np.concatenate([user_metric, item_metric])),
                columns=[node_col, metr]
            )
        elif metr == "Sharing Potentiality":
            n_users = user_hist[1:].shape[0]
            user_user = get_node_node_graph_data(user_hist[1:].numpy())
            user_user = np.asarray(sp.coo_matrix(
                (user_user[:, -1], (user_user[:, 0], user_user[:, 1])), shape=(n_users, n_users)
            ).todense())
            user_sp = compute_sharing_potentiality(
                user_user, user_hist_len[1:].numpy(), **sp_kwargs
            )

            n_items = item_hist[1:].shape[0]
            item_item = get_node_node_graph_data(item_hist[1:].numpy())
            item_item = np.asarray(sp.coo_matrix(
                (item_item[:, -1], (item_item[:, 0], item_item[:, 1])), shape=(n_items, n_items)
            ).todense())
            item_sp = compute_sharing_potentiality(
                item_item, item_hist_len[1:].numpy(), **sp_kwargs
            )

            sp_data = np.concatenate([user_sp, item_sp])
            df = pd.DataFrame(zip(range(len(sp_data)), sp_data), columns=[node_col, metr])

        if G_df is None:
            G_df = df
        else:
            G_df = G_df.join(df.set_index(node_col), on=node_col)
    return G_df


def get_user_reachability(graph, last_user_id):
    return get_reachability_per_node(graph, last=last_user_id)


def get_item_reachability(graph, first_item_id):
    return get_reachability_per_node(graph, first=first_item_id)


def get_reachability_per_node(graph, first=None, last=None, nodes=None):
    if nodes is not None:
        nodes = sorted(nodes)
    else:
        nodes = np.arange(graph.vcount())

#         dist = dist[first:][:, first:] if first is not None else dist
#         nodes = nodes[first:] if first is not None else nodes

#         dist = dist[:(last + 1)][:, :(last + 1)] if last is not None else dist
#         nodes = nodes[:(last + 1)] if last is not None else nodes
        if first is not None:
            nodes = nodes[first:]
        if last is not None:
            nodes = nodes[:(last + 1)]

    dist = np.array(graph.distances(source=nodes, target=nodes))

    reach = _get_reachability_per_node(dist)

    return dict(zip(nodes, reach))


def get_reachability_source_target(graph, source, target):
    dist = np.array(graph.distances(source=source, target=target))
    import pdb; pdb.set_trace()

    reach = _get_reachability_per_node(dist)

    return dict(zip(nodes, reach))


@numba.jit(nopython=True, parallel=True)
def _get_reachability_per_node(dist):
    n_nodes = dist.shape[0]
    reach = np.zeros((n_nodes,), dtype=np.float32)

    for i in numba.prange(n_nodes):
        n_dist = dist[i]
        mask = (~np.isinf(n_dist)) & (n_dist > 0)
        n_reach = np.bincount(n_dist[mask].astype(np.int32))
        n_reach = n_reach[n_reach > 0]
        reach[i] = compute_reachability(n_reach) / n_nodes

    return reach


@numba.jit(nopython=True, parallel=True)
def compute_reachability(reach):
    discount = np.arange(reach.shape[0]) + 1
    return (reach / discount).sum()


@numba.jit(nopython=True, parallel=True)
def compute_sharing_potentiality(common_data, hist_len, length=2, depth=2):
    n = common_data.shape[0]
    res = np.zeros((n,), dtype=np.float32)
    for i in numba.prange(n):
        most_sim = np.argsort(common_data[i])[::-1][:length]
        res[i] += _compute_sp_length(i, most_sim, common_data, hist_len)
        for d in range(depth - 1):
            most_d_sim = np.argsort(common_data[most_sim[d]])[::-1][:length]
            res[i] += _compute_sp_length(most_sim[d], most_d_sim, common_data, hist_len) / (2 + d)
    return res


@numba.jit(nopython=True)
def _compute_sp_length(data_i, most_sim, common_data, hist_len):
    sp_length = 0
    for i in numba.prange(most_sim.shape[0]):
        sim = common_data[data_i, most_sim[i]]
        if hist_len[data_i] == 0 and hist_len[most_sim[i]] > 0:
            sp_length += 1  # most_sim[i] can share every item
        elif hist_len[data_i] == 0 or hist_len[most_sim[i]] == 0:
            sp_length += 0
        else:
            sp_length += sim / hist_len[data_i] * (1 - sim / hist_len[most_sim[i]])
    return sp_length / most_sim.shape[0]


def get_centrality_graph_df(graph_nx, top, original=True, sens_attr_map=None):
    label = "Original" if original else "Perturbed"

    bottom = set(graph_nx) - top

    node_type_map = dict(zip(top, ["users"] * len(top)))
    node_type_map.update(dict(zip(bottom, ["items"] * len(bottom))))

    centr = nx.bipartite.degree_centrality(graph_nx, top)

    top, bottom = list(top), list(bottom)
    df_data = zip(top + bottom, [centr[n] for n in (top + bottom)])

    df = pd.DataFrame(df_data, columns=["node_id_minus_1", "Centrality"])

    df["Node Type"] = df["node_id_minus_1"].map(node_type_map)

    user_df, item_df = df[df["Node Type"] == "users"], df[df["Node Type"] == "items"]

    if sens_attr_map is not None:
        user_df["Group"] = user_df["node_id_minus_1"].map(sens_attr_map)

    user_df["Graph Type"], item_df["Graph Type"] = label, label

    return user_df, item_df


def get_user_user_data_sens_df(dataset, user_df, sens_attr, attr_map=None):
    user_history, _, _ = dataset.history_item_matrix()
    user_graph_df = get_node_node_data_feature_df(user_history.numpy(), user_df, dataset.uid_field, sens_attr, attr_map=attr_map)

    return user_graph_df


def get_item_item_data_pop_df(dataset, item_df, pop_attr):
    item_history, _, _ = dataset.history_user_matrix()
    item_graph_df = get_node_node_data_feature_df(item_history.numpy(), item_df, dataset.iid_field, pop_attr)

    return item_graph_df


def get_node_node_data_feature_df(history, node_df, id_label, feat_attr, attr_map=None):
    graph = get_node_node_graph_data(history)
    graph = graph[graph[:, -1] > 0]

    graph_df = pd.DataFrame(graph, columns=[f'{id_label}_1', f'{id_label}_2', 'n_common_edges'])
    graph_df = graph_df.join(
        node_df.set_index(id_label), on=f"{id_label}_1"
    ).reset_index(drop=True).rename(columns={feat_attr: f"{feat_attr}_1"})

    if attr_map is not None:
        graph_df[f"{feat_attr}_1"] = graph_df[f"{feat_attr}_1"].map(attr_map)

    graph_df = graph_df.join(
        node_df.set_index(id_label), on=f'{id_label}_2'
    ).reset_index(drop=True).rename(columns={feat_attr: f"{feat_attr}_2"})

    if attr_map is not None:
        graph_df[f"{feat_attr}_2"] = graph_df[f"{feat_attr}_2"].map(attr_map)

    return graph_df


def get_node_node_graph_data(history):
    return _get_node_node_graph_data(history)


@numba.jit(nopython=True, parallel=True)
def _get_node_node_graph_data(history):
    hist = [set(h) for h in history]
    n_nodes = history.shape[0] - 1  # removed padding 0
    node_node = np.zeros((n_nodes * (n_nodes - 1) // 2, 3), dtype=np.int32)  # number of combinations

    for n1 in numba.prange(1, n_nodes + 1):
        _inner_combinations(n1, n_nodes, hist, node_node)

    return node_node


@numba.jit(nopython=True, parallel=True)
def _inner_combinations(n1, n_nodes, hist, node_node):
    for n2 in numba.prange(n1 + 1, n_nodes + 1):
        node_node[sum(range(n_nodes - n1 + 1, n_nodes)) + (n2 - n1 - 1)] = [
            n1, n2, len(hist[n1] & hist[n2]) - 1
        ]


def compute_homophily(graph_df, group_sizes, feat_attr):
    homophily = {}
    for (gr1, gr2), gr_df in graph_df.groupby([f"{feat_attr}_1", f"{feat_attr}_2"]):
        if gr1 == gr2:
            all_gr_edges = graph_df.loc[
                (graph_df[f"{feat_attr}_1"] == gr1) |  # interactions of gr1 with any group (also gr1)
                ((graph_df[f"{feat_attr}_2"] == gr1) & (graph_df[f"{feat_attr}_1"] != gr1)),  # interactions from gr2 to gr1
                'n_common_edges'
            ].sum()
            homophily[gr1] = gr_df['n_common_edges'].sum() / all_gr_edges - group_sizes[gr1]

    return homophily
