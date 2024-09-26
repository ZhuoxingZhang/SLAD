import copy
import os
import random

import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm


class Node:
    def __init__(self, trace_id, name, gid, error_type):
        self.error_type = error_type
        self.trace_id = trace_id
        self.gid = gid
        self.name = name
        self.children = []


def build_tree_from_txt(file_path, gid):
    root = Node("", "root", gid, 'no error')
    nodes = {"root:0": (root, -1, '', 0)}
    edge_info_tuples = []  # (target, source, weight)
    dup_node_count = {}  # key = node name, value = repeated number of the node
    anomalous_nodes = []  # anomalous nodes
    error_type = 'no error'

    with open(file_path, "r") as file:
        lines = file.readlines()
        # print(file_path)
        start_filter = lines.index("network[son<-parent]=\n")
        for line in lines[:start_filter]:
            if line.startswith("Error_Type="):
                error_type = line.strip().split("=")[1]
            if line.startswith("traceID="):
                trace_id = line.strip().split("=")[1]
                root.trace_id = trace_id
            if line.startswith("label="):
                ano_node = line.strip().split("=")[1]
                anomalous_nodes.append(ano_node)

        for line in lines[start_filter + 1:]:
            line = line.strip()
            if line:
                parts = line.split(",")
                edge = parts[0].split("<-")
                weight = int(parts[1])
                target_node_name = edge[0].strip()
                source_node_name = edge[1].strip()
                target_node_props = ",".join(parts[2:])

                if target_node_name in dup_node_count.keys():
                    dup_count = dup_node_count[target_node_name]
                    dup_node_count[target_node_name] = dup_count + 1
                else:
                    dup_count = 0
                    dup_node_count[target_node_name] = 1

                edge_info_tuples.append((target_node_name + ":" + str(dup_count), source_node_name, weight))
                if target_node_name in anomalous_nodes:  # 1 anomalous, 0 normal
                    node_label = 1
                else:
                    node_label = 0
                nodes[target_node_name + ":" + str(dup_count)] = (
                    Node(trace_id, target_node_name + ":" + str(dup_count), gid,
                         error_type if node_label == 1 else 'no error'), weight, target_node_props,
                    node_label)
    for target_node_name, source_node_name, weight in edge_info_tuples:
        target_node, t_weight, t_props, t_node_label = nodes[target_node_name]
        # print(target_node, ",", t_weight, ",", t_props, ",", t_node_label)
        src_nodes_candidate = []
        for k in nodes.keys():
            k1 = k.split(":")[0]
            if k1 == source_node_name:
                src_nodes_candidate.append(nodes[k])
        src_nodes_candidate.sort(key=lambda t: t[1])
        for src_node, src_weight, _, _ in src_nodes_candidate:
            if weight < src_weight or source_node_name == "root":
                src_node.children.append((target_node, weight, t_props, t_node_label))
                break

    return root


def read_event_one_hot(event_onehot_path, one_hot_length):
    event_name_map = {}
    with open(event_onehot_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line_old = line.split(" ->", 1)[0].replace("\n", "")
            line_new = line.split(" ->", 1)[1].replace("\n", "").replace("[", "").replace("]", "").replace(" ", "")
            ori_feat = [float(x) for x in line_new.split(",") if x]
            event_one_hot = ori_feat + [float(0) for i in range(one_hot_length - len(ori_feat))]
            if line_old in event_name_map:
                continue
            else:
                event_name_map[line_old] = event_one_hot
    return event_name_map


def read_file_one_hot(file_embedding_path, one_hot_length):
    file_name_map = {}
    with open(file_embedding_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            line_old = line.split(" ->", 1)[0].replace("\n", "")
            line_new = line.split(" ->", 1)[1].replace("\n", "").replace("[", "").replace("]", "").replace(" ", "")
            ori_feat = [float(x) for x in line_new.split(",") if x]
            file_one_hot = ori_feat + [float(0) for i in range(one_hot_length - len(ori_feat))]
            if line_old in file_name_map:
                continue
            else:
                file_name_map[line_old] = file_one_hot
    return file_name_map


def read_exception_one_hot(exception_embedding_path, one_hot_length):
    excep_name_map = {}
    with open(exception_embedding_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            line_old = line.split(" ->", 1)[0].replace("\n", "")
            line_new = line.split(" ->", 1)[1].replace("\n", "").replace("[", "").replace("]", "").replace(" ", "")
            ori_feat = [float(x) for x in line_new.split(",") if x]
            file_one_hot = ori_feat + [float(0) for i in range(one_hot_length - len(ori_feat))]
            if line_old in excep_name_map:
                continue
            else:
                excep_name_map[line_old] = file_one_hot
    return excep_name_map


def read_file_id_map(path):
    file_id_map = {"root": "0"}
    with open(path, "r") as read:
        lines = read.readlines()
    for line in lines:
        file, id = line.strip().split("->")
        file_id_map[file] = id
    return file_id_map


def read_event_id_map(path):
    event_id_map = {"": "0"}
    with open(path, "r") as read:
        lines = read.readlines()
    for i, line in enumerate(lines):
        event = line.strip()
        event_id_map[event] = str(i + 1)
    return event_id_map


def draw_pruning_graph(G, R, file_name):
    file_id_map = read_file_id_map("./forum_data/file_id_dict.txt")
    event_id_map = read_event_id_map("./forum_data/all_events.txt")
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    node_size = len(G.nodes())
    if node_size >= 50:
        plt.figure(figsize=(38, 20), dpi=100)
    elif 40 < node_size < 50:
        plt.figure(figsize=(34, 16), dpi=100)
    elif 20 <= node_size <= 40:
        plt.figure(figsize=(26, 14), dpi=100)
    else:
        plt.figure(figsize=(20, 12), dpi=100)
    # else:
    #     plt.figure(figsize=(12, 9), dpi=100)

    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=3000, alpha=0.8)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", edge_color="gray")

    node_labels = {
        n: n.replace(n.split(":")[0], file_id_map[n.split(":")[0]]) + "\nexp=" + str(d["exception"]) + "\nevent=" +
           event_id_map[str(d["event"])] + "\ngid=" + str(d["gid"]) + "\nR=" + str(R) for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=13, font_color="black")

    edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=15)

    plt.axis("off")
    # plt.tight_layout()

    if not os.path.exists("./figs"):
        os.makedirs("./figs")
    plt.savefig(f"./figs/{file_name}.png")
    plt.close()


def draw_graph(G):
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", edge_color="gray")

    node_labels = {n: n + "\n" + str(d["exception"]) + ":" + str(d["call_paths"]) for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_color="black")

    edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    plt.axis("off")
    plt.tight_layout()

    plt.show()


def print_mct(mct):
    G = nx.DiGraph()
    node_id_dict = {n: idx for idx, n in enumerate(mct.coalition)}
    print(mct.coalition)
    print(str(sorted([node_id_dict[x] for x in mct.coalition])))
    G.add_node(str(sorted([node_id_dict[x] for x in mct.coalition])))
    G.nodes[str(sorted([node_id_dict[x] for x in mct.coalition]))]["W"] = mct.W
    G.nodes[str(sorted([node_id_dict[x] for x in mct.coalition]))]["C"] = mct.C
    G.nodes[str(sorted([node_id_dict[x] for x in mct.coalition]))]["R"] = mct.R

    def traverse(node):
        for child in node.children:
            G.add_edge(str(sorted([node_id_dict[x] for x in node.coalition])),
                       str(sorted([node_id_dict[x] for x in child.coalition])))
            G.nodes[str(sorted([node_id_dict[x] for x in child.coalition]))]["W"] = child.W
            G.nodes[str(sorted([node_id_dict[x] for x in child.coalition]))]["C"] = child.C
            G.nodes[str(sorted([node_id_dict[x] for x in child.coalition]))]["R"] = child.R
            traverse(child)

    def draw_graph(G):
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
        nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=500, alpha=0.8)
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", edge_color="gray")
        pos_labels = {k: [v[0], v[1] + 4] for k, v in pos.items()}
        nx.draw_networkx_labels(G, pos_labels,
                                labels={node: ",".join([str(round(data["W"], 2)), str(round(data["C"], 2)),
                                                        str(round(data["R"], 2))]) + "\n" + node
                                        for node, data
                                        in G.nodes(data=True)}, font_color='red', font_size=10)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    traverse(mct)
    draw_graph(G)


def construct_tree_to_nx_with_feature(tree, event_map, file_name_map, exception_map):
    G = nx.Graph()
    feature_dim = []

    call_chain_paths = []  # root -> leaf
    current_call_chain = []
    weight_list = []

    def traverse(node, feature_dim):
        current_call_chain.append(node.name)
        if len(node.children) == 0:  # leaf node
            call_chain_paths.append(copy.deepcopy(current_call_chain))
        for child, weight, props, label in node.children:
            G.add_edge(child.name, node.name)
            _, cost, event, exception = props.split(",")  # using "," to split attributes of properties
            file_name_key = child.name.split(":")[0]
            if file_name_key not in file_name_map.keys():
                file_name_embedding = [float(0)] * len(list(file_name_map.values())[0])
            else:
                file_name_embedding = list(file_name_map[file_name_key])
            G.nodes[child.name]["feature"] = event_map[event.replace("event=", "", 1)] + file_name_embedding + \
                                             exception_map[
                                                 exception.replace("exception=", "", 1)] + [
                                                 float(cost.split("=")[1].split("m")[0])]
            if not feature_dim:
                feature_dim.append(len(G.nodes[child.name]["feature"]))
            G.nodes[child.name]["event"] = event.replace("event=", "", 1)
            G.nodes[child.name]["label"] = label
            G.nodes[child.name]["weight"] = weight
            G.nodes[child.name]["cost"] = float(cost.split("=")[1].split("m")[0])
            G.nodes[child.name]["trace_id"] = child.trace_id
            G.nodes[child.name]["name"] = child.name
            G.nodes[child.name]["gid"] = child.gid
            G.nodes[child.name]["error_type"] = child.error_type
            G.edges[(child.name, node.name)]["weight"] = weight
            weight_list.append(weight)
            if exception.replace("exception=", "", 1) != "null":
                G.nodes[child.name]["exception"] = True
            else:
                G.nodes[child.name]["exception"] = False
            traverse(child, feature_dim)
            current_call_chain.pop()  # pop last item of current call chain

    traverse(tree, feature_dim)
    """add each node's call chain path info"""
    for node in G.nodes():
        node_call_paths = []
        for i, path in enumerate(call_chain_paths):
            if node in path:
                node_call_paths.append(i)
        G.nodes[node]["call_paths"] = node_call_paths
    """add root node attributes"""
    G.nodes["root"]["event"] = ""
    G.nodes["root"]["feature"] = [float(0)] * feature_dim[0]
    G.nodes["root"]["label"] = 0
    G.nodes["root"]["weight"] = max(weight_list) + 1
    G.nodes["root"]["cost"] = 0.0
    G.nodes["root"]["exception"] = False
    G.nodes["root"]["trace_id"] = tree.trace_id
    G.nodes["root"]["name"] = "root"
    G.nodes["root"]["gid"] = tree.gid
    G.nodes["root"]["error_type"] = tree.error_type
    return G


def construct_graph_to_nx_with_feature(file_path, event_map, file_name_map, exception_map):
    anomalous_nodes = []  # anomalous nodes
    error_type = 'no error'
    edges_with_props = []
    with open(file_path, "r") as file:
        lines = file.readlines()
        start_filter = lines.index("network[son<-parent]=\n")
        for line in lines[:start_filter]:
            if line.startswith("Error_Type="):
                error_type = line.strip().split("=")[1]
            if line.startswith("traceID="):
                trace_id = line.strip().split("=")[1]
            if line.startswith("label="):
                ano_node = line.strip().split("=")[1]
                anomalous_nodes.append(ano_node)

        for line in lines[start_filter + 1:]:
            line = line.strip()
            if line:
                parts = line.split(",")
                edge = parts[0].split("<-")
                edge_info = [edge, parts[3], parts[4], parts[5],
                             1 if edge[0] in anomalous_nodes else 0]  # edge, cost, event, exception
                edges_with_props.append(edge_info)

    G = nx.Graph()
    feature_dim = []

    for edge_info in edges_with_props:
        target, source = edge_info[0]
        cost = edge_info[1]
        event = edge_info[2]
        exception = edge_info[3]
        label = edge_info[4]

        if not G.has_edge(target, source):
            G.add_edge(target, source)
        target_feature = event_map[event.replace("event=", "", 1)] + list(file_name_map[target]) + \
                         exception_map[
                             exception.replace("exception=", "", 1)] + [
                             float(cost.split("=")[1].split("m")[0])]
        if not feature_dim:
            feature_dim.append(len(target_feature))
        G.nodes[target]["label"] = label
        if label == 1:
            G.nodes[target]["error_type"] = error_type
        else:
            G.nodes[target]["error_type"] = 'no error'
        G.nodes[target]["name"] = target
        if G.nodes[target].get('exception', None) is None:
            if exception.replace("exception=", "", 1) != "null":
                G.nodes[target]["exception"] = True
            else:
                G.nodes[target]["exception"] = False
        else:
            if exception.replace("exception=", "", 1) != "null":
                G.nodes[target]["exception"] = True

        feature_value = G.nodes[target].get('feature', None)
        if feature_value is not None:
            feature_value = [x + y for x, y in zip(target_feature, feature_value)]
            G.nodes[target]["feature"] = feature_value
        else:
            G.nodes[target]["feature"] = target_feature
    root_path = []
    for i, node in enumerate(list(G.nodes())):
        if node != "root":
            G.nodes[node]["call_paths"] = [i]
            root_path.append(i)
    """add root node attributes"""
    G.nodes["root"]["feature"] = [float(0)] * feature_dim[0]
    G.nodes["root"]["label"] = 0
    G.nodes["root"]["exception"] = False
    G.nodes["root"]["name"] = "root"
    G.nodes["root"]["call_paths"] = root_path
    G.nodes["root"]["error_type"] = 'no error'
    return G


def parse_k_hop_graph_to_data(k_hop_subgraph, target_node_label, target_node):
    feats = torch.tensor([k_hop_subgraph.nodes[node]["feature"] for node in k_hop_subgraph.nodes()], dtype=torch.float)
    edges = [[list(k_hop_subgraph.nodes).index(u), list(k_hop_subgraph.nodes).index(v)] for u, v in
             k_hop_subgraph.edges]
    edge_index = np.transpose(edges).tolist()
    data = Data(x=feats, edge_index=torch.tensor(edge_index).long(), y=torch.tensor(target_node_label).long(),
                target_node_index=list(k_hop_subgraph.nodes()).index(target_node))
    return data


def read_same_graph_structures_dict(dataset, directory, event_map, file_name_map, exception_map):
    graph_dataset_dict = {}
    for file in tqdm([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]):
        gid = file.split("-")[0]
        if dataset == "halo":
            g = construct_graph_to_nx_with_feature(os.path.join(directory, file), event_map, file_name_map,
                                                   exception_map)
        else:
            root = build_tree_from_txt(os.path.join(directory, file), gid)
            g = construct_tree_to_nx_with_feature(root, event_map, file_name_map, exception_map)
        if gid in graph_dataset_dict:
            graph_dataset_dict[gid].append((g, os.path.join(directory, file)))
        else:
            graph_dataset_dict[gid] = [(g, os.path.join(directory, file))]
    return graph_dataset_dict


def parse_graphs_to_dataset(graphs, error_type_dict):
    dataset = []
    for graph in graphs:
        feats = torch.tensor([graph.nodes[node]["feature"] for node in graph.nodes()], dtype=torch.float)
        edges = [[list(graph.nodes).index(u), list(graph.nodes).index(v)] for u, v in
                 graph.edges]
        edge_index = np.transpose(edges).tolist()
        labels = torch.tensor([graph.nodes[node]["label"] for node in graph.nodes()], dtype=torch.long)
        error_types = torch.tensor([error_type_dict[graph.nodes[node]["error_type"]] for node in graph.nodes()], dtype=torch.long)
        data = Data(x=feats, edge_index=torch.tensor(edge_index).long(), y=labels, error_types=error_types)
        dataset.append(data)
    return dataset


def are_graphs_equal(graph1, graph2):
    nodes_equal = set(graph1.nodes()) == set(graph2.nodes())
    edges_equal = sorted([sorted([u, v]) for u, v in graph1.edges()]) == sorted(
        [sorted([u, v]) for u, v in graph2.edges()])
    return nodes_equal and edges_equal


def get_k_hop_subgraph_of_target_node(dataset_name, graph_dataset, k, id_start):
    all_k_hop_subgraphs = []
    graph_unique_id = id_start
    for graph in graph_dataset:
        graph_copy = copy.deepcopy(graph)
        graph_copy_undirected = graph_copy.to_undirected()
        neighbors_dict = {
            node: list(nx.single_source_shortest_path_length(graph_copy_undirected, node, cutoff=k).keys()) for node in
            graph_copy_undirected.nodes}
        for target_node in graph.nodes:
            if target_node == "root":
                continue
            neighbors = neighbors_dict[target_node]
            subgraph = graph.subgraph(neighbors)
            target_node_label = graph.nodes[target_node]["label"]
            ok = True
            if dataset_name != "halo":
                for _, tar_node, _, subg, _ in all_k_hop_subgraphs:
                    if tar_node == target_node and are_graphs_equal(subg, subgraph):
                        ok = False
                        break
            if ok:
                all_k_hop_subgraphs.append((graph_unique_id, target_node, target_node_label, subgraph,
                                            parse_k_hop_graph_to_data(subgraph, target_node_label,
                                                                      target_node)))
                # id, target_node, label, k-hop subgraph, data object of subgraph
                graph_unique_id += 1

    return all_k_hop_subgraphs


def split_train_val_test_set(train_set_ratio, val_set_ratio, all_graphs_dict, n):
    train_set_num = int(len(all_graphs_dict) * train_set_ratio)
    val_set_num = int(len(all_graphs_dict) * val_set_ratio)
    data = []
    for gid, same_graphs in all_graphs_dict.items():
        data.append((gid, same_graphs))
    random.shuffle(data)  # 打乱数据
    train_graph_struc_set = []  # pick one of each structural graphs in train set
    train_set = []
    train_set_paths = []
    val_set = []
    val_set_paths = []
    test_set = []
    test_set_unique = {}
    test_set_paths = []
    for i in range(len(data)):
        if i < train_set_num:
            train_graph_struc_set.append(data[i][1][0][0])
            if n == 0:
                train_set.extend([g for g, gpath in data[i][1]])
                train_set_paths.extend([(data[i][0], gpath) for g, gpath in data[i][1]])
            else:
                train_set.extend([g for g, gpath in data[i][1]][:n])
                train_set_paths.extend([(data[i][0], gpath) for g, gpath in data[i][1]][:n])
        elif train_set_num <= i < train_set_num + val_set_num:
            if n == 0:
                val_set.extend([g for g, gpath in data[i][1]])
                val_set_paths.extend([(data[i][0], gpath) for g, gpath in data[i][1]])
            else:
                val_set.extend([g for g, gpath in data[i][1]][:n])
                val_set_paths.extend([(data[i][0], gpath) for g, gpath in data[i][1]][:n])
        else:
            test_set.extend([g for g, gpath in data[i][1]])
            test_set_unique[data[i][0]] = data[i][1][0][0]
            test_set_paths.extend([(data[i][0], gpath) for g, gpath in data[i][1]])
    test_set_unique = [g for gid, g in test_set_unique.items()]
    return train_set, val_set, test_set, test_set_unique, train_graph_struc_set, train_set_paths, val_set_paths, test_set_paths


def write_train_val_test_set_paths(train_set_paths, val_set_paths, test_set_paths, output_path):
    with open(output_path, 'w', encoding='utf8') as file:
        file.write("train set paths:\n")
        for g_id, p in train_set_paths:
            file.write(str(g_id) + " : " + p + "\n")
        file.write("validation set paths:\n")
        for g_id, p in val_set_paths:
            file.write(str(g_id) + " : " + p + "\n")
        file.write("test set paths:\n")
        for g_id, p in test_set_paths:
            file.write(str(g_id) + " : " + p + "\n")


def load_specific_train_val_test_set(dataset, dataset_path, event_map, file_name_map, exception_map, n):
    with open(dataset_path, 'r', encoding='utf8') as file:
        lines = file.readlines()
    train_idx = lines.index("train set paths:\n")
    val_idx = lines.index("validation set paths:\n")
    test_idx = lines.index("test set paths:\n")
    train_paths = [line.strip() for line in lines[train_idx + 1:val_idx] if line.strip()]
    val_paths = [line.strip() for line in lines[val_idx + 1:test_idx] if line.strip()]
    test_paths = [line.strip() for line in lines[test_idx + 1:] if line.strip()]
    train_graph_set = []
    train_graph_struc_set = {}
    val_graph_set = []
    test_graph_set = []
    test_graph_set_unique = {}
    gid_num_dict = {}
    for line in train_paths:
        gid, path = line.split(" : ")
        if gid not in gid_num_dict:
            gid_num_dict[gid] = 1
            if dataset == "halo":
                g = construct_graph_to_nx_with_feature(path, event_map, file_name_map, exception_map)
            else:
                root = build_tree_from_txt(path, gid)
                g = construct_tree_to_nx_with_feature(root, event_map, file_name_map, exception_map)
            train_graph_set.append(g)
            train_graph_struc_set[gid] = g
        else:
            if n == 0 or gid_num_dict[gid] < n:
                gid_num_dict[gid] = gid_num_dict[gid] + 1
                if dataset == "halo":
                    g = construct_graph_to_nx_with_feature(path, event_map, file_name_map, exception_map)
                else:
                    root = build_tree_from_txt(path, gid)
                    g = construct_tree_to_nx_with_feature(root, event_map, file_name_map, exception_map)
                train_graph_set.append(g)
                train_graph_struc_set[gid] = g
    for line in val_paths:
        gid, path = line.split(" : ")
        if gid not in gid_num_dict:
            gid_num_dict[gid] = 1
            if dataset == "halo":
                g = construct_graph_to_nx_with_feature(path, event_map, file_name_map, exception_map)
            else:
                root = build_tree_from_txt(path, gid)
                g = construct_tree_to_nx_with_feature(root, event_map, file_name_map, exception_map)
            val_graph_set.append(g)
        else:
            if n == 0 or gid_num_dict[gid] < n:
                gid_num_dict[gid] = gid_num_dict[gid] + 1
                if dataset == "halo":
                    g = construct_graph_to_nx_with_feature(path, event_map, file_name_map, exception_map)
                else:
                    root = build_tree_from_txt(path, gid)
                    g = construct_tree_to_nx_with_feature(root, event_map, file_name_map, exception_map)
                val_graph_set.append(g)
    for line in test_paths:
        gid, path = line.split(" : ")
        if dataset == "halo":
            g = construct_graph_to_nx_with_feature(path, event_map, file_name_map, exception_map)
        else:
            root = build_tree_from_txt(path, gid)
            g = construct_tree_to_nx_with_feature(root, event_map, file_name_map, exception_map)
        test_graph_set.append(g)
        test_graph_set_unique[gid] = g
    train_graph_struc_set = [g for gid, g in train_graph_struc_set.items()]
    test_graph_set_unique = [g for gid, g in test_graph_set_unique.items()]
    return train_graph_set, val_graph_set, test_graph_set, test_graph_set_unique, train_graph_struc_set


def load_dataset(dataset_name, dataset_id, use_specific_dataset, most_n_same_graph_structure, train_set_ratio,
                 val_set_ratio, dataloader_shuffle, batch_size, k):
    dataset_dir = f"./{dataset_name}_dataset"
    root = f"./{dataset_name}_data"
    event_embedding_path = root + "/events_compress_one_hot.txt"
    file_embedding_path = root + "/file_name_one_hot.txt"
    exception_embedding_path = root + "/exception_list.txt"
    if dataset_name == "forum":
        one_hot_length = 82
    elif dataset_name == "halo":
        one_hot_length = 161
    else:
        raise ValueError("not supportive dataset name for one ont length.")
    event_map = read_event_one_hot(event_embedding_path, one_hot_length)
    file_name_map = read_file_one_hot(file_embedding_path, one_hot_length)
    exception_map = read_exception_one_hot(exception_embedding_path, one_hot_length)
    specific_train_val_test_set_path = root + f"/specific_dataset_{dataset_id}.txt"

    if use_specific_dataset:
        train_graph_set, val_graph_set, test_graph_set, test_graph_set_unique, train_graph_struc_set = load_specific_train_val_test_set(
            dataset_name,
            specific_train_val_test_set_path, event_map, file_name_map, exception_map, most_n_same_graph_structure)
    else:
        all_graphs_dict = read_same_graph_structures_dict(dataset_name, dataset_dir, event_map, file_name_map,
                                                          exception_map)
        train_graph_set, val_graph_set, test_graph_set, test_graph_set_unique, train_graph_struc_set, train_set_paths, val_set_paths, test_set_paths = split_train_val_test_set(
            train_set_ratio,
            val_set_ratio,
            all_graphs_dict, most_n_same_graph_structure)
        write_train_val_test_set_paths(train_set_paths, val_set_paths, test_set_paths,
                                       specific_train_val_test_set_path)
    if dataset_name == "halo":  # sample
        candidate_positive = []
        candidate_negative = []
        for g in train_graph_struc_set:
            if any(node_data['label'] == 1 for node_data in g.nodes.values()):
                candidate_positive.append(g)
            else:
                candidate_negative.append(g)
        print("positive graphs:", len(candidate_positive))
        positive_num = min(2000, len(candidate_positive))
        train_graph_struc_set = []
        train_graph_struc_set.extend(random.sample(candidate_positive, positive_num))
        # train_graph_struc_set.extend(random.sample(candidate_negative, positive_num))
    print(
        f"train graph num: {len(train_graph_set)}, validation graph num:{len(val_graph_set)}, test graph num:{len(test_graph_set)}, train graph structure num: {len(train_graph_struc_set)}")
    "%---------error type dict----------%"
    "0: no error; 1: chain change; 2: call change; 3: condition change; 4: argument change"
    error_type_dict = {'no error': 0, 'chain change': 1, 'call change': 2, 'condition change': 3, 'argument change': 4}
    "%----------------------------------%"
    train_dataset = parse_graphs_to_dataset(train_graph_set, error_type_dict)
    val_dataset = parse_graphs_to_dataset(val_graph_set, error_type_dict)
    test_dataset = parse_graphs_to_dataset(test_graph_set, error_type_dict)
    test_dataset_unique = parse_graphs_to_dataset(test_graph_set_unique, error_type_dict)
    train_prot_dataset = parse_graphs_to_dataset(train_graph_struc_set, error_type_dict)
    train_k_hop_graphs = get_k_hop_subgraph_of_target_node(dataset_name, train_graph_struc_set, k=k, id_start=0)

    print("k hop train graph number:", len(train_k_hop_graphs))
    train_prot_dataloader = DataLoader(train_prot_dataset, batch_size=len(train_prot_dataset),
                                       shuffle=dataloader_shuffle)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=dataloader_shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=dataloader_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=dataloader_shuffle)
    test_dataloader_unique = DataLoader(test_dataset_unique, batch_size=batch_size, shuffle=dataloader_shuffle)
    return train_dataloader, val_dataloader, test_dataloader, test_dataloader_unique, train_prot_dataloader, train_k_hop_graphs
