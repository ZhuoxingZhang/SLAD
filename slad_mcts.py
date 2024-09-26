import math
import random

import torch
import networkx as nx
from sklearn.cluster import KMeans
from torch_geometric.data import Data, Batch
import numpy as np
from collections import Counter

from tqdm import tqdm
import concurrent.futures


class MCTSNode():

    def __init__(self, coalition: list,
                 ori_graph: nx.Graph, c_puct: float = 10.0,
                 W: float = 0, C: int = 0, R: float = 0, alpha_e: float = 1000.0, alpha_c: float = 1000.0):
        self.coalition = coalition  # node set
        self.ori_graph = ori_graph
        self.c_puct = c_puct  # control the trade-off between exploration and exploitation
        self.children = []
        self.W = W  # total reward
        self.C = C  # count of being selected
        self.R = R  # immediate reward
        self.alpha_e = alpha_e  # considering if pruning nodes with exception info
        self.alpha_c = alpha_c  # considering if pruning nodes in same call chain to target node

    def Q(self):
        return self.W / self.C if self.C > 0 else 0

    def U(self, n):
        return self.c_puct * self.R * math.sqrt(n) / (1 + self.C)

    def E(self):
        """try to not remove nodes first which have exception info"""
        cur_subgraph_exception_count = len(
            [node for node in self.coalition if self.ori_graph.nodes[node]["exception"] == True])
        return self.alpha_e * (cur_subgraph_exception_count / self.ori_graph.number_of_nodes())

    def CC(self, target_node):
        """try to not remove nodes first which is not in same call chain with target node"""
        target_node_paths = self.ori_graph.nodes[target_node]["call_paths"]
        node_count_in_chain = 0
        for cand_node in self.coalition:
            cand_node_paths = self.ori_graph.nodes[cand_node]["call_paths"]
            if set(cand_node_paths) & set(target_node_paths):
                node_count_in_chain += 1
        return self.alpha_c * (node_count_in_chain / self.ori_graph.number_of_nodes())


def mcts_rollout(k_neighbors, all_node_embeddings, all_node_labels, mcts_stop_strategy,
                 reward_list_in_decision_path, mcts_pruning_strategy,
                 current_MCT_node,
                 state_map, subgraph_embedding_dict,
                 k_hop_input_graph, min_atoms,
                 target_node, target_node_label, gnnNet, decision_path_nodes):
    cur_graph_coalition = current_MCT_node.coalition
    if mcts_stop_strategy == "auto":
        if len(reward_list_in_decision_path) > 5:  # path length must > 5
            avg_reward = sum(reward_list_in_decision_path[:-1]) / len(reward_list_in_decision_path[:-1])
            if reward_list_in_decision_path[-1] < avg_reward or len(cur_graph_coalition) <= 3:
                # if last node's reward, in path, is smaller than average reward of the previous nodes of the last node
                return
    elif mcts_stop_strategy == "specific_size":
        if len(cur_graph_coalition) <= min_atoms:  # minimal sub-graph size
            return
    else:
        raise ValueError("not supportive mcts stop strategy.")

    # Expand if this node has never been visited
    if len(current_MCT_node.children) == 0:
        node_degree_list = list(
            k_hop_input_graph.subgraph(cur_graph_coalition).degree)  # node degree,like [(node0, 2), (node1, 1)]
        node_degree_list = sorted(node_degree_list, key=lambda x: x[1],
                                  reverse=False)  # node degree increasing order, like [(node1, 1),(node0, 2)]
        all_nodes = [x[0] for x in node_degree_list]

        expand_nodes = [x[0] for x in node_degree_list if x[1] == node_degree_list[0][1]]
        if target_node in expand_nodes:
            expand_nodes.remove(target_node)  # ensure target node not being removed
            if not expand_nodes:
                expand_nodes = [x[0] for x in node_degree_list if
                                x[1] == node_degree_list[1][1]]
        """
        strategy: try not to prune the node 
        that is either in same call path with target node or exception info node
        """
        if mcts_pruning_strategy == "normal":
            no_prune_nodes = []
            current_g = k_hop_input_graph.subgraph(cur_graph_coalition)
            target_node_paths = current_g.nodes[target_node]["call_paths"]
            for cand_node in expand_nodes:
                cand_node_paths = current_g.nodes[cand_node]["call_paths"]
                if (set(cand_node_paths) & set(target_node_paths)) or current_g.nodes[cand_node]["exception"] == True:
                    no_prune_nodes.append(cand_node)

            if len(no_prune_nodes) != len(expand_nodes):
                expand_nodes = [element for element in expand_nodes if element not in no_prune_nodes]

        for each_node in expand_nodes:
            # for each node, pruning it and get the remaining sub-graph
            # here we check the resulting sub-graphs and only keep the largest one
            subgraph_coalition = [node for node in all_nodes if node != each_node]
            new_graph_coalition = sorted(subgraph_coalition)
            # check the state map and merge the same sub-graph
            Find_same = False
            for old_graph_node in state_map.values():
                if Counter(old_graph_node.coalition) == Counter(new_graph_coalition):
                    new_node = old_graph_node
                    Find_same = True
                    break

            if Find_same == False:
                new_node = MCTSNode(new_graph_coalition, ori_graph=k_hop_input_graph)
                state_map[str(new_graph_coalition)] = new_node

            Find_same_child = False
            for cur_child in current_MCT_node.children:
                if Counter(cur_child.coalition) == Counter(new_graph_coalition):
                    Find_same_child = True
                    break

            if Find_same_child == False:
                current_MCT_node.children.append(new_node)  # add new node to MCT

        # compute each node instant reward
        for child in current_MCT_node.children:
            if str(child.coalition) not in subgraph_embedding_dict:
                score, subgraph_emb = subgraph_score(k_neighbors, all_node_embeddings, all_node_labels,
                                                     target_node, target_node_label,
                                                     child.coalition,
                                                     k_hop_input_graph,
                                                     gnnNet)
                child.R = score
                subgraph_embedding_dict[str(child.coalition)] = subgraph_emb

    sum_count = sum([c.C for c in current_MCT_node.children])
    """pruning strategy: Q + U/ Q + U + E + CC"""
    if mcts_pruning_strategy == "soft_pruning":
        next_node_candidates = []
        possible_selected_MCT_node = max(current_MCT_node.children,
                                         key=lambda x: x.Q() + x.U(sum_count) + x.E() + x.CC(target_node))
        max_value = possible_selected_MCT_node.Q() + possible_selected_MCT_node.U(
            sum_count) + possible_selected_MCT_node.E() + possible_selected_MCT_node.CC(target_node)
        for node in current_MCT_node.children:
            if (node.Q() + node.U(sum_count) + node.E() + node.CC(target_node)) == max_value:
                next_node_candidates.append(node)
        selected_MCT_node = random.choice(next_node_candidates)
    elif mcts_pruning_strategy == "normal" or mcts_pruning_strategy == "original":
        next_node_candidates = []
        possible_selected_MCT_node = max(current_MCT_node.children, key=lambda x: x.Q() + x.U(sum_count))
        max_value = possible_selected_MCT_node.Q() + possible_selected_MCT_node.U(sum_count)
        for node in current_MCT_node.children:
            if (node.Q() + node.U(sum_count)) == max_value:
                next_node_candidates.append(node)
        selected_MCT_node = random.choice(next_node_candidates)
    elif mcts_pruning_strategy == "random":
        selected_MCT_node = random.choice(current_MCT_node.children)
    else:
        raise ValueError("not supportive mcts strategy.")
    """Update value"""
    decision_path_nodes.append(selected_MCT_node)
    reward_list_in_decision_path.append(selected_MCT_node.R)
    mcts_rollout(k_neighbors, all_node_embeddings, all_node_labels, mcts_stop_strategy,
                 reward_list_in_decision_path, mcts_pruning_strategy,
                 selected_MCT_node,
                 state_map, subgraph_embedding_dict,
                 k_hop_input_graph, min_atoms,
                 target_node, target_node_label, gnnNet, decision_path_nodes)
    return


def mcts_single_subgraph_in_parallel(k_neighbors, all_node_embeddings, all_node_labels,
                                     mcts_stop_strategy, mcts_pruning_strategy, rollout_num,
                                     min_atoms, gnnNet,
                                     target_node, target_node_label, k_hop_graph):
    root_coalition = sorted(list(k_hop_graph.nodes()))
    root = MCTSNode(root_coalition, ori_graph=k_hop_graph)
    state_map = {}
    subgraph_embedding_dict = {}
    all_path_subgraph_dict = {}
    for i in range(rollout_num):
        decision_path_nodes = []
        reward_list_in_decision_path = []
        mcts_rollout(k_neighbors, all_node_embeddings, all_node_labels, mcts_stop_strategy,
                     reward_list_in_decision_path, mcts_pruning_strategy,
                     root,
                     state_map, subgraph_embedding_dict, k_hop_graph,
                     min_atoms, target_node, target_node_label,
                     gnnNet, decision_path_nodes)  # 搜索最优路径
        """visualization"""
        # draw_pruning_graph(k_hop_graph, 0, "original_k_hop")
        # j = 0
        # for sub_g, r in zip(decision_path_nodes, reward_list_in_decision_path):
        #     sub_g = k_hop_graph.subgraph(sub_g.coalition)
        #     draw_pruning_graph(sub_g, r, target_node + "_" + str(i) + "_" + str(j))
        #     j = j + 1

        """Update path node value"""
        for path_subgraph in decision_path_nodes:
            all_path_subgraph_dict[str(path_subgraph.coalition)] = path_subgraph
        average_reward = sum(node.R for node in decision_path_nodes) / len(decision_path_nodes) if len(
            decision_path_nodes) != 0 else 0.0
        for node in decision_path_nodes:
            node.W += average_reward
            node.C += 1

    all_subgraphs = [node for _, node in all_path_subgraph_dict.items()]
    if mcts_pruning_strategy == "random":
        random.shuffle(all_subgraphs)
    else:
        all_subgraphs = sorted(all_subgraphs, key=lambda x: x.R, reverse=True)

    res = []
    topk = 10
    for selected in all_subgraphs:
        if mcts_pruning_strategy == "random":
            if len(res) < topk:
                candidate_prototype = subgraph_embedding_dict[str(selected.coalition)]
                res.append((target_node_label, candidate_prototype.detach(), selected.R))
        else:
            if selected.R > 0 and len(res) < topk:
                candidate_prototype = subgraph_embedding_dict[str(selected.coalition)]
                res.append((target_node_label, candidate_prototype.detach(), selected.R))
    return res


def select_prototypes_with_most_distance_each_other(bound, num_prototype, all_cand_prototype_list):
    if bound > len(all_cand_prototype_list):  # in case of out of bound
        bound = len(all_cand_prototype_list)
    top_bound_prots = all_cand_prototype_list[:bound]
    tensor_matrix = torch.stack(top_bound_prots)
    distances = torch.norm(tensor_matrix.unsqueeze(0) - tensor_matrix.unsqueeze(1), dim=2)
    dis_threshold = torch.max(distances).item()
    print("max distance over selected prots:", dis_threshold)
    candidates = []
    need_loop = True
    while need_loop:
        for i in range(bound):
            if not candidates:
                candidates.append((top_bound_prots[i], i))
            else:
                ok = True
                for cand, idx in candidates:
                    if distances[i][idx].item() < dis_threshold:
                        ok = False
                        break
                if ok:
                    candidates.append((top_bound_prots[i], i))
            if i == bound - 1 and len(candidates) != num_prototype:
                dis_threshold = dis_threshold * 0.99
                candidates.clear()
            if len(candidates) == num_prototype:
                need_loop = False
                break
    candidates = [candidate for candidate, _ in candidates]
    print("min distance over final prots:", dis_threshold)
    return torch.stack(candidates)


def mcts(dataset, train_prot_dataloader, rollout_num, min_atoms, k_hop_graphs, gnnNet, prototype_num_each_class,
         random_negative_sample, device, mcts_pruning_strategy,
         mcts_stop_strategy, k_neighbors, mcts_top_bound, mcts_prot_selection):
    all_node_embeddings = []
    all_node_labels = []
    for batch in train_prot_dataloader:
        batch = batch.to(device)
        with torch.no_grad():
            node_embeddings = gnnNet(batch.x, batch.edge_index)
        node_emb_labels = batch.y
        all_node_embeddings.append(node_embeddings)
        all_node_labels.append(node_emb_labels)

    all_node_embeddings = torch.cat(all_node_embeddings, dim=0)
    all_node_labels = torch.cat(all_node_labels, dim=0)

    """sample"""
    if random_negative_sample:
        positive_k_hop_graphs = []
        negative_k_hop_graphs = []
        for graph_id, target_node, target_node_label, k_hop_graph, k_hop_graph_data in k_hop_graphs:
            if target_node_label == 1:
                positive_k_hop_graphs.append((graph_id, target_node, target_node_label, k_hop_graph, k_hop_graph_data))
            if target_node_label == 0:
                negative_k_hop_graphs.append((graph_id, target_node, target_node_label, k_hop_graph, k_hop_graph_data))
        if dataset == 'halo':
            sample_num = 1000
            print("positive k-hop graphs:", len(positive_k_hop_graphs), "negative k-hop graphs:",
                  len(negative_k_hop_graphs))
            negative_k_hop_graphs = random.sample(negative_k_hop_graphs, sample_num)
            positive_k_hop_graphs = random.sample(positive_k_hop_graphs, sample_num)
        else:
            negative_k_hop_graphs = random.sample(negative_k_hop_graphs,
                                                  len(positive_k_hop_graphs))
        print(
            f"positive k hop subgraph num: {len(positive_k_hop_graphs)}, negative k hop subgraph num: {len(negative_k_hop_graphs)}")
        random_k_hop_graphs = []
        random_k_hop_graphs.extend(positive_k_hop_graphs)
        random_k_hop_graphs.extend(negative_k_hop_graphs)
    else:
        random_k_hop_graphs = k_hop_graphs

    """visualization"""
    # for item in random_k_hop_graphs_new:
    #     k_hop_g = item[3][0]
    #     if "pub.developers.forum.facade.impl.UserApiServiceImpl:0" in item[1] and k_hop_g.nodes[list(k_hop_g.nodes())[0]][
    #         "gid"] == "106":
    #         visual = item
    #         break
    # print("find specific k hop subgraph!")
    # random_k_hop_graphs_new = [visual]

    # multi-process mcts
    task_parameters = [(k_neighbors, all_node_embeddings, all_node_labels, mcts_stop_strategy,
                        mcts_pruning_strategy, rollout_num,
                        min_atoms, gnnNet,
                        target_node, target_node_label, k_hop_graph) for
                       graph_id, target_node, target_node_label, k_hop_graph, k_hop_graph_data in
                       random_k_hop_graphs if
                       k_hop_graph.number_of_nodes() > min_atoms]

    """parallel computing"""
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(mcts_single_subgraph_in_parallel, *params) for params in task_parameters]
        list(tqdm(concurrent.futures.as_completed(futures), total=len(task_parameters)))

    prot_candidates_list = []
    for future in futures:
        prot_candidates_list.extend(future.result())

    prot_candidates_dict = {}
    for target_node_label, candidate_prototype, R in prot_candidates_list:
        if target_node_label in prot_candidates_dict:
            prot_cand_list = prot_candidates_dict[target_node_label]
            prot_cand_list.append((candidate_prototype, R))
            prot_candidates_dict[target_node_label] = prot_cand_list
        else:
            prot_candidates_dict[target_node_label] = [(candidate_prototype, R)]

    new_prots_list = []
    if mcts_prot_selection == "kmeans" and mcts_pruning_strategy != "random":
        for prototype_class, cand_prot_list in prot_candidates_dict.items():
            cand_prot_list = sorted(cand_prot_list, key=lambda x: x[1], reverse=True)  # decreasing order of similarity score
            cand_prots = [(prot.tolist(), R) for prot, R in cand_prot_list]
            unique_cand_prots = []
            for prot, R in cand_prots:
                unique = True
                for u, r in unique_cand_prots:
                    if prot == u:
                        unique = False
                        break
                if unique:
                    unique_cand_prots.append((prot, R))
            """select prototype with greatest R value in every clusters by kmeans"""
            new_prototypes = select_m_prototypes_by_kmeans(unique_cand_prots[:mcts_top_bound],
                                                           num_clusters=prototype_num_each_class)
            new_prots_list.append((prototype_class, new_prototypes))
    elif mcts_prot_selection == "topk" and mcts_pruning_strategy != "random":
        for prototype_class, cand_prot_list in prot_candidates_dict.items():
            cand_prot_list = sorted(cand_prot_list, key=lambda x: x[1], reverse=True)
            cand_prots = [prot for prot, _ in cand_prot_list]
            unique_cand_prots = []
            for prot in cand_prots:
                unique = True
                for u in unique_cand_prots:
                    if torch.equal(u, prot):
                        unique = False
                        break
                if unique:
                    unique_cand_prots.append(prot)
            """from top bound of candidates, get new representative sub-structures"""
            new_prototypes = select_prototypes_with_most_distance_each_other(mcts_top_bound, prototype_num_each_class,
                                                                             unique_cand_prots)
            new_prots_list.append((prototype_class, new_prototypes))
    elif mcts_pruning_strategy == "random":
        for prototype_class, cand_prot_list in prot_candidates_dict.items():
            cand_prots = [prot for prot, _ in cand_prot_list]
            random.shuffle(cand_prots)
            new_prots_list.append((prototype_class, torch.stack(cand_prots[:prototype_num_each_class])))
    else:
        raise ValueError("not supportive prototype selection ways.")
    new_prots_list = sorted(new_prots_list, key=lambda x: x[0])
    combined_prot_layer = [new_prototypes for p_class, new_prototypes in new_prots_list]
    new_prototype_layer = torch.stack(combined_prot_layer)
    return new_prototype_layer.detach()


def select_m_prototypes_by_kmeans(data, num_clusters):
    data_array = np.array([(np.array(x), r) for x, r in data], dtype=object)
    X = np.array([x for x, r in data_array])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
    cluster_assignments = kmeans.fit_predict(X)
    res = []
    for i in range(num_clusters):
        cluster_data = data_array[cluster_assignments == i]
        max_R_index = np.argmax(cluster_data[:, 1])
        max_R_data = cluster_data[max_R_index]
        res.append(list(max_R_data[0]))
    return torch.Tensor(res).to('cuda') if torch.cuda.is_available() else torch.Tensor(res).to('cpu')


def subgraph_score(k_neighbors, all_node_embeddings, all_node_labels,
                   target_node, target_node_label,
                   coalition, k_hop_input_graph, gnnNet):
    subgraph_emb = subgraph_embedding(target_node, coalition, k_hop_input_graph, gnnNet)
    avg_similarity_score = graph_emb_and_node_embs_similarity(k_neighbors, all_node_embeddings,
                                                              all_node_labels,
                                                              target_node_label, subgraph_emb)
    return avg_similarity_score, subgraph_emb


def subgraph_embedding(target_node, coalition, k_hop_input_graph, gnnNet):
    subG = k_hop_input_graph.subgraph(coalition)

    subG_features = [k_hop_input_graph.nodes[node]['feature'] for node in subG.nodes]

    subG_features = torch.tensor(subG_features, dtype=torch.float32)

    subG_edges = [[list(subG.nodes).index(u), list(subG.nodes).index(v)] for u, v in subG.edges]
    subG_edge_index = np.transpose(subG_edges).tolist()

    data = Data(x=subG_features, edge_index=torch.tensor(subG_edge_index).long())
    if torch.cuda.is_available():
        data = data.to('cuda')
    data = Batch.from_data_list([data])

    target_node_idx = list(subG.nodes).index(target_node)
    with torch.no_grad():
        target_node_embedding = gnnNet(data.x, data.edge_index)[target_node_idx]
    return target_node_embedding


def graph_emb_and_node_embs_similarity(k_neighbors, all_node_embeddings, all_node_labels, target_node_label,
                                       subgraph_emb):
    """ the average similarity value of subgraph and k nearest node embeddings"""
    score = knn(all_node_embeddings, all_node_labels, subgraph_emb, target_node_label, k_neighbors)
    # return score if score > 0 else 0.0
    return score


def knn(all_node_embeddings, all_node_labels, subg_embedding, target_node_label, topk):
    distances = torch.norm(all_node_embeddings - subg_embedding, dim=1) ** 2
    epsilon = 1e-4
    similarity_list = torch.log((distances + 1) / (distances + epsilon))
    top_k_indices = torch.topk(similarity_list, k=topk, largest=True).indices
    labels_top_k = all_node_labels[top_k_indices]
    weighted_distances = torch.sum(
        similarity_list[top_k_indices] * (2 * (labels_top_k == target_node_label).float() - 1)) / topk
    return weighted_distances.item()
