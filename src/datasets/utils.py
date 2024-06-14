import os
import shutil
import pickle
from tqdm import tqdm
from math import log
from itertools import combinations
import heapq
import numpy as np
import random
import networkx as nx

def load_graph(datasetname):
    graph_list_file = os.path.join(f'../data/{datasetname}/common/raw', 'graph_list.pickle')
    with open(graph_list_file, 'rb') as f:
        graph_list = pickle.load(f)
    return graph_list

def save_graph(destination_directory, graph_list):
    graph_list_file = destination_directory + '/graph_list.pickle'
    with open(graph_list_file, 'wb') as f:
        pickle.dump(graph_list, f)

def save_train_graph(destination_directory, graph_list):
    graph_list_file = destination_directory + '/train_graph_list.pickle'
    with open(graph_list_file, 'wb') as f:
        pickle.dump(graph_list, f)

def save_train_equal_dict(destination_directory, equal_dict):
    graph_list_file = destination_directory + '/train_equal_dict.pickle'
    with open(graph_list_file, 'wb') as f:
        pickle.dump(equal_dict, f)

def save_test_equal_dict(destination_directory, equal_dict):
    graph_list_file = destination_directory + '/test_equal_dict.pickle'
    with open(graph_list_file, 'wb') as f:
        pickle.dump(equal_dict, f)


def save_test_graph(destination_directory, graph_list):
    graph_list_file = destination_directory + '/test_graph_list.pickle'
    with open(graph_list_file, 'wb') as f:
        pickle.dump(graph_list, f)


def save_idx_file(destination_directory, idxs):
    idx_file = destination_directory + '/idx_file.pickle'
    with open(idx_file, 'wb') as f:
        pickle.dump(idxs, f)

def load_equal_dict(datasetname):
    equal_dict_file = os.path.join(f'../data/{datasetname}/common/raw', 'equal_dict.pickle')
    with open(equal_dict_file, 'rb') as f:
        equal_dict = pickle.load(f)
    return equal_dict


def load_node_mapping(datasetname):
    node_mapping_file = os.path.join(f'../data/{datasetname}/common/raw', 'node_mapping.pickle')
    with open(node_mapping_file, 'rb') as f:
        node_mapping = pickle.load(f)
    return node_mapping

def load_edge_mapping(datasetname):
    edge_mapping_file = os.path.join(f'../data/{datasetname}/common/raw', 'edge_mapping.pickle')
    with open(edge_mapping_file, 'rb') as f:
        edge_mapping = pickle.load(f)
    return edge_mapping


def get_key_combinations(input_dict):
    keys = list(input_dict.keys())  # 获取字典中的所有键
    key_combinations = list(combinations(keys, 2))  # 获取所有键的排列组合
    result_list = [[key1, key2] for key1, key2 in key_combinations]  # 将每个组合转换为包含两个元素的列表
    return result_list

def load_node_features(datasetname):
    node_features_file = os.path.join(f'../data/{datasetname}/common/raw', 'entity_embeddings.pickle')
    with open(node_features_file, 'rb') as f:
        node_features = pickle.load(f)
    return node_features

def load_edge_features(datasetname):
    edge_features_file = os.path.join(f'../data/{datasetname}/common/raw', 'relation_embeddings.pickle')
    with open(edge_features_file, 'rb') as f:
        edge_features = pickle.load(f)
    return edge_features

def random_remove(graph):
    edges = list(graph.edges)
    chosen_edge = random.choice(edges)
    # graph.remove_edge(chosen_edge[0], chosen_edge[1])
    graph.remove_edge(*chosen_edge)
    return graph

def random_add(graph):
    nonedges = list(nx.non_edges(graph))
    if len(nonedges) != 0:
        chosen_nonedge = random.choice(nonedges)
        # graph.add_edge(chosen_nonedge[0], chosen_nonedge[1])
        graph.add_edge(*chosen_nonedge)
    return graph

def graph_list_random_remove(graph_list, num_links):
    for idx, graph in tqdm(enumerate(graph_list)):
        edges_num = len(graph.edges)
        if 0.3 * edges_num < num_links:
            continue
        for i in range(num_links):
            graph = random_remove(graph)
        graph_list[idx] = graph
    return graph_list

def graph_list_random_add(graph_list, num_links):
    for idx, graph in tqdm(enumerate(graph_list)):
        edges_num = len(graph.edges)
        if 0.3 * edges_num < num_links:
            continue
        for i in range(num_links):
            graph = random_add(graph)
        graph_list[idx] = graph
    return graph_list

def add_gaussian_noise(features, std=1, mean=0):
    noise = np.random.normal(mean, std, features.shape)
    return noise

def calculate_reference_amplitude(features):
    absolute_values = np.abs(features)
    reference_amplitude = np.max(absolute_values)
    return reference_amplitude


def random_sample_with_id(input_list, sample_number):
    if sample_number <= 0:
        return [], []

    # 生成索引列表
    index_list = list(range(len(input_list)))

    # 对索引列表进行随机采样
    sampled_indices = random.sample(index_list, min(sample_number, len(index_list)))

    # 根据采样的索引获取元素和对应的id
    sampled_elements = [input_list[i] for i in sampled_indices]
    sampled_ids = [sampled_indices[i] for i in range(len(sampled_indices))]

    return sampled_elements, sampled_ids

def copy_files(source_dir, destination_dir):
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    for filename in os.listdir(source_dir):
        source_file_path = os.path.join(source_dir, filename)
        destination_file_path = os.path.join(destination_dir, filename)

        if os.path.isfile(source_file_path):
            shutil.copy(source_file_path, destination_file_path)
            print(f"Done Copying File: {filename} To {destination_dir}")
        elif os.path.isdir(source_file_path):
            copy_files(source_file_path, os.path.join(destination_dir, filename))

def get_edge_triplet(graph_list):
    edge_triplets = []
    for graph in graph_list:
        edge_triplet = [(graph.edges[source, target, key]['src'],
                         graph.edges[source, target, key]['dst'],
                         graph.edges[source, target, key]['relation'])
                        for source, target, key in
                         graph.edges(keys=True)]
        edge_triplets.append(edge_triplet)
    return edge_triplets

def k_smallest_keys(input_dict, k):
    if k <= 0:
        return []

    # 使用最小堆来保存字典的键值对，以值为比较对象
    heap = [(value, key) for key, value in input_dict.items()]
    heapq.heapify(heap)

    # 获取前k个最小值的键
    smallest_keys = [heapq.heappop(heap)[1] for _ in range(min(k, len(heap)))]

    return smallest_keys

def select_elements_with_ratio(input_list, train_split):
    if not 0 < train_split <= 1:
        raise ValueError("train_split must be between 0 and 1")

    # 计算需要选择的元素数量
    num_elements_to_select = int(len(input_list) * train_split)

    # 随机选择元素
    selected_elements = random.sample(input_list, num_elements_to_select)

    return selected_elements

def other_part_of_list(original_list, part):
    other_part = []
    for item in original_list:
        if item not in part:
            other_part.append(item)
    return other_part

def get_events(edge_triplets):
    events = set()
    for edges in edge_triplets:
        for edge in edges:
            events.add(edge)
    return events

def get_idf(edge_triplets, events):
    idf = {}
    S = len(edge_triplets)+1
    for event in events:
        idf[event] = 0
        for edge_triplet in edge_triplets:
            for edge in edge_triplet:
                if event == edge:
                    idf[event] += 1
                    break
    for k in idf.keys():
        idf[k] = float(log(float(S / (idf[k]+1)),10))

    return idf

def get_graphs_from_ids(idxs, all_graphs):
    graphs = list()
    for id in idxs:
        graphs.append(all_graphs[id])
    return graphs

def update_big_list(big_list, small_list, index_list):
    for i, index in enumerate(index_list):
        big_list[index] = small_list[i]
    return big_list

def get_index_mapping(my_list):
    my_dict = {}
    for index, value in enumerate(my_list):
        my_dict[value] = index
    return my_dict


def attack_substructure(dataset, type_1, type_2, graph_list, equal_dict, portion, substructure_portion):
    if type_1 != 'all':
        graph_1_idxs = equal_dict[type_1]
        graphs_1 = get_graphs_from_ids(graph_1_idxs, graph_list)
    else:
        graph_1_idxs = list(range(len(graph_list)))
        graphs_1 = graph_list

    if type_2 != 'all':
        graph_2_idxs = equal_dict[type_2]
        graphs_2 = get_graphs_from_ids(graph_2_idxs, graph_list)
    else:
        graph_2_idxs = list(range(len(graph_list)))
        graphs_2 = graph_list


    edge_triplets_all = get_edge_triplet(graph_list)
    edge_triplets_type_2 = get_edge_triplet(graphs_2)

    events = get_events(edge_triplets_type_2)
    idf = get_idf(edge_triplets_all, events)

    substru_size = int(len(idf) * substructure_portion)
    sample_number = int(len(graphs_1) * portion)
    substructure = k_smallest_keys(idf, substru_size)

    poison_graph_list, sampled_ids =  random_sample_with_id(graphs_1, sample_number)
    edge_mapping = load_edge_mapping(dataset)
    node_mapping = load_node_mapping(dataset)

    edge_list = []
    for edge in substructure:
        src_id = node_mapping[edge[0]]
        dst_id = node_mapping[edge[1]]
        rel_id = edge_mapping[edge[2]]
        edge_list.append([src_id, dst_id, rel_id])

    for sampled_id in sampled_ids:
        graph = graphs_1[sampled_id]
        for idx, edge in enumerate(edge_list):
            check_add = True
            node_1_attr = dict()
            node_2_attr = dict()
            if not graph.has_node(edge[0]):
                for checkgraph in graphs_2:
                    if checkgraph.has_node(edge[0]):
                        node_1_attr = checkgraph.nodes[edge[0]]
                        if 'name' not in node_1_attr.keys():
                            check_add = False
                            print('Check Add')
                        else:
                            graph.add_node(edge[0], name=node_1_attr['name'],
                                                     entity_type=node_1_attr['entity_type'])
                            break
            if not graph.has_node(edge[1]):
                for checkgraph in graphs_2:
                    if checkgraph.has_node(edge[1]):
                        node_2_attr = checkgraph.nodes[edge[1]]
                        if 'name' not in node_2_attr.keys():
                            check_add = False
                            print('Check Add')

                        else:
                            graph.add_node(edge[1], name=node_2_attr['name'],
                                                     entity_type=node_2_attr['entity_type'])
                            break
            if check_add:
                if 'name' not in node_1_attr.keys():
                    pass
                    # print('Check Add')
                else:
                    graph.add_node(edge[0], name=node_1_attr['name'],
                                   entity_type=node_1_attr['entity_type'])
                if 'name' not in node_2_attr.keys():
                    pass
                    #print('Check Add')
                else:
                    print('Add successfully')
                    graph.add_node(edge[1], name=node_2_attr['name'],
                                   entity_type=node_2_attr['entity_type'])
                if 'name'  in node_1_attr.keys() and 'name'  in node_2_attr.keys():
                    graph.add_edge(edge[0], edge[1], src=substructure[idx][0], dst=substructure[idx][1], type='audit', relation=substructure[idx][2], log_type=substructure[idx][2])
        graphs_1[sampled_id] = graph

    graph_list = update_big_list(graph_list, graphs_1, graph_1_idxs)
    return graph_list


def evasion(source_directory, dataset, mode, target, source, portion, substructure_portion):
    train_split = 0.8
    destination_directory = f"../data/{dataset}/{mode}_{target.replace('/', '_')}_{source.replace('/', '_')}_{portion}_{substructure_portion}/raw"
    copy_files(source_directory, destination_directory)
    graph_list = load_graph(dataset)
    equal_dict = load_equal_dict(dataset)

    train_equal_dict = dict()
    test_equal_dict = dict()
    valid_equal_dict = dict()
    train_idxs = list()
    test_idxs = list()
    valid_idxs = list()

    for key, value in equal_dict.items():
        train_equal_dict[key] = select_elements_with_ratio(equal_dict[key], train_split)
        test_valid = other_part_of_list(equal_dict[key], train_equal_dict[key])
        # 使用切片将列表分成两半
        half_length = len(test_valid) // 2
        test_equal_dict[key] = test_valid[:half_length]
        valid_equal_dict[key] = test_valid[half_length:]

        train_idxs.extend(train_equal_dict[key])
        test_idxs.extend(test_equal_dict[key])
        valid_idxs.extend(valid_equal_dict[key])

    train_idxs.sort()
    test_idxs.sort()
    valid_idxs.sort()

    idxs = {'train': np.array(train_idxs), 'valid': np.array(valid_idxs), 'test': np.array(test_idxs)}

    # train_mapping = get_index_mapping(train_idxs)
    test_mapping = get_index_mapping(test_idxs)

    train_graph_list = get_graphs_from_ids(train_idxs, graph_list)
    test_graph_list = get_graphs_from_ids(test_idxs, graph_list)
    valid_graph_list = get_graphs_from_ids(valid_idxs, graph_list)

    # for key, value in train_equal_dict.items():
    #     for idx in range(len(train_equal_dict[key])):
    #         train_equal_dict[key][idx] = train_mapping[train_equal_dict[key][idx]]
    #
    for key, value in test_equal_dict.items():
        for idx in range(len(test_equal_dict[key])):
            test_equal_dict[key][idx] = test_mapping[test_equal_dict[key][idx]]

    poison_graph_list = attack_substructure(dataset, target, source, test_graph_list, test_equal_dict,
                                            portion, substructure_portion)

    all_graph_list = [None for i in range(len(graph_list))]
    for idx, graph in zip(train_idxs, train_graph_list):
        all_graph_list[idx] = graph

    for idx, graph in zip(test_idxs, poison_graph_list):
        all_graph_list[idx] = graph

        # 填充列表c的图到combined中
    for idx, graph in zip(valid_idxs, valid_graph_list):
        all_graph_list[idx] = graph

    # 移除None元素（如果索引列表不是连续的）
    for graph in all_graph_list:
        if graph == None:
            print('error!')
            exit()

    save_graph(destination_directory, all_graph_list)
    save_idx_file(destination_directory, idxs)


def posion(source_directory, dataset, mode, target, source, portion, substructure_portion):
    equal_dict = load_equal_dict(dataset)
    destination_directory = f"../data/{dataset}/{mode}_{target.replace('/', '_')}_{source.replace('/', '_')}_{portion}_{substructure_portion}/raw"
    copy_files(source_directory, destination_directory)
    graph_list = load_graph(dataset)
    poison_graph_list = attack_substructure(dataset, target, source, graph_list, equal_dict, portion,
                                            substructure_portion)
    save_graph(destination_directory, poison_graph_list)

def features(source_directory, dataset, mode, lamda_value):
    destination_directory = f"../data/{dataset}/{mode}_{lamda_value}/raw"
    copy_files(source_directory, destination_directory)
    node_features = load_node_features(dataset)
    reference_amplitude = calculate_reference_amplitude(node_features)
    noise = add_gaussian_noise(node_features)
    noisy_node_features = reference_amplitude * noise * lamda_value + node_features
    with open(f"../data/{dataset}/{mode}_{lamda_value}/entity_embeddings.pickle", 'wb') as f:
        pickle.dump(noisy_node_features, f)

def structure(source_directory, dataset, mode, numlink):
    destination_directory = f"../data/{dataset}/{mode}_remove_{numlink}/raw"
    copy_files(source_directory, destination_directory)
    graph_list = load_graph(dataset)
    graph_list = graph_list_random_remove(graph_list, numlink)
    save_graph(destination_directory, graph_list)
