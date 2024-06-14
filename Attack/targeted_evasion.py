from parse_args import args
from utils import *
import numpy as np

if __name__ == '__main__':
    attacktype = 'evasion'
    portions = [0.125, 0.25, 0.5]  # poison all data portion
    source_directory = f"../data/{args.dataset}/raw"
    equal_dict = load_equal_dict(args.dataset)
    substructure_portions = [0.1, 0.125, 0.2, 0.25] # poison structure size
    key_pair_list = get_key_combinations(equal_dict)
    train_split = 0.8
    valid_split = 0.1
    test_split = 0.1

    for portion in tqdm(portions):
        for substructure_portion in substructure_portions:
            destination_directory = f"../data/{args.dataset}/{attacktype}_all_all_{portion}_{substructure_portion}"
            copy_files(source_directory, destination_directory)
            graph_list = load_graph(args.dataset)
            equal_dict = load_equal_dict(args.dataset)

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
            train = set(train_idxs)
            test = set(test_idxs)
            valid = set(valid_idxs)
            union_set = test | train | valid
            print(len(union_set))
            print(type(train_idxs[1]))

            idxs = {'train': np.array(train_idxs), 'valid': np.array(valid_idxs), 'test': np.array(test_idxs)}
            print(idxs)
            print(type(idxs))

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


            #idxs = list(range(len(graph_list)))
            poison_graph_list = attack_substructure(args.dataset, 'all', 'all', test_graph_list, test_equal_dict,
                                                    portion, substructure_portion)
            all_graph_list = [None for i in range(len(graph_list))]
            for idx, graph in zip(train_idxs, train_graph_list):
                all_graph_list[idx] = graph

            for idx, graph in zip(test_idxs, test_graph_list):
                all_graph_list[idx] = graph

                # 填充列表c的图到combined中
            for idx, graph in zip(valid_idxs, valid_graph_list):
                all_graph_list[idx] = graph

            # 移除None元素（如果索引列表不是连续的）
            for idx, graph in enumerate(all_graph_list):
                if graph == None:
                    print('error!')
                    print(idx)
                    exit()

            save_graph(destination_directory, all_graph_list)
            save_idx_file(destination_directory, idxs)

            # save_test_graph(destination_directory, poison_graph_list)
            # save_train_graph(destination_directory, train_graph_list)
            # save_train_equal_dict(destination_directory, train_equal_dict)
            # save_test_equal_dict(destination_directory, test_equal_dict)
            print('Done one experiment')


    for portion in tqdm(portions):
        for substructure_portion in substructure_portions:
            for key_pair in key_pair_list:
                destination_directory = f"../data/{args.dataset}/{attacktype}_{key_pair[0].replace('/', '_')}_{key_pair[1].replace('/', '_')}_{portion}_{substructure_portion}"
                copy_files(source_directory, destination_directory)
                graph_list = load_graph(args.dataset)
                equal_dict = load_equal_dict(args.dataset)

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

                poison_graph_list = attack_substructure(args.dataset, key_pair[0], key_pair[1], test_graph_list, test_equal_dict, portion, substructure_portion)

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
                print('Done one experiment')