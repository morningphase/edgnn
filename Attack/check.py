import networkx as nx

from utils import *
from parse_args import args
import sys
from collections import defaultdict
import matplotlib.pyplot as plt


def find_unique_keys_and_relations(nested_dict):
    unique_key_relations = defaultdict(list)

    for first_level_key, second_level_dict in nested_dict.items():
        for second_level_key, value in second_level_dict.items():
            if second_level_key not in unique_key_relations or len(unique_key_relations[second_level_key]) == 1:
                unique_key_relations[second_level_key].append(first_level_key)
            else:
                break

    unique_keys = {key: value for key, value in unique_key_relations.items() if len(value) == 1}

    return unique_keys

if __name__ == '__main__':
    graph_list = load_graph(args.dataset)
    equal_dict = load_equal_dict(args.dataset)
    print_file = open("output.txt", "w")
    sys.stdout = print_file

    edge_triplet = get_edge_triplet(graph_list)
    edge_triplet_set_list = list()
    for edge_list in edge_triplet:
        tmp_set = set()
        for edge in edge_list:
            tmp_set.add(edge)
        edge_triplet_set_list.append(tmp_set)

    all_dict_cnt = dict()

    for k in equal_dict.keys():
        print(k)
        if k not in all_dict_cnt.keys():
            all_dict_cnt[k] = dict()
        check_num = dict()
        print("-------------------------")
        for idx, i in enumerate(equal_dict[k]):
            print("-------------------------")
            print(i)
            print(edge_triplet[i])
            # # 创建一个有向图
            # G = nx.MultiDiGraph()
            #
            # # 添加边到图中
            # for src, dst, edge_type in edge_triplet[i]:
            #     G.add_edge(src, dst, label=edge_type)
            #
            # # 绘制图
            # plt.figure(figsize=(12, 12))
            # pos = nx.spring_layout(G, k=0.15, iterations=20)
            # # edge_labels = nx.get_edge_attributes(G, 'label')
            #
            # # 绘制节点和边
            # nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold',
            #         arrows=True)
            #
            # # 自定义绘制多重边的标签
            # edge_labels = nx.get_edge_attributes(G, 'label')
            # for edge in G.edges(data=True):
            #     src, dst, data = edge
            #     x1, y1 = pos[src]
            #     x2, y2 = pos[dst]
            #     plt.text((x1 + x2) / 2, (y1 + y2) / 2, s=data['label'], fontsize=8, color='red')
            #
            # # 保存图像
            # plt.savefig(f'fig_{i}.png')
            #
            # if idx < len(equal_dict[k])-1:
            #     print(nx.vf2pp_is_isomorphic(graph_list[equal_dict[k][idx]], graph_list[equal_dict[k][idx+1]], 'name'))
            # print("-------------------------")
            for triplet in edge_triplet_set_list[i]:
                if triplet not in check_num.keys():
                    check_num[triplet] = 1
                else:
                    check_num[triplet] = 1 + check_num[triplet]
        # 使用sorted函数和lambda表达式按照value从大到小排序
        sorted_items = sorted(check_num.items(), key=lambda x: x[1], reverse=True)
        # 输出排序后的结果
        print(f'All Relation Number in {k}: {len(check_num.keys())}')
        for key, value in sorted_items:
            if value > len(equal_dict[k]) * 0.75:
                all_dict_cnt[k][key] = value
                print(f"{key}: {value}")
        print(' ')
    print("-------------------------")
    # 调用函数并打印结果
    unique_keys_and_relations = find_unique_keys_and_relations(all_dict_cnt)
    for second_level_key, first_level_keys in unique_keys_and_relations.items():
        print(f"First-level key(s): {first_level_keys} contains unique second-level key '{second_level_key}'")