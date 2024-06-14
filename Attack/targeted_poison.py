from parse_args import args
from utils import *
import random

if __name__ == '__main__':
    attacktype = 'poison'
    portions = [0.5, 0.7]  # poison all data portion
    source_directory = f"../data/{args.dataset}/common/raw"
    equal_dict = load_equal_dict(args.dataset)
    substructure_portions = [0.5] # poison structure size
    key_pair_list = get_key_combinations(equal_dict)

    for portion in tqdm(portions):
        for substructure_portion in substructure_portions:
            for key_pair in key_pair_list:
                destination_directory = f"../data/{args.dataset}/{attacktype}_{key_pair[0].replace('/', '_')}_{key_pair[1].replace('/', '_')}_{portion}_{substructure_portion}"
                copy_files(source_directory, destination_directory)
                graph_list = load_graph(args.dataset)
                poison_graph_list = attack_substructure(args.dataset, key_pair[0], key_pair[1], graph_list, equal_dict, portion, substructure_portion)
                save_graph(destination_directory, poison_graph_list)
                print('Done one experiment')

    for portion in tqdm(portions):
        for substructure_portion in substructure_portions:
            destination_directory = f"../data/{args.dataset}/{attacktype}_all_all_{portion}_{substructure_portion}"
            copy_files(source_directory, destination_directory)
            graph_list = load_graph(args.dataset)
            poison_graph_list = attack_substructure(args.dataset, 'all', 'all', graph_list, equal_dict,
                                                    portion, substructure_portion)
            save_graph(destination_directory, poison_graph_list)
            print('Done one experiment')