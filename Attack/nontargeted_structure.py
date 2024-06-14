from parse_args import args
from utils import *

if __name__ == '__main__':
    attacktype = 'structure'
    source_directory = f"../data/{args.dataset}/common/raw"
    for numlink in [3, 6, 9]:
        destination_directory = f"../data/{args.dataset}/{attacktype}_add_{numlink}"
        copy_files(source_directory, destination_directory)
        graph_list = load_graph(args.dataset)
        graph_list = graph_list_random_add(graph_list, numlink)
        save_graph(destination_directory, graph_list)
        print('Add Done')

        destination_directory = f"../data/{args.dataset}/{attacktype}_remove_{numlink}"
        copy_files(source_directory, destination_directory)
        graph_list = load_graph(args.dataset)
        graph_list = graph_list_random_remove(graph_list, numlink)
        save_graph(destination_directory, graph_list)
        print('Remove Done')
