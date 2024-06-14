from parse_args import args
from utils import *
import pickle
if __name__ == '__main__':
    attacktype = 'feature'
    source_directory = f"../data/{args.dataset}/raw"
    for lamda_value in [0.5, 1.0, 1.5]:
        destination_directory = f"../data/{args.dataset}/{attacktype}_{lamda_value}"
        copy_files(source_directory, destination_directory)
        node_features = load_node_features(args.dataset)
        reference_amplitude = calculate_reference_amplitude(node_features)
        noise = add_gaussian_noise(node_features)
        noisy_node_features = reference_amplitude * noise * lamda_value + node_features
        with open(f"../data/{args.dataset}/{attacktype}_{lamda_value}/entity_embeddings.pickle", 'wb') as f:
            pickle.dump(noisy_node_features, f)
