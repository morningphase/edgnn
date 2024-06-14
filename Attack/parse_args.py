import argparse
parser = argparse.ArgumentParser(description='Tools For Arguments',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, default='Apache-1', help="Choose dataset name")
parser.add_argument("--num_links", type=int, default=5, help="Choose operated links")
parser.add_argument("--operation_type", type=str, default='add', help="Choose operated types")
parser.add_argument("--lamda", type=float, default=0.5, help="Choose attack_degree")

args = parser.parse_args()