import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train DGIB',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # =============Global config=============
    parser.add_argument("--dataset", type=str, default='IM-1', help="Choose dataset name")
    parser.add_argument("--backbone", type=str, default='GIN', help='backbone model used')
    parser.add_argument("--cuda", type=int, default=0, help='cuda device id, -1 for cpu')
    parser.add_argument("--mode", type=str, default='evasion', help="Choose attack type or simply use raw data")
    parser.add_argument("--isHetero", type=bool, default='False', help="Choose whether graph is heterogeneous")
    parser.add_argument("--data_dir", type=str, default='../data', help="Set data directory")
    parser.add_argument("--num_seeds", type=int, default=2, help='Set number of seeds')

    parser.add_argument("--number", type=int, default=5, help="Choose operated links")
    parser.add_argument("--operation_type", type=str, default='add', help="Choose operated types")
    parser.add_argument("--lamda", type=float, default=0.5, help="Choose attack feature degree")
    parser.add_argument("--class1", type=str, default='_var_www_html_uploads_input ', help="Choose target class")
    parser.add_argument("--class2", type=str, default='_var_www_html_uploads_poc1 ', help="Choose source class")
    parser.add_argument("--portion", type=float, default=0.25, help="Choose attack data portion")
    parser.add_argument("--structuresize", type=float, default=0.25, help="Choose attack graph structure size")

    args = parser.parse_args()

    return args