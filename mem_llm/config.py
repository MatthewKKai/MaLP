import argparse

def get_opt():
    parser = argparse.ArgumentParser()

    # training

    # DDP
    parser.add_argument("--local_rank", type = int, default = 0)

    return parser.parse_args()