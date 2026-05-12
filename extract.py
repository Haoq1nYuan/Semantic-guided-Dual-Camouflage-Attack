import argparse
from modules.extractor import ColorExtractor


DEFAULT_IMAGE_IDX = [1, 2, 3, 4, 5]
DEFAULT_CLUSTER_CNT = 10
DEFAULT_ROUNDS = 10


def parse_args():
    parser = argparse.ArgumentParser(description='Extract dominant colors from Unity background images.')
    parser.add_argument('--image-idx', nargs='+', type=int, default=DEFAULT_IMAGE_IDX)
    parser.add_argument('--cluster-cnt', type=int, default=DEFAULT_CLUSTER_CNT)
    parser.add_argument('--rounds', type=int, default=DEFAULT_ROUNDS)
    return parser.parse_args()


def main():
    args = parse_args()
    extractor = ColorExtractor(args.image_idx)
    extractor.run_cluster(args.cluster_cnt, args.rounds)


if __name__ == '__main__':
    main()
