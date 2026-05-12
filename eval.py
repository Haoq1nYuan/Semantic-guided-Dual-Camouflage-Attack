import argparse
import yaml
from modules.evaluator import DualDimensionalEvaluator


DEFAULT_CONFIG = 'config/config_eval.yaml'
DEFAULT_MODE = 'adversarial'
DEFAULT_MODEL = 'yolov5s'


def SDCA_eval(config, mode, model, is_save):
    evaluator = DualDimensionalEvaluator(mode, config)
    if mode == 'adversarial':
        evaluator.adv_evaluate(model=model, is_save=is_save)
    elif mode == 'natural':
        evaluator.nat_evaluate(is_save=is_save)
    else:
        raise ValueError(f"Unsupported evaluation mode: {mode}")


def parse_args():
    parser = argparse.ArgumentParser(description='Run SDCA evaluation.')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='Path to the evaluation YAML config.')
    parser.add_argument('--mode', choices=['adversarial', 'natural'], default=DEFAULT_MODE)
    parser.add_argument(
        '--model',
        default=DEFAULT_MODEL,
        help='Detector for adversarial evaluation: yolov5n/s/m/l/x, yolov8n/s/m/l, EfDetd0/1/2, FrRCNN, or DETR.',
    )
    parser.add_argument('--save', action='store_true', help='Save rendered evaluation images.')
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    SDCA_eval(config, args.mode, args.model, args.save)


if __name__ == "__main__":
    main()
