"""Predict class labels using a full SGMoE checkpoint."""
import argparse
from sgmoe.inference import predict
from sgmoe.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--config', help='Optional local paths after moving a trained model')
    args = parser.parse_args()
    predict(args.checkpoint, args.input, args.output, args.device,
            load_config(args.config) if args.config else None)


if __name__ == '__main__':
    main()
