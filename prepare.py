"""Train semantic features and construct the full SGMoE expert assignment."""
import argparse
from sgmoe.config import load_config
from sgmoe.preparation import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    prepare(load_config(args.config), args.device, args.resume, args.dry_run)


if __name__ == '__main__':
    main()
