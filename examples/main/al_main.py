import os
import yaml
import argparse
from al_mlp.run_al import active_learning


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-yml", required=True, help="Path to the config file")
    return parser


def main(args):
    config_yml = args.config_yml
    basedir = config_yml[: config_yml.rindex("/") + 1]
    os.chdir(basedir)

    config = yaml.safe_load(open(config_yml, "r"))
    active_learning(config)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
