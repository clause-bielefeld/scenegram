import os
import os.path as osp
import json
from random import choices, seed
import pandas as pd

from tqdm.autonotebook import tqdm
from tangram_utils import build_grid_sample_from_template, SCENE_CATEGORIES

from configuration import Config, Configuration

import argparse

seed(123)


def main(config):

    # collect available tangrams
    with open(config.data_path) as f:
        data = json.load(f)
    config.tangram_data = sorted(data.keys())

    # read template file
    item_template_file = osp.join(args.items_out_dir, f'{args.mode}_items.json')
    item_templates = pd.read_json(item_template_file)

    item_path = osp.join(config.data_out_dir, f"{config.split}_items.json")
    print("read items from file")
    combinations_data = pd.read_json(item_path).to_dict(orient="records")

    print(
        f"{len(config.tangram_data)} tangrams * {len(SCENE_CATEGORIES)} context types = {len(combinations_data)} combinations"
    )

    # start building process
    outdir = config.items_out_dir
    print(f"Store items in {outdir}", "\n")

    # save samples
    assert config.mode == "grid", 'only grid mode is supported'
    print(f"Building samples in {config.mode} mode")
    build_sample = build_grid_sample_from_template

    # iterate through possible combinations
    for entry in tqdm(combinations_data):
        tangram_id = entry["tangram_id"]
        item_id = entry["item_id"]
        # retrieve configuration template for current item
        template = item_templates.loc[item_templates.item_id == item_id]
        for _, row in template.iterrows():
            item_configuration = row.item_configuration
            # build image
            sample_image = build_sample(
                tangram_id, item_configuration=item_configuration, **vars(config))
            item_id_str = str(item_id).rjust(3, "0")
            pos_str = ["tl", "tr", "bl", "br"][row.pos_idx]
            # save image
            sample_image.save(osp.join(outdir, f"{item_id_str}_{config.mode}_{pos_str}.png"))    

if __name__ == "__main__":

    local_config = Config()

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default=local_config.mode, choices=["inline", "side", "grid"])
    parser.add_argument("--split", default=local_config.split)
    parser.add_argument("--data_path", default=local_config.data_path, type=osp.abspath)
    parser.add_argument(
        "--tangram_dir", default=local_config.tangram_dir, type=osp.abspath
    )
    parser.add_argument(
        "--context_dir", default=local_config.context_dir, type=osp.abspath
    )
    parser.add_argument(
        "--data_out_dir", default=local_config.data_out_dir, type=osp.abspath
    )
    parser.add_argument(
        "--items_out_dir", default=local_config.items_out_dir, type=osp.abspath
    )
    parser.add_argument("--size", default=local_config.size)
    args = parser.parse_args()

    # merge args and local config
    config_dict = vars(local_config)
    config_dict.update(vars(args))
    config = Configuration(**config_dict)

    print(f"Settings:")
    print(config, "\n")

    if not osp.isdir(config.data_out_dir):
        os.mkdir(config.data_out_dir)
    if not osp.isdir(config.samples_out_dir):
        os.mkdir(config.samples_out_dir)
    if not osp.isdir(config.items_out_dir):
        os.mkdir(config.items_out_dir)

    main(config)
