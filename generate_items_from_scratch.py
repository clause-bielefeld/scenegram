import os
import os.path as osp
import json
from random import choices, seed
import pandas as pd

from tqdm.autonotebook import tqdm
from tangram_utils import build_grid_sample, SCENE_CATEGORIES

from configuration import Config, Configuration

import argparse

seed(123)


def main(config):

    # collect available tangrams
    with open(config.data_path) as f:
        data = json.load(f)
    config.tangram_data = sorted(data.keys())
    
    # retrieve SND data
    with open(config.snd_path) as f:
        snd_values = json.load(f)

    # compute possible combinations
    combinations = []
    for i, tg in enumerate(config.tangram_data):
        for context in SCENE_CATEGORIES:
            combinations.append((i, tg, context))

    print(
        f"{len(config.tangram_data)} tangrams * {len(SCENE_CATEGORIES)} context types = {len(combinations)} combinations"
    )

    item_path = osp.join(config.data_out_dir, f"{config.split}_items.json")
    if osp.isfile(item_path) and not config.force_refresh:
        print("read items from file")
        combinations_data = pd.read_json(item_path).to_dict(orient="records")
    else:
        combinations_data = [
            {"item_id": i, "tangram_id": tangram_id, "tangram": tangram, "scene": scene, "kilogram_snd": snd_values[tangram]}
            for i, (tangram_id, tangram, scene) in enumerate(combinations)
        ]
        item_df = pd.DataFrame(combinations_data)
        # save combination df as json
        item_df.to_json(item_path, orient="records")

    # sample
    if config.sample_size > -1:
        print(f"Generating sample with {config.sample_size} items")
        # restrict combinations_data to sample
        combinations_data = choices(combinations_data, k=config.sample_size)

    # start building process
    outdir = config.samples_out_dir if config.sample_size > -1 else config.items_out_dir
    print(f"Store items in {outdir}", "\n")

    # save samples
    assert config.mode == "grid", 'only grid mode is supported'
    print(f"Building samples in {config.mode} mode")
    build_sample = build_grid_sample

    out_data = []
    for entry in tqdm(combinations_data):
        tangram_id = entry["tangram_id"]
        scene = entry["scene"]
        item_id = entry["item_id"]

        # iterate through positions for grid mode
        for pos_idx, pos_str in zip(range(4), ["tl", "tr", "bl", "br"]):
            sample_image, img_part_names = build_sample(tangram_id, scene, **vars(config), tangram_pos=pos_idx)
            item_id_str = str(item_id).rjust(3, "0")
            sample_image.save(osp.join(outdir, f"{item_id_str}_{config.mode}_{pos_str}.png"))
            out_data.append({
                **entry,
                'pos_idx': pos_idx,
                'item_configuration': img_part_names,
            })
    
    with open(osp.join(outdir, f"{config.mode}_items.json"), 'w') as f:
        json.dump(out_data, f)        

if __name__ == "__main__":

    local_config = Config()

    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", required=True, choices=["inline", "side", "grid"])
    parser.add_argument("--split", default=local_config.split)
    parser.add_argument("--data_path", default=local_config.data_path, type=osp.abspath)
    parser.add_argument("--snd_path", default=local_config.snd_path, type=osp.abspath)
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
    parser.add_argument(
        "--samples_out_dir", default=local_config.samples_out_dir, type=osp.abspath
    )
    parser.add_argument("--size", default=local_config.size)
    parser.add_argument("--sample_size", default=-1, type=int)
    parser.add_argument("--force_refresh", action="store_true")
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
