import argparse
import os
import os.path as osp
import json
import pandas as pd
import spacy

from PIL import Image
from tqdm import tqdm

from mllm_wrappers_chat import LLaVA, Phi, Qwen# , Molmo  # , Pixtral
from modelling_utils import location_prompt, description_prompt, few_shot_description_prompt, response_start, get_wn_lemmas, two_step_predictions


def main(args):

    cols = [
        #"item_identifyer",
        "tangram",
        "scene",
        "tangram_id",
        "item_id",
        "tangram_pos",
        "image_name",
    ]
    samples_df = pd.read_csv(args.input_data)
    samples_df = samples_df[cols].groupby("item_id").first().reset_index()

    if args.model_type == "llava":
        Model = LLaVA
    elif args.model_type == "phi":
        Model = Phi
    elif args.model_type == "qwen":
        Model = Qwen
    # elif args.model_type == "molmo":
    #     Model = Molmo
    else:
        raise NotImplementedError(f'Model type {args.model_type} is not implemented')

    model_args = dict()
    if args.model_id is not None:
        model_args["model_id"] = args.model_id
    if args.quant is not None:
        model_args["quant"] = args.quant
    if args.cache_dir is not None:
        model_args["cache_dir"] = args.cache_dir
        
    print('Building model with args:')
    print(*[f'{k}: {v}' for k, v in model_args.items()], sep='\n')
        
    model = Model(**model_args)
    args.model_id = model.model_id

    response_start = None if args.response_start == '' else args.response_start.strip()
    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "top_p": args.top_p,
        "response_start": response_start,
        "location_prompt": location_prompt,
        "description_prompt": description_prompt if not args.few_shot else few_shot_description_prompt,
        "k": args.k
    }

    # response set function
      
    nlp = spacy.load("en_core_web_sm")
    wn_lemmas = list(get_wn_lemmas())

    generate_kwargs['exclusion_words'] = ['figure', 'image', 'representation']
    generate_kwargs['possible_outputs'] = wn_lemmas
    generate_kwargs['max_tries_per_k'] = args.max_tries_per_k

    print(f"starting prediction with model {model.model_id} (quant {model.quant}) and kwargs:")
    print({k:(v if k != 'possible_outputs' else 'WordNet Lemmas') for k, v in generate_kwargs.items()})
    outputs = []

    for i, entry in tqdm(list(samples_df.iterrows())):

        if args.limit is not None and i >= args.limit:
            break

        image_path = osp.join(args.image_dir, entry.image_name)
        image = Image.open(image_path)

        location_tuple, responses, _ = two_step_predictions(
            model, image, spacy_model=nlp, 
            verbose=args.verbose, **generate_kwargs)
        
        if len(responses) == 0:
            print('no responses for entry:', entry)
            responses = [(None, None)]
      
        full_responses, labels = zip(*responses)
        location_response, location_label = location_tuple
        out_data = {
            "response": list(map(str, full_responses)),
            "location_response": location_response,
            "label": list(map(str, labels)),
            "location_label": location_label
        }

        out_entry = {
            **entry,
            **out_data
        }

        if args.verbose:
            print(out_entry)

        outputs.append(out_entry)

    if not osp.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    out_data = {
        "args": vars(args),
        "generate_kwargs": generate_kwargs,
        "data": outputs,
    }
    model_name = model.model_id.split("/")[-1].replace(".", "-")
    out_path = osp.join(
        args.output_dir, f"predictions_{model_name}_q:{model.quant}_p:{str(args.top_p).replace('.', '-')}_twostep{'_fewshot' if args.few_shot else ''}.json"
    )

    print(f"Writing Results to {out_path}")
    with open(out_path, "w") as f:
        json.dump(out_data, f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_type", default="llava", type=str.lower)
    parser.add_argument("--quant", default=None, choices=["4bit", "8bit", None])
    parser.add_argument("--model_id", default=None)

    parser.add_argument("--max_new_tokens", default=100, type=int)
    parser.add_argument("--k", default=10, type=int)
    parser.add_argument("--top_p", default=0.5, type=float)
    parser.add_argument("--response_start", default=response_start)
    parser.add_argument("--max_tries_per_k", default=10, type=int)
    parser.add_argument("--few_shot", action="store_true")

    parser.add_argument("--limit", default=None, type=int)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument(
        "--input_data", default="../collected_data/raw_collected_data.csv", type=osp.abspath
    )
    parser.add_argument(
        "--image_dir", default="../generated_items/", type=osp.abspath
    )
    parser.add_argument(
        "--output_dir", default="../predicted_data/model_predictions/", type=osp.abspath
    )
    parser.add_argument("--cache_dir")

    args = parser.parse_args()

    print("Running with Arguments:")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    if not osp.isdir(args.output_dir):
        print(f"make output directory {args.output_dir}")
        os.makedirs(args.output_dir)

    main(args)
