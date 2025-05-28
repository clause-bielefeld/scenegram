import argparse
import os
from mllm_wrappers_chat import LLaVA, Phi, Pixtral, Molmo, Qwen

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def main(args):
    if args.model_id.startswith('llava-hf'):
        model = LLaVA(args.model_id, cache_dir=args.cache_dir)
    elif args.model_id.startswith('microsoft'):
        model = Phi(args.model_id, cache_dir=args.cache_dir)
    elif args.model_id.startswith('Qwen'):
        model = Qwen(args.model_id, cache_dir=args.cache_dir)

    print('done!')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_id', help='Huggingface model ID')
    parser.add_argument('--cache_dir', help='Cache dir', default=os.path.join(FILE_PATH, 'model_weights'))
    args = parser.parse_args()

    print(vars(args))
    main(args)