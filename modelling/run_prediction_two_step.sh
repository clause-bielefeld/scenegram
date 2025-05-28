for model_id in llava-hf/llava-v1.6-34b-hf llava-hf/llava-v1.6-vicuna-13b-hf llava-hf/llava-v1.6-vicuna-7b-hf llava-hf/llava-next-72b-hf

do
  python predict_two_step.py --model_type llava --model_id $model_id --cache_dir ./model_weights --top_p 0.5 --max_new_tokens 100 --k 10 --max_tries_per_k 10 --few_shot
done
