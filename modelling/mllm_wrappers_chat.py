import torch
import warnings
from transformers import GenerationConfig, BitsAndBytesConfig


class MLLMWrapper:
    """Wrapper class with common methods"""
    
    def add_start_to_inputs(self, inputs, idx_offset=0, add_str="The tangram depicts"):

        # prepare ids and att masks
        answer_start = self.processor.tokenizer(add_str, return_tensors="pt")
        answer_start_ids = answer_start["input_ids"][:, idx_offset:].to(self.model.device)
        answer_start_att = answer_start["attention_mask"][:, idx_offset:].to(self.model.device)

        # concat previous and additional ids and att masks
        extended_input_ids = torch.concat([inputs["input_ids"], answer_start_ids], axis=1)

        extended_input_atts = torch.concat(
            [inputs["attention_mask"], answer_start_att], axis=1
        )

        # re-define inputs and return
        inputs["input_ids"] = extended_input_ids
        inputs["attention_mask"] = extended_input_atts

        return inputs

    @staticmethod
    def prune_generated_tokens_to_response(generated_ids, split_id):
        selection_start = (generated_ids == split_id).nonzero().max().item()
        response_ids = generated_ids[selection_start:]
        return response_ids


class LLaVA(MLLMWrapper):
    """
    Wrapper for LLaVA Models
    """

    # Documentation:
    # https://huggingface.co/docs/transformers/model_doc/llava_next

    def __init__(
        self, model_id="llava-hf/llava-v1.6-mistral-7b-hf", quant=None, **kwargs
    ):
        """
        Constructor method

        Args:
            model_id (str, optional): huggingface model ID. Defaults to "llava-hf/llava-v1.6-mistral-7b-hf".
            quant (str or NoneType, optional): Quantization setting. Defaults to None.
            kwargs: Further parameters, e.g. cache_dir.
        """

        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

        # set up quantization
        if quant is not None:
            if quant == "4bit":
                self.quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
            elif quant == "8bit":
                self.quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
        else:
            self.quantization_config = None

        print(f"building {self.__class__.__name__} model...")

        # set up model and processor
        self.processor = LlavaNextProcessor.from_pretrained(
            model_id, cache_dir=kwargs.get("cache_dir", None),device_map="auto"
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            quantization_config=self.quantization_config,
            cache_dir=kwargs.get("cache_dir", None),
            device_map="auto",
        )
        self.model.generation_config.pad_token_id = (
            self.processor.tokenizer.pad_token_id
        )

        self.model_id = model_id
        
        self.model_size = None
        for possible_size in ['7b', '13b', '34b', '72b']:
            if f'-{possible_size}-' in self.model_id.lower():
                self.model_size = possible_size
        assert self.model_size is not None
        
        self.quant = quant
        self.device = self.model.device
        
    def prune_output_sequence_to_response(self, output_sequence):
        if 'vicuna' in self.model_id:
            sep = 'ASSISTANT: '
        elif self.model_size == '7b':
            sep = '[/INST]'
        else:
            sep = 'assistant\n'
        return output_sequence.split(sep)[-1].strip()
            

    def generate(self, image, prompt, prune_output_to_response=True, **generate_kwargs):
        """
        Generate response for a simple prompt with a single input image.

        Args:
            prompt (str): The prompt given to the model.
            image (PIL.Image): The input image
            prune_output_to_response (bool, optional): Prune the output to the model response (excluding the input prompt).
                Defaults to True.
            generate_kwargs: Further arguments for the huggingface generate API

        Returns:
            str: The model response.
        """


        # create prompt in the right format
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # tokenize and make torch tensor
        inputs = self.processor(image, prompt, return_tensors="pt").to(
            self.model.device
        )
        
        # if specified: force the model to start with a given partial response
        if generate_kwargs.get("response_start", None) is not None:
            response_start = generate_kwargs.pop("response_start")
            inputs = self.add_start_to_inputs(inputs, add_str=response_start)

        # make GenerationConfig from generate_kwargs and predict response with model
        generation_config = GenerationConfig(**generate_kwargs)
        output = self.model.generate(**inputs, generation_config=generation_config)

        # transform output ids to string
        response_sentence = self.processor.decode(output[0], skip_special_tokens=True)

        # prune output to model response
        if prune_output_to_response:
            response_sentence = self.prune_output_sequence_to_response(response_sentence)
            
        return response_sentence

    def generate_from_messages(
        self, images, messages, prune_output_to_response=True, **generate_kwargs
    ):
        """
        Generate response given a chat history and (possibly) multiple images.

        Args:
            messages (list[dict]): The chat history with image placeholders.
            images (list[PIL.Image]): List with one or multiple input images.
            prune_output_to_response (bool, optional): Prune the output to the model response (excluding the input prompt).
                Defaults to True.
            generate_kwargs: Further arguments for the huggingface generate API

        Returns:
            str: The model response.
        """

        # transform input chat to prompt string
        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        # tokenize and make torch tensor
        inputs = self.processor(
            images=images, text=prompt, padding=True, return_tensors="pt"
        ).to(self.model.device)
        
        # if specified: force the model to start with a given partial response
        if generate_kwargs.get("response_start", None) is not None:
            response_start = generate_kwargs.pop("response_start")
            inputs = self.add_start_to_inputs(inputs, add_str=response_start)
        
        # make GenerationConfig from generate_kwargs and predict response with model
        generation_config = GenerationConfig(**generate_kwargs)
        output = self.model.generate(**inputs, generation_config=generation_config)

        # transform output ids to string
        response_sentence = self.processor.decode(output[0], skip_special_tokens=True)
        # prune output to model response
        if prune_output_to_response:
            response_sentence = self.prune_output_sequence_to_response(response_sentence)

        return response_sentence

    @staticmethod    
    def init_chat(prompt, *args):
        return [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ]}]

    @staticmethod
    def add_to_chat(chat, role, text):
        chat.append({"role": role, "content": [{"type": "text", "text": text}]})
        return chat


class Phi(MLLMWrapper):

    # Documentation:
    # https://huggingface.co/microsoft/Phi-3.5-vision-instruct

    def __init__(
        self, model_id="microsoft/Phi-3.5-vision-instruct", quant=None, **kwargs
    ):

        from transformers import AutoModelForCausalLM, AutoProcessor

        if quant is not None:
            warnings.warn(
                f"Quantization not implemented in {self.__class__.__name__} wrapper"
            )

        print(f"building {self.__class__.__name__} model...")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="eager",  #  or 'flash_attention_2' if flash_attn is installed
            cache_dir=kwargs.get('cache_dir', None)
        )

        # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_crops=4,
            cache_dir=kwargs.get('cache_dir', None)
        )

        self.model_id = model_id
        self.quant = quant
        self.device = self.model.device

    def generate(self, image, prompt, prune_output_to_response=True, **generate_kwargs):

        placeholder = f"<|image_1|>\n"

        messages = [
            {"role": "user", "content": placeholder + prompt},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(prompt, [image], return_tensors="pt").to(
            self.model.device
        )

        if generate_kwargs.get("response_start", None) is not None:
            response_start = generate_kwargs.pop("response_start")
            inputs = self.add_start_to_inputs(inputs, add_str=response_start)

        generation_config = GenerationConfig(**generate_kwargs)
        response_ids = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            generation_config=generation_config
        )[0]
        
        if prune_output_to_response:
            split_id = self.processor.tokenizer.vocab['<|assistant|>']
            response_ids = self.prune_generated_tokens_to_response(response_ids, split_id)
        else:
            response_ids = response_ids[response_ids >= 0]
        
        response = self.processor.decode(response_ids, skip_special_tokens=True)
        
        return response
    
    def generate_from_messages(self, images, messages, prune_output_to_response=True, **generate_kwargs):
        """
        Generate response given a chat history and (possibly) multiple images.

        Args:
            images (list[PIL.Image]): List with one or multiple input images.
            messages (list[dict]): The chat history with image placeholders.
            prune_output_to_response (bool, optional): Prune the output to the model response (excluding the input prompt). 
                Defaults to True.
            generate_kwargs: Further arguments for the huggingface generate API

        Returns:
            str: The model response.
        """
        
        # transform input chat to prompt string
        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # tokenize and make torch tensor
        inputs = self.processor(prompt, images, return_tensors="pt").to(
            self.model.device
        )
        
        if generate_kwargs.get("response_start", None) is not None:
            response_start = generate_kwargs.pop("response_start")
            inputs = self.add_start_to_inputs(inputs, add_str=response_start)

        # make GenerationConfig from generate_kwargs and predict response with model
        generation_config = GenerationConfig(**generate_kwargs)
        response_ids = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            generation_config=generation_config
        )[0]

        if prune_output_to_response:
            split_id = self.processor.tokenizer.vocab['<|assistant|>']
            response_ids = self.prune_generated_tokens_to_response(response_ids, split_id)
        else:
            response_ids = response_ids[response_ids >= 0]
                    
        # transform output ids to string
        response = self.processor.decode(
            response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        ).strip()

        return response

    @staticmethod    
    def init_chat(prompt, *args):
        return [{
            "role": "user", 
            "content": f"<|image_1|>\n{prompt}"
            }]

    @staticmethod        
    def add_to_chat(chat, role, text):
        chat.append({"role": role, "content": text})
        return chat


class Molmo(MLLMWrapper):

    # Documentation:
    # https://huggingface.co/allenai/Molmo-7B-D-0924

    def __init__(self, model_id="allenai/Molmo-7B-D-0924", quant=None, **kwargs):

        from transformers import AutoModelForCausalLM, AutoProcessor

        if quant is not None:
            if quant == "4bit":
                self.quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
            elif quant == "8bit":
                self.quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
        else:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=False, load_in_8bit=False
            )

        print(f"building {self.__class__.__name__} model...")

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=kwargs.get('cache_dir', None)
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="auto",
            quantization_config=self.quantization_config,
            cache_dir=kwargs.get('cache_dir', None)
        )

        self.model.generation_config.pad_token_id = (
            self.processor.tokenizer.pad_token_id
        )

        self.model_id = model_id
        self.quant = quant
        self.device = self.model.device
        
    def add_start_to_inputs(self, inputs, add_str="The tangram depicts"):

        # prepare ids and att masks
        add_str = ' ' + add_str.lstrip()  # has to start with space
        answer_start = self.processor.tokenizer(add_str, return_tensors="pt")
        answer_start_ids = answer_start["input_ids"].to(self.model.device)

        extended_input_ids = torch.concat([inputs["input_ids"], answer_start_ids], axis=1)

        # re-define inputs and return
        inputs["input_ids"] = extended_input_ids

        return inputs

    def generate(self, image, prompt, prune_output_to_response=True, **generate_kwargs):

        inputs = self.processor.process(images=[image], text=prompt)
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        if generate_kwargs.get("response_start", None) is not None:
            response_start = generate_kwargs.pop("response_start")
            inputs = self.add_start_to_inputs(inputs, add_str=response_start)  # fails since processor can't be called

        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(**generate_kwargs, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer,
        )

        response_ids = output[0, :]

        if prune_output_to_response:

            split_id = self.processor.tokenizer(' Assistant')['input_ids'][0]
            response_ids = self.prune_generated_tokens_to_response(response_ids, split_id)
            
            assert response_ids[:2].tolist() == [21388, 25]
            response_ids = response_ids[2:]

        generated_text = self.processor.tokenizer.decode(
            response_ids, skip_special_tokens=True
        ).strip()

        return generated_text


class Pixtral(MLLMWrapper):
    
    # BUG sampling does not work

    # Documentation:
    # https://huggingface.co/mistral-community/pixtral-12b

    def __init__(self, model_id="mistral-community/pixtral-12b", quant=None, **kwargs):

        from transformers import AutoProcessor, LlavaForConditionalGeneration

        if quant is not None:
            if quant == "4bit":
                self.quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                )
            elif quant == "8bit":
                self.quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
        else:
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=False, load_in_8bit=False
            )

        print(f"building {self.__class__.__name__} model...")

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,  # 'mistral-community/pixtral-12b'
            quantization_config=self.quantization_config,
            cache_dir=kwargs.get('cache_dir', None)
        )
        self.processor = AutoProcessor.from_pretrained(
            model_id,  # 'mistral-community/pixtral-12b'
            cache_dir=kwargs.get('cache_dir', None)
        )

        self.model_id = model_id
        self.quant = quant
        self.device = self.model.device

    def generate(self, image, prompt, prune_output_to_response=True, **generate_kwargs):

        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "content": prompt}],
            }
        ]

        text = self.processor.apply_chat_template(messages)
        inputs = self.processor(text=text, images=[image], return_tensors="pt").to(
            self.model.device
        )

        if generate_kwargs.get("response_start", None) is not None:
            response_start = generate_kwargs.pop("response_start")
            inputs = self.add_start_to_inputs(inputs, add_str=response_start)

        generation_config = GenerationConfig(**generate_kwargs)
        response_ids = self.model.generate(**inputs, generation_config=generation_config)[0]
        
        if prune_output_to_response:

            split_id = self.processor.tokenizer.encode('[/INST]')[0]
            response_ids = self.prune_generated_tokens_to_response(response_ids, split_id)
            response_ids = response_ids[1:]
                    
        response = self.processor.decode(
            response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return response


class Qwen(MLLMWrapper):

    # Documentation:
    # https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
    # -> https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
    # -> https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct

    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct", quant=None, **kwargs):

        from transformers import (
            Qwen2VLForConditionalGeneration,
            AutoTokenizer,
            AutoProcessor,
        )

        if quant is not None:
            warnings.warn(
                f"Quantization not implemented in {self.__class__.__name__} wrapper"
            )

        print(f"building {self.__class__.__name__} model...")

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",  # or torch.bfloat16
            #  attn_implementation="flash_attention_2",
            device_map="auto",
            cache_dir=kwargs.get('cache_dir', None)
        )

        # default processer
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=kwargs.get('cache_dir', None))

        self.model_id = model_id
        self.quant = quant
        self.device = self.model.device
        

    def generate(self, image, prompt, prune_output_to_response=True, **generate_kwargs):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        ).to(self.model.device)
        
        if generate_kwargs.get("response_start", None) is not None:
            response_start = generate_kwargs.pop("response_start")
            inputs = self.add_start_to_inputs(inputs, add_str=response_start)

        # Inference: Generation of the output
        generation_config = GenerationConfig(**generate_kwargs)
        response_ids = self.model.generate(**inputs, generation_config=generation_config)[0]
        
        if prune_output_to_response:

            split_id = self.processor.tokenizer.encode('<|im_start|>')[0]
            response_ids = self.prune_generated_tokens_to_response(response_ids, split_id)
            
            assert response_ids[:3].tolist() == [split_id, 77091, 198], response_ids[:3].tolist()
            response_ids = response_ids[3:]
        
        response = self.processor.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return response
    
    
    def generate_from_messages(self, images, messages, prune_output_to_response=True, **generate_kwargs):
        """
        Generate response given a chat history and (possibly) multiple images.

        Args:
            images (list[PIL.Image]): List with one or multiple input images.
            messages (list[dict]): The chat history with image placeholders.
            prune_output_to_response (bool, optional): Prune the output to the model response (excluding the input prompt). 
                Defaults to True.
            generate_kwargs: Further arguments for the huggingface generate API

        Returns:
            str: The model response.
        """
        
        # Preparation for inference
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[prompt], images=images, padding=True, return_tensors="pt"
        ).to(self.model.device)
        
        if generate_kwargs.get("response_start", None) is not None:
            response_start = generate_kwargs.pop("response_start")
            inputs = self.add_start_to_inputs(inputs, add_str=response_start)

        # Inference: Generation of the output
        generation_config = GenerationConfig(**generate_kwargs)
        response_ids = self.model.generate(**inputs, generation_config=generation_config)[0]
        
        if prune_output_to_response:

            split_id = self.processor.tokenizer.encode('<|im_start|>')[0]
            response_ids = self.prune_generated_tokens_to_response(response_ids, split_id)
            
            assert response_ids[:3].tolist() == [split_id, 77091, 198], response_ids[:3].tolist()
            response_ids = response_ids[3:]
        
        response = self.processor.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return response


    @staticmethod    
    def init_chat(prompt, image):
        return [{
            "role": "user",
            "content": [
                {"type": "image", "image": image,},
                {"type": "text", "text": prompt},
        ]}]

    @staticmethod        
    def add_to_chat(chat, role, text):
        chat.append({"role": role, "content": [{"type": "text", "text": text}]})
        return chat