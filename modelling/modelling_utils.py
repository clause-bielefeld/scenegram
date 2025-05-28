import re
from tqdm import tqdm
from nltk.corpus import wordnet as wn
from extraction_utils import extract_noun_from_response, get_location_from_model_response

inline_prompt = "There is a tangram in the middle of this image within the grey square. What does it look like? Give your answer in the form: The tangram depicts a #object"
side_prompt = "There is a tangram on the right side of the image. What does it look like? Give your answer in the form: The tangram depicts a #object"
grid_prompt = "In this 2 by 2 grid, exactly one tile contains a tangram figure. Describe what the tangram looks like. Ignore the other tiles. Keep your answer short and concise. Give your answer in the form: The tangram depicts a _."

location_prompt = "In this 2 by 2 grid, exactly one tile contains a tangram figure. In which grid cell is it? Pick your response from the following options: Top left, top right, bottom left, bottom right."
description_prompt = 'Describe what this tangram looks like. Ignore the other tiles. Keep your answer short and concise. Give your answer in the form: The tangram depicts a _.'
few_shot_description_prompt = 'Describe what this tangram looks like. Ignore the other tiles. Keep your answer short and concise, by providing answers like "table", "turtle", or "bathtub". Give your answer in the form: The tangram depicts a _.'

extraction_query = r"The tangram depicts an? ([\w# ]*)\.?"
response_start = "The tangram depicts"


def extract_label_from_response(response_sentence, query):

    match = re.search(query, response_sentence, re.IGNORECASE)

    if match:
        return match.group(1)
    else:
        return ""


def get_response_set(
    model, image, prompt, k, show_progress_bar=False, **generate_kwargs
):

    assert (
        generate_kwargs.get("do_sample", False) is True
    ), "`do_sample` has to be set to True"
    assert (
        generate_kwargs.get("top_k", 1) > 1 or generate_kwargs.get("top_p", 1.0) < 1.0
    ), "`top_k` has to be > 1 or `top_p` has to be < 1.0"

    responses = [
        model.generate(image, prompt, **generate_kwargs)
        for _ in tqdm(range(k), disable=not show_progress_bar)
    ]

    return responses


def get_constrained_response_set(
    model, image, prompt, k, spacy_model, possible_outputs, exclusion_words=[], max_tries_per_k=25, show_progress_bar=False, verbose=False, **generate_kwargs
):

    assert (
        generate_kwargs.get("do_sample", False) is True
    ), "`do_sample` has to be set to True"
    assert (
        generate_kwargs.get("top_k", 1) > 1 or generate_kwargs.get("top_p", 1.0) < 1.0
    ), "`top_k` has to be > 1 or `top_p` has to be < 1.0"

    responses = []
    pbar = tqdm(total=k, disable=not show_progress_bar)
    tries = 0
    while len(responses) < k:
        response = model.generate(image, prompt, **generate_kwargs)
        
        response_start = generate_kwargs.get('response_start', None)
        pruned_response = response.replace(response_start, '') if response_start else response
        
        head_text, compound_text, pos = extract_noun_from_response(pruned_response, spacy_model, exclusion_words)
        compound_text = compound_text.replace(' ', '_').strip()
    
        label = compound_text if compound_text in possible_outputs else head_text  # try compound, default to head noun
        if label in possible_outputs:
            responses.append((response, label))
        if verbose:
            print(f'{response} | {label} | valid: {label in possible_outputs}')
        pbar.update(1)
        tries += 1
        if tries > k * max_tries_per_k:
            break
    pbar.close()

    return responses


def get_constrained_response_set_from_messages(
    model, image, messages, k, spacy_model, possible_outputs, exclusion_words=[], max_tries_per_k=25, show_progress_bar=False, verbose=False, **generate_kwargs
):

    assert (
        generate_kwargs.get("do_sample", False) is True
    ), "`do_sample` has to be set to True"
    assert (
        generate_kwargs.get("top_k", 1) > 1 or generate_kwargs.get("top_p", 1.0) < 1.0
    ), "`top_k` has to be > 1 or `top_p` has to be < 1.0"

    responses = []
    pbar = tqdm(total=k, disable=not show_progress_bar)
    tries = 0
    while len(responses) < k:
        response = model.generate_from_messages([image], messages, **generate_kwargs)
        
        response_start = generate_kwargs.get('response_start', None)
        pruned_response = response.replace(response_start, '') if response_start else response
        
        head_text, compound_text, pos = extract_noun_from_response(pruned_response, spacy_model, exclusion_words)
        compound_text = compound_text.replace(' ', '_').strip()
    
        label = compound_text if compound_text in possible_outputs else head_text  # try compound, default to head noun
        if label in possible_outputs:
            responses.append((response, label))
        if verbose:
            print(f'{response} | {label} | valid: {label in possible_outputs}')
        pbar.update(1)
        tries += 1
        if tries > k * max_tries_per_k:
            break
    pbar.close()

    return responses


def two_step_predictions(
    model, image, location_prompt, description_prompt, k, spacy_model, 
    possible_outputs, exclusion_words=[], max_tries_per_k=10, 
    show_progress_bar=False, verbose=False, **generate_kwargs
):

    # init
    conversation = model.init_chat(location_prompt, image)
    # generate location prediction (deterministic decoding)
    location_response = model.generate_from_messages(
        [image], conversation, max_new_tokens=generate_kwargs.get('max_new_tokens', 100), do_sample=False)
    location_tuple = location_response, get_location_from_model_response(location_response)
    # append answer
    conversation = model.add_to_chat(conversation, "assistant", location_response)
    # append second prompt
    conversation = model.add_to_chat(conversation, "user", description_prompt)
    if verbose:
        print(conversation)
    
    # generate response set for tangram description
    response_set = get_constrained_response_set_from_messages(
        model, image, conversation, k, spacy_model, 
        possible_outputs, exclusion_words, max_tries_per_k, 
        show_progress_bar, verbose, **generate_kwargs)
    
    # append answer
    conversation = model.add_to_chat(conversation, "assistant", '; '.join([r[0] for r in response_set]))

    return location_tuple, response_set, conversation


def get_wn_lemmas():
    lemmas = set()

    for synset in wn.all_synsets():
        if synset.pos() == 'n':
            synset_lemmas = set(synset.lemma_names())
            lemmas |= synset_lemmas

    lemmas = {
        l.lower().replace('_', ' ') for l in lemmas
    }

    return lemmas