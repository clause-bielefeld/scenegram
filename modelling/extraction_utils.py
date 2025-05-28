from spacy.tokens.span import Span
import re

def is_phrase_head(token):
    return token['id'] == token['head']

def get_phrase_head(doc):
    head_mask = [is_phrase_head(t) for t in doc.to_json()['tokens']]
    assert head_mask.count(True) == 1
    head_idx = head_mask.index(True)
    return doc[head_idx]

def get_compounds(token):
    
    tokens = [token]
    for t in token.children:
        if t.dep_ == 'compound':
            # recursive call
            tokens += get_compounds(t)
        
    return tokens

def get_compound_str(token):

    compound_tokens = get_compounds(token)
    sorted_compound_tokens = sorted(compound_tokens, key=lambda x: x.i)
    # lemmatize final token in compound
    compound_strings = [
        t.lemma_ if i == len(sorted_compound_tokens) -1 else t.text 
        for i, t in enumerate(sorted_compound_tokens)]
    compound_string = ' '.join(compound_strings)
    
    return compound_string

def get_chunk_head(chunk):
    if type(chunk) == Span:
        chunk = chunk.as_doc()
    head = get_phrase_head(chunk)
    return head.lemma_, get_compound_str(head), head.pos_

def extract_noun_from_response(response, model, exclusion_words=[]):
    doc = model(response)

    chunks = list(doc.noun_chunks)

    if len(chunks) > 0:
        for chunk in chunks:
            head_text, compound_text, pos = get_chunk_head(chunk)
            if pos == 'NOUN' and head_text not in exclusion_words:
                break
    else:
        head_text = compound_text = pos = ''

    return head_text, compound_text, pos

def get_location_from_model_response(model_response):
    options = [
        (r'top left', 'tl'), 
        (r'top right', 'tr'), 
        (r'bottom left', 'bl'), 
        (r'bottom right', 'br')]
    
    re_map = {
        re.compile(o[0], flags=re.IGNORECASE): o[1] for o in options
    }
    
    matches = [v for k, v in re_map.items() if k.search(model_response)]
    location = matches[0] if len(matches) == 1 else None
    
    return location