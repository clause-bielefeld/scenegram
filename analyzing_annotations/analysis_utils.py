import os.path as osp
import pandas as pd
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize as tokenize
from nltk.corpus import stopwords
from PIL import Image
from IPython.display import display
from nltk.corpus import wordnet as wn

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def clean_str(s):
    return s.strip().lower()


def read_ann_df(path):
    ann_df = pd.read_csv(path, index_col=0)

    # order for workspaces / users
    workspaces = pd.unique(ann_df.workspace_name)
    for workspace in workspaces:
        workspace_selection = ann_df.loc[ann_df.workspace_name == workspace]
        workspace_selection = workspace_selection.sort_values(by="time_delta")
        workspace_selection.loc[:, "order_idx"] = range(len(workspace_selection))
        ann_df.loc[workspace_selection.index, "order_idx"] = (
            workspace_selection.order_idx
        )
    ann_df = ann_df.astype({"order_idx": int})

    # clean strings
    ann_df.clean_annotation = ann_df.clean_annotation.map(clean_str)
    ann_df.head_noun = ann_df.head_noun.map(clean_str)

    # set tangram + scene as multiindex
    ann_df = ann_df.set_index(["tangram", "scene"]).sort_index()

    return ann_df


def clean_wl(x):
    words = tokenize(x.lower())
    wl = [
        stemmer.stem(w)
        for w in words
        if w not in stop_words and (w.islower() or w.isalnum())
    ]
    return wl


def naming_div(anns):
    cleaned_ann_list = []

    for ann in anns:
        cleaned_ann = clean_wl(ann)
        cleaned_ann_list.append(cleaned_ann)

    nd = 0
    num_ann = len(cleaned_ann_list)
    # each annotation
    for i in range(len(cleaned_ann_list)):
        frq = 0
        # each word in one annotation
        wl = cleaned_ann_list[i]
        for w in wl:
            appeared = 0
            for j in range(len(cleaned_ann_list)):
                if j != i:
                    wll = cleaned_ann_list[j]
                    if w in wll:
                        appeared += 1
            frq += 1 - appeared / (
                num_ann - 1
            )  # proportion of the word appearing in other annotations
        if len(wl) != 0:
            nd += frq / len(wl)  # nd += mean frq (1-p) of each annotation
    return nd / num_ann


def display_img(idx, ref_df, img_location, size=(256, 256)):
    x = ref_df.loc[idx].iloc[0]
    img_path = osp.join(img_location, x.image_url)
    img = Image.open(img_path).resize(size)

    display(img)


def make_synset(s):
    if type(s) == str:
        return wn.synset(s)
    return s


def get_hypernyms(synset, include_self=True):
    synset = make_synset(synset)
    hypernyms = {synset} if include_self else set()
    for hypernym in synset.hypernyms():
        hypernyms |= set(get_hypernyms(hypernym))
    return hypernyms | set(synset.hypernyms())


def get_hyponyms(synset, include_self=True):
    synset = make_synset(synset)
    hyponyms = {synset} if include_self else set()
    for hyponym in synset.hyponyms():
        hyponyms |= set(get_hyponyms(hyponym))
    return hyponyms | set(synset.hyponyms())


def is_hypernym_of(synset, *reference_synsets, include_self=True):
    synset = make_synset(synset)
    reference_hypernyms = set()
    for r in reference_synsets:
        reference_hypernyms |= get_hypernyms(make_synset(r), include_self=include_self)
    return synset in reference_hypernyms


def is_hyponym_of(synset, *reference_synsets, include_self=True):
    synset = make_synset(synset)
    reference_hyponyms = set()
    for r in reference_synsets:
        reference_hyponyms |= get_hyponyms(make_synset(r), include_self=include_self)
    return synset in reference_hyponyms


def get_first_lemma(synset):
    return make_synset(synset).lemma_names()[0]


# Utils for head noun extraction with spacy


def get_compounds(token):

    tokens = [token]
    for t in token.children:
        if t.dep_ == "compound":
            # recursive call
            tokens += get_compounds(t)

    return tokens


def get_compound_str(token):

    compound_tokens = get_compounds(token)
    sorted_compound_tokens = sorted(compound_tokens, key=lambda x: x.i)
    compound_string = " ".join([t.text for t in sorted_compound_tokens])

    return compound_string


# adapted from kilogram code (+ compounds)


def get_np_head(s, spacy_model, normalize=True):  # -> Any | None:

    if normalize:
        s = s.lower().strip()

    #  hard coded fix typo
    if s.startswith("aa "):
        s = s.replace("aa ", "a ")

    #  get tree
    doc = spacy_model(s)

    #  single word
    if len(doc) == 1:
        return doc[0]

    np_head = None
    for token in doc:
        if token.dep_ == "ROOT" and token.head.pos_ in [
            "NOUN",
            "INTJ",
            "PROPN",
            "PRON",
            "ADJ",
            "ADV",
        ]:
            np_head = token

        if token.dep_ == "ROOT" and token.head.pos_ == "VERB":
            if list(token.children)[0].dep_ == "prep":
                np_head = token
            else:
                np_head = list(token.children)[0]

        if token.dep_ == "ROOT" and token.head.pos_ == "ADP":
            np_head = list(token.children)[-1]

        #  hard code "xx can" utterances
        if token.dep_ == "ROOT" and token.text == "can":
            np_head = token

    return np_head


def get_head_string(s, spacy_model, normalize=True):
    head = get_np_head(s, spacy_model, normalize=normalize)
    if head:
        return head.text


def get_head_compound_string(s, spacy_model, normalize=True):
    head = get_np_head(s, spacy_model, normalize=normalize)
    if head:
        return get_compound_str(head)


def transform_list_into_heads(l, spacy_model, normalize=True):
    return [get_head_compound_string(x, spacy_model, normalize=normalize) for x in l]


def mean_reciprocal_rank(l, ref):
    # ensure only valid strings
    ref = [r for r in ref if type(r) == str]
    # get unique labels and counts
    labels, counts = np.unique(ref, return_counts=True)
    # transform counts into ranks
    unique_counts = sorted(set(counts), reverse=True)
    # get dict of label ranks
    ref_label_ranks = {l: unique_counts.index(c) + 1 for l, c in zip(labels, counts)}

    # map labels to referece ranks
    ranks = [ref_label_ranks.get(x, len(ref_label_ranks) + 1) for x in l]
    # inverse ranks
    inverse_ranks = [1 / r for r in ranks]
    # sum
    inverse_rank_sum = sum(inverse_ranks)
    # mean
    mean_inverse_rank = inverse_rank_sum / len(l)
    return mean_inverse_rank
