# SceneGram

Code and data for the paper ["SceneGram: Conceptualizing and Describing Tangrams in Scene Context"](https://aclanthology.org/2025.findings-acl.1229/) (ACL Findings 2025)

## Steps for Building Dataset:

1. clone repository (including submodules)
2. run `bash generate_items.sh`

After this:
- Annotations can be found in `scenegram_data/scenegram.csv`
    - `image_url` column specifies the corresponding image for each annotation
- Images are stored in `generated_items`

------------

## Project Structure

### Annotations

#### Data Collection

- Notebooks in `argilla`:
    1. `argilla_data_preparation.ipynb` $\rightarrow$ general setup
    2. `argilla_experiment_slices.ipynb` $\rightarrow$ experiment slice generation
    3. `argilla_setup_from_experiment_slices.ipynb` $\rightarrow$ creating annotator tasks from experiment slices
    4. `retrieve_annotations.ipynb` $\rightarrow$ downloading annotations from argilla
    5. `view_annotations.ipynb` $\rightarrow$ inspecting annotations
    6. `additional_user.ipynb` $\rightarrow$ used if annotations have to be repeated: marks user / partition as invalid and creates new copy of the annotation partition on argilla
- Steps 5 and 6: repeated until data is complete

#### Processing Pipeline / Annotations

- Notebooks in `processing_annotations`: 
    1. `merge_raw_annotations.ipynb` $\rightarrow$ merge annotation slices and filter out invalid annotations
    2. `np_extraction.ipynb` $\rightarrow$ create cleaned annotations and extract head nouns using SpaCy (validated afterwards)
    3. `merge_processed_annotations.ipynb` $\rightarrow$ merge validated and corrected data cleaning results
    4. `auto_synset_assignment.ipynb` $\rightarrow$ map to WordNet using NLTK (validated afterwards)
    5. `merge_wn_anns.ipynb` $\rightarrow$ merge validated and corrected WordNet mapping results
    6. `clip_encode.ipynb` $\rightarrow$ create CLIP encodings
    7. `static_encode.ipynb` $\rightarrow$ create embeddings with GloVe and ConceptNet Numberbatch
    8. `remove_cols.ipynb` $\rightarrow$ remove columns from data which are not required

#### Analysis Pipeline / Annotations

- Notebooks in `analyzing_annotations`
    1. `analyze_snd.ipynb` $\rightarrow$ SND scores
    2. `analyze_labels.ipynb` $\rightarrow$ frequency analysis and entropy
    3. `analyze_clip.ipynb` $\rightarrow$ CLIP based analysis
    4. `analyze_static_embeds.ipynb` $\rightarrow$ GloVe and ConceptNet analysis
    5. `examples.ipynb` $\rightarrow$ Examples

### Model Predictions

#### Collecting Predictions

- Code in `modelling`
    - `predict_two_step.py` $\rightarrow$ run the inference pipeline

#### Processing Pipeline / Models

- Notebooks in `process_predictions`:
    1. `process_predictions.ipynb` $\rightarrow$ reformat and merge model predictions
    2. `clip_encode.ipynb` $\rightarrow$ create CLIP encodings
    3. `static_encode.ipynb` $\rightarrow$ create embeddings with GloVe and ConceptNet Numberbatch

### Analysis Pipeline / Models

- Notebooks in `analyzing_predictions`:
    1. `analyze_labels.ipynb` $\rightarrow$ frequency analysis and entropy
    2. `analyze_static_embeds.ipynb` $\rightarrow$ GloVe and ConceptNet analysis
    3. `examples.ipynb` $\rightarrow$ Examples

### Data

- `generate_items.py`: item generation script
- `scenegram_data`: SceneGram annotations
- `kilogram`: Kilogram dataset as git submodule
- `generated_scenes`: generated scene images
- `generated_items`: generated items / combinations of kilogram tangrams and generated scene images
- `generated_data`: raw data for setting up the crowdsourcing process
- `collected_data`: results of the crowdsourcing process, processed files

# Citations

## Resources Used in this Project

- Tangrams from KiloGram: [github.com/lil-lab/kilogram](https://github.com/lil-lab/kilogram/tree/main)
- Scene categories from [Lauer2018: The role of scene summary statistics in object recognition](https://www.nature.com/articles/s41598-018-32991-1)
- Scene images created with [SDXL-Lightning](https://huggingface.co/spaces/ByteDance/SDXL-Lightning) (4 inference steps)
    - Prompt: "a photograph of a(n) [scene]"

## Citing this Project

```
@inproceedings{junker-zarriess-2025-scenegram,
    title = "{S}cene{G}ram: Conceptualizing and Describing Tangrams in Scene Context",
    author = "Junker, Simeon  and
      Zarrie{\ss}, Sina",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1229/",
    pages = "23976--23992",
    ISBN = "979-8-89176-256-5",
    abstract = "Research on reference and naming suggests that humans can come up with very different ways of conceptualizing and referring to the same object, e.g. the same abstract tangram shape can be a ``crab'', ``sink'' or ``space ship''. Another common assumption in cognitive science is that scene context fundamentally shapes our visual perception of objects and conceptual expectations. This paper contributes SceneGram, a dataset of human references to tangram shapes placed in different scene contexts, allowing for systematic analyses of the effect of scene context on conceptualization. Based on this data, we analyze references to tangram shapes generated by multimodal LLMs, showing that these models do not account for the richness and variability of conceptualizations found in human references."
}
```
