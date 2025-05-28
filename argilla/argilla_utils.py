import base64
from string import ascii_letters, digits, ascii_lowercase
from random import choices, choice, Random
from os import path as osp
import argilla as rg
from argilla.client.feedback.utils import get_file_data
from textwrap import dedent


def img_to_html(img_path, mode="link", style_attrs=None):

    # create url image (link or data url)
    if mode == "link":
        img_url = img_path
    elif mode == "embed":
        file_data, file_type = get_file_data(img_path, media_type="image")
        media_base64 = base64.b64encode(file_data).decode("utf-8")
        img_url = f"data:image/{file_type};base64,{media_base64}"
    else:
        raise NotImplementedError

    # parse style attributes
    if style_attrs is not None:
        style_str = (
            ' style="' + ";".join([f"{k}:{v}" for k, v in style_attrs.items()]) + ';"'
        )
    else:
        style_str = ""

    # create html img tag with image url and style info
    html = f'<img src="{img_url}"{style_str}>'

    return html


def make_password(length=10, random_seed=None):
    chars = set(ascii_letters + digits)
    chars -= {"0", "O"}  # avoid confusion
    return "".join(Random(random_seed).choices(list(chars), k=length))


def make_name(length=8, random_seed=None):
    return "".join(Random(random_seed).choices(ascii_lowercase, k=length))


def get_img_url(img_id, mode, location, tangram_pos=None):
    id_string = str(int(img_id)).rjust(3, "0")
    img_filename = (
        f"{id_string}_{mode}.png"
        if tangram_pos is None
        else f"{id_string}_{mode}_{tangram_pos}.png"
    )
    return osp.join(location, img_filename)


def make_html_img(path):
    html = f'<img src="{path}" style="height:70vh;margin:auto;display:block;">'
    return html


def make_instruction_html_img(path):
    html = f'<img src="{path}" style="max-width:500px;margin:auto;display:block;">'
    return html


def build_info_dataset_v0():
    # define INFO dataset

    example_url = "URL/tangram_example.png"
    example_html = make_instruction_html_img(example_url)

    info_message = dedent(f"""
    **Annotation Guidelines**

    The task for this annotation series is to locate and name *tangrams* in image grids. 

    Tangrams are abstract figures composed of simple shapes that can be interpreted in different ways. 
    For example, the following tangram could be interpreted as a *table*, *turtle*, or *bathtub*, among other possibilities: 

    {example_html}

    In the data set `02_annotations` you will find a set of items, each of which consists of a 2 by 2 grid. Each grid contains exactly one tangram tile. 
    **Locate the tangram in the grid and describe what it looks like in the text field.**

    After you've finished the annotations in `02_annotations`, move on to `03_finish`, where you'll find the completion code for Prolific.

    Please indicate that you have read these instructions by selecting the `Yes` label and proceed to `02_annotations`. 
    """.strip())

    info_dataset_name = f"01_info"

    # define dataset
    fields = [rg.TextField(name="content", title="", use_markdown=True)]
    questions = [
        rg.LabelQuestion(
            name="response",
            title="Did you read the instructions?",
            labels={"YES": "Yes", "NO": "No"},  # or ["YES","NO"]
            required=True,
            visible_labels=None,
        )
    ]
    info_dataset = rg.FeedbackDataset(
        fields=fields, questions=questions
    )

    # final record with information to finish study on prolific
    last_metadata = {
        "item_id": None,
        # "workspace_name": workspace_name,
        "dataset_name": info_dataset_name,
        "meta_record": True,
    }
    last_record = rg.FeedbackRecord(
        fields={"content": info_message}, metadata=last_metadata, external_id=None
    )
    records = [last_record]

    # add records to dataset & push to workspace
    info_dataset.add_records(records)

    return info_dataset


def build_completion_dataset_v0(completion_code):
    # define COMPLETION dataset

    completion_message = dedent(f"""
    **Thank you for participating!**

    If you have submitted responses for all items, please select the `Yes` button. You can then return to Prolific to complete the assignment.
    **It will be verified that you have submitted answers for all items.**

    Use the following completion code to conclude the task on Prolific:

    *{completion_code}*

    Thank you for your participation!

    (Please do not refresh or close this page before entering the code on Prolific.
    You can view this message again by selecting the "Submitted" instead of the "Pending" records at the top left of the screen and then skip to the last entry).
    """.strip())

    completion_dataset_name = f"03_completion"

    # define dataset
    fields = [rg.TextField(name="content", title="", use_markdown=True)]
    questions = [
        rg.LabelQuestion(
            name="response",
            title="Did complete all annotations?",
            labels={"YES": "Yes", "NO": "No"},  # or ["YES","NO"]
            required=True,
            visible_labels=None,
        )
    ]
    completion_dataset = rg.FeedbackDataset(
        fields=fields, questions=questions
    )

    # final record with information to finish study on prolific
    last_metadata = {
        "item_id": None,
        # "workspace_name": workspace_name,
        "dataset_name": completion_dataset_name,
        "meta_record": True,
    }
    last_record = rg.FeedbackRecord(
        fields={"content": completion_message}, metadata=last_metadata, external_id=None
    )
    records = [last_record]

    # add records to dataset & push to workspace
    completion_dataset.add_records(records)

    return completion_dataset


def build_annotation_dataset_v0(workspace_name, partition_df, base_img_path, mode):

    partition = partition_df.loc[partition_df.workspace == workspace_name]
    dataset_name = f"02_annotation_{workspace_name}"

    # define dataset
    fields = [rg.TextField(name="content", title="", use_markdown=True)]
    questions = [
        rg.TextQuestion(
            name="response",
            title="Please provide your response:",
            required=True,
            use_markdown=False,
        )
    ]

    annotation_dataset = rg.FeedbackDataset(
        fields=fields, questions=questions
    )

    # compile records
    records = []
    for item_id in partition.item_id:
        # pick tangram position for grid setup
        tangram_pos = choice(["tl", "tr", "bl", "br"]) if mode == "grid" else None
        image_url = get_img_url(item_id, mode, base_img_path, tangram_pos)
        image_html = make_html_img(image_url)
        metadata = {
            "item_id": item_id,
            "workspace_name": workspace_name,
            "dataset_name": dataset_name,
            "meta_record": False,
        }
        record = rg.FeedbackRecord(
            fields={"content": image_html}, metadata=metadata, external_id=item_id
        )
        records.append(record)

    # add records to dataset
    annotation_dataset.add_records(records)

    return annotation_dataset

##############################
#  for individual workspaces #
##############################


def build_info_dataset(workspace):
    # define INFO dataset

    example_url = "URL/tangram_example.png"
    example_html = make_instruction_html_img(example_url)

    info_message = dedent(f"""
**Annotation Guidelines**

The task for this annotation series is to name *tangrams* in image grids. 

Tangrams are abstract figures composed of simple shapes that can be interpreted in different ways. 
For example, the following tangram could be interpreted as a *table*, *turtle*, or *bathtub*, among other possibilities: 

{example_html}

In the dataset `02_annotation_{workspace.name}` you will find a set of items, each of which consists of a 2 by 2 grid. Each grid contains exactly one tangram tile. 
**Locate the tangram in the grid and describe what it looks like in the text field.** Use short descriptions such as *bathtub* or *swimming turtle*.

After reading this, please continue to the `02_annotation_{workspace.name}` dataset by clicking on `{workspace.name}` at the top left of this page.
After you've finished the annotations in `02_annotation_{workspace.name}`, move on to `03_completion_{workspace.name}`, where you'll find the completion code for Prolific.

Please indicate that you have read these instructions by selecting the `Yes` label and enter your Prolific ID in the text field. After this, proceed to `02_annotation_{workspace.name}`. 
    """.strip())
    
    info_dataset_name = f"01_info_{workspace.name}"

    # define dataset
    fields = [rg.TextField(name="content", title="", use_markdown=True)]
    questions = [
        rg.LabelQuestion(
            name="info_confirmation",
            title="Did you read the instructions?",
            labels={"YES": "Yes", "NO": "No"},  # or ["YES","NO"]
            required=True,
            visible_labels=None,
        ),         
        rg.TextQuestion(
            name="prolific_id",
            title="Please enter your Prolific ID:",
            required=True,
            use_markdown=False,
        )
    ]
    info_dataset = rg.FeedbackDataset(
        fields=fields, questions=questions
    )

    # final record with information to finish study on prolific
    last_metadata = {
        "item_id": None,
        # "workspace_name": workspace_name,
        "dataset_name": info_dataset_name,
        "meta_record": True,
    }
    last_record = rg.FeedbackRecord(
        fields={"content": info_message}, metadata=last_metadata, external_id=None
    )
    records = [last_record]

    # add records to dataset & push to workspace
    info_dataset.add_records(records)

    return info_dataset


def build_completion_dataset(workspace, completion_code, completion_url):
    # define COMPLETION dataset

    completion_message = dedent(f"""
**Thank you for participating!**

If you have submitted responses for all items, please select the `Yes` button. Please use the text box on the right if you have any comments or remarks. You can then return to Prolific to complete the assignment.
**It will be verified that you have submitted answers for all items.**

Use the following completion code to conclude the task on Prolific:

*{completion_code}*

You can also click the following link to directly go back to Prolific:
<a href='{completion_url}'>{completion_url}</a>

Thank you for your participation!

(Please do not refresh or close this page before entering the code on Prolific.
You can view this message again by selecting the "Submitted" instead of the "Pending" records at the top left of the screen and then skip to the last entry).
    """.strip())    

    completion_dataset_name = f"03_completion_{workspace.name}"

    # define dataset
    fields = [rg.TextField(name="content", title="", use_markdown=True)]
    questions = [
        rg.LabelQuestion(
            name="submission_confirmation",
            title="Did complete all annotations?",
            labels={"YES": "Yes", "NO": "No"},  # or ["YES","NO"]
            required=True,
            visible_labels=None,
        ),         
        rg.TextQuestion(
            name="comments",
            title="Please enter your comments or remarks:",
            required=False,
            use_markdown=False,
        )
    ]
    completion_dataset = rg.FeedbackDataset(
        fields=fields, questions=questions
    )

    # final record with information to finish study on prolific
    last_metadata = {
        "item_id": None,
        # "workspace_name": workspace_name,
        "dataset_name": completion_dataset_name,
        "meta_record": True,
    }
    last_record = rg.FeedbackRecord(
        fields={"content": completion_message}, metadata=last_metadata, external_id=None
    )
    records = [last_record]

    # add records to dataset & push to workspace
    completion_dataset.add_records(records)

    return completion_dataset


def build_annotation_dataset(workspace, workspace_partition_map, partition_df, base_img_path, mode, random_seed=123):

    partition_name = workspace_partition_map[workspace.name]
    partition = partition_df.loc[partition_df.partition == partition_name]
    dataset_name = f"02_annotation_{workspace.name}"

    # define dataset
    fields = [rg.TextField(name="content", title="", use_markdown=True)]
    questions = [
        rg.TextQuestion(
            name="response",
            title="Please provide your response:",
            required=True,
            use_markdown=False,
        )
    ]

    guidelines = "Describe what the tangram looks like in the text field. Use short descriptions such as *bathtub* or *swimming turtle*."
    annotation_dataset = rg.FeedbackDataset(
        fields=fields, questions=questions, guidelines=guidelines
    )

    # compile records
    records = []
    for item_id in partition.item_id:
        # pick tangram position for grid setup
        tangram_pos = Random(random_seed + item_id).choice(["tl", "tr", "bl", "br"]) if mode == "grid" else None
        image_url = get_img_url(item_id, mode, base_img_path, tangram_pos)
        image_html = make_html_img(image_url)
        metadata = {
            "item_id": item_id,
            "workspace_name": workspace.name,
            "partition_name": partition_name,
            "dataset_name": dataset_name,
            "tangram_pos": tangram_pos,
            "image_url": image_url,
            "meta_record": False,
        }
        record = rg.FeedbackRecord(
            fields={"content": image_html}, metadata=metadata, external_id=item_id
        )
        records.append(record)

    # add records to dataset
    annotation_dataset.add_records(records)

    return annotation_dataset


def build_dummy_annotation_dataset(workspace, base_img_path, mode):

    dataset_name = f"02_annotation_{workspace.name}"

    # define dataset
    fields = [rg.TextField(name="content", title="", use_markdown=True)]
    questions = [
        rg.TextQuestion(
            name="response",
            title="Please provide your response:",
            required=True,
            use_markdown=False,
        )
    ]

    annotation_dataset = rg.FeedbackDataset(
        fields=fields, questions=questions
    )

    # compile records
    records = []
    for item_id in range(10):
        
        image_filename = f'{item_id}.png'
        image_url = osp.join(base_img_path, image_filename)
        image_html = make_html_img(image_url)
        metadata = {
            "item_id": item_id,
            "workspace_name": workspace.name,
            "partition_name": 'x',
            "dataset_name": dataset_name,
            "meta_record": False,
        }
        record = rg.FeedbackRecord(
            fields={"content": image_html}, metadata=metadata, external_id=item_id
        )
        records.append(record)

    # add records to dataset
    annotation_dataset.add_records(records)

    return annotation_dataset