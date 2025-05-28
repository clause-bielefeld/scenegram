import os.path as osp
from PIL import Image, ImageDraw
from svg_utils import set_svg_attributes, svg_string_to_img
from random import Random

SCENE_CATEGORIES = [
    # photo of a kitchen
    # Lauer2018: The role of scene summary statistics in object recognition
    "none",         # 1 / baseline 
    "kitchen",      # 2 / indoor 1
    "bathroom",     # 3 / indoor 2    
    "bedroom",      # 4 / indoor 3      
    "office",       # 5 / indoor 4       
    "forest",       # 6 / outdoor 1        
    "mountain",     # 7 / outdoor 2   
    "beach",        # 8 / outdoor 3          
    "street",       # 9 / outdoor 4
    ###########################
    "sky",          # 10 / outdoor 5
    "sea_bottom"    # 11 / outdoor 6    
    
]


def center_starting_pos(outer_dim, inner_dim):
    return (outer_dim - inner_dim) // 2


def get_centering_pos(outer, inner):
    return tuple([center_starting_pos(outer.size[i], inner.size[i]) for i in range(2)])


def build_tangram(idx, size, tangram_data, tangram_dir, svg_attrs):

    tg = tangram_data[idx]
    svg_path = osp.join(tangram_dir, f"{tg}.svg")

    svg_attrs["maxdim"] = size

    with open(svg_path) as f:
        s = f.read()
    s = set_svg_attributes(s, **svg_attrs)
    tangram_img = svg_string_to_img(s)

    return tangram_img


def build_context_from_filename(filename, context_dir, size):

    scene_img = Image.open(osp.join(context_dir, filename))
    scene_img = scene_img.resize((size, size))

    return scene_img


def build_context(context_type, context_dir, size, variant=0):

    filename = (
        f"{context_type}.png" if variant is None else f"{context_type}_{variant}.png"
    )
    
    return build_context_from_filename(filename, context_dir, size)


def build_rectangle(w, h, alpha=200, fill="gray", outline="white", width=2):
    rectangle = Image.new("RGB", (w, h))
    # create rectangle image
    img1 = ImageDraw.Draw(rectangle)
    img1.rectangle(((0, 0), (w - 1, h - 1)), fill=fill, outline=outline, width=width)
    if not 0 <= alpha <= 255:
        # add alpha channel
        alpha = 255
    rectangle.putalpha(alpha)
    return rectangle


def build_square(square_dim, alpha=200, fill="gray", outline="white", width=2):
    w = h = square_dim
    return build_rectangle(
        w=w, h=h, alpha=alpha, fill=fill, outline=outline, width=width
    )


def build_tangram_wrapper(
    tangram_image,
    size=None,
    size_ratio=1.2,
    alpha=200,
    fill="gray",
    outline="white",
    width=2,
):
    if size is not None:
        sq_dim = size
    else:
        sq_dim = int(max(tangram_image.size) * size_ratio)
    return build_square(sq_dim, alpha=alpha, fill=fill, outline=outline, width=width)


def build_inline_sample(
    tangram_idx,
    context_category,
    tangram_data,
    svg_attrs,
    tangram_dir,
    context_dir,
    size=512,
    context_size_ratio=2,
    wrapper_size_ratio=1.2,
    wrapper_alpha=200,
    wrapper_fill="gray",
    wrapper_outline="white",
    wrapper_border_width=2,
    **kwargs,
):

    # tangram image
    tangram_size = int(size // context_size_ratio)
    tangram = build_tangram(
        tangram_idx, tangram_size, tangram_data, tangram_dir, svg_attrs
    )

    # context image
    context_size = size
    variant = 3 if context_category == 'none' else 0
    context_image = build_context(context_category, context_dir, context_size, variant=variant)

    # tangram wrapper
    wrapper_size = int(max(tangram.size) * wrapper_size_ratio)
    rectangle = build_tangram_wrapper(
        tangram,
        size=wrapper_size,
        alpha=wrapper_alpha,
        fill=wrapper_fill,
        outline=wrapper_outline,
        width=wrapper_border_width,
    )

    # merge images
    merged = context_image.copy()
    merged.paste(rectangle, get_centering_pos(merged, rectangle), mask=rectangle)
    merged.paste(tangram, get_centering_pos(merged, tangram), mask=tangram)

    return merged


def build_sidebyside_sample(
    tangram_idx,
    context_category,
    tangram_data,
    svg_attrs,
    tangram_dir,
    context_dir,
    size=512,
    wrapper_size_ratio=1.2,
    wrapper_fill="gray",
    wrapper_outline="white",
    wrapper_border_width=2,
    **kwargs,
):

    # tangram image
    tangram_size = int(size // wrapper_size_ratio)
    tangram = build_tangram(
        tangram_idx, tangram_size, tangram_data, tangram_dir, svg_attrs
    )

    # tangram wrapper

    tangram_wrapper = build_tangram_wrapper(
        tangram,
        size=size,
        fill=wrapper_fill,
        outline=wrapper_outline,
        width=wrapper_border_width,
        alpha=255,
    )
    tangram_wrapper.paste(
        tangram, get_centering_pos(tangram_wrapper, tangram), mask=tangram
    )

    # context image
    variant = 3 if context_category == 'none' else 0
    context_image = build_context(context_category, context_dir, size, variant=variant)

    # merge images
    merged = build_rectangle(size * 2, size)
    merged.paste(context_image, (0, 0))
    merged.paste(tangram_wrapper, (size, 0))

    return merged


def build_grid_sample(
    tangram_idx,
    context_category,
    tangram_data,
    svg_attrs,
    tangram_dir,
    context_dir,
    tangram_pos=0,  # 0: upper left, 1: upper right, 2: lower left, 3: lower right
    shuffle_contexts=True,
    random_seed=123,
    size=512,
    wrapper_size_ratio=1.2,
    wrapper_fill="gray",
    wrapper_outline="white",
    wrapper_border_width=2,
    **kwargs,
):

    # tangram image
    tangram_size = int(size // wrapper_size_ratio) // 2
    tangram = build_tangram(
        tangram_idx, tangram_size, tangram_data, tangram_dir, svg_attrs
    )

    # tangram wrapper

    tangram_wrapper = build_tangram_wrapper(
        tangram,
        size=size // 2,
        fill=wrapper_fill,
        outline=wrapper_outline,
        width=wrapper_border_width,
        alpha=255,
    )

    tangram_wrapper.paste(
        tangram, get_centering_pos(tangram_wrapper, tangram), mask=tangram
    )

    # context image
    context_images = [
        build_context(context_category, context_dir, size // 2, variant=i)
        for i in range(3)
    ]
    if shuffle_contexts:
        seed = (
            random_seed + tangram_pos + tangram_idx
        )  # ensure that order is different for different tangram positions and tangrams
        Random(seed).shuffle(context_images)
        img_part_names = [f"{context_category}_{i}" for i in range(3)]
        Random(seed).shuffle(img_part_names)

    # combine contexts and tangram
    img_parts = context_images
    img_parts.insert(tangram_pos, tangram_wrapper)  # include tangram at tangram_pos
    img_part_names.insert(tangram_pos, 'tgt')
    # wrapper
    merged = build_square(size)

    merged.paste(img_parts[0])  # upper left
    merged.paste(img_parts[1], (size // 2, 0))  # upper right
    merged.paste(img_parts[2], (0, size // 2))  # lower left
    merged.paste(img_parts[3], (size // 2, size // 2))  # lower right

    return merged, img_part_names


def build_grid_sample_from_template(
    tangram_idx,
    tangram_data,
    svg_attrs,
    tangram_dir,
    context_dir,
    item_configuration,
    size=512,
    wrapper_size_ratio=1.2,
    wrapper_fill="gray",
    wrapper_outline="white",
    wrapper_border_width=2,
    **kwargs,
):

    # tangram image
    tangram_size = int(size // wrapper_size_ratio) // 2
    tangram = build_tangram(
        tangram_idx, tangram_size, tangram_data, tangram_dir, svg_attrs
    )

    # tangram wrapper

    tangram_wrapper = build_tangram_wrapper(
        tangram,
        size=size // 2,
        fill=wrapper_fill,
        outline=wrapper_outline,
        width=wrapper_border_width,
        alpha=255,
    )

    tangram_wrapper.paste(
        tangram, get_centering_pos(tangram_wrapper, tangram), mask=tangram
    )
    
    tangram_pos = item_configuration.index('tgt')
    context_order = [i for i in item_configuration if i != 'tgt']
    context_filenames = [c + '.png' for c in context_order]

    # context image
    context_images = [
        build_context_from_filename(filename, context_dir, size // 2)
        for filename in context_filenames
    ]

    # combine contexts and tangram
    img_parts = context_images
    img_parts.insert(tangram_pos, tangram_wrapper)  # include tangram at tangram_pos
    # wrapper
    merged = build_square(size)

    merged.paste(img_parts[0])  # upper left
    merged.paste(img_parts[1], (size // 2, 0))  # upper right
    merged.paste(img_parts[2], (0, size // 2))  # lower left
    merged.paste(img_parts[3], (size // 2, size // 2))  # lower right

    return merged
