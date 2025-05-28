from io import BytesIO
from PIL import Image 
import cairosvg
from xml.dom import minidom
from math import ceil


def display_svg(path):
    out = BytesIO()
    cairosvg.svg2png(url=path, write_to=out)
    return Image.open(out)


def svg_string_to_img(s):
    out = BytesIO()
    cairosvg.svg2png(bytestring=s, write_to=out)
    return Image.open(out)


def scale_to_maxdim(w, h, maxdim):
    if w > h:  # landscape format
        #   width == maxdim
        #   height scaled
        hw_ratio = h / w
        w_out = str(maxdim)
        h_out = str(ceil(maxdim * hw_ratio))
    else:  # portrait format or square
        #   width == maxdim * ratio
        #   height scaled (by 1 if square)
        wh_ratio = w / h
        w_out = str(ceil(maxdim * wh_ratio))
        h_out = str(maxdim)
    return w_out, h_out


def set_svg_attributes(s, maxdim='auto', fill='darkgray', stroke='white', strokewidth=1):

    doc = minidom.parseString(s)

    if maxdim != 'auto':
        # scale the full svg
        assert type(maxdim) is int, 'maxdim parameter has to be "auto" or type int'
        root = doc.getElementsByTagName('svg')[0]
        _, _, width, height = map(float, root.getAttribute('viewBox').split())
        w, h = scale_to_maxdim(width, height, maxdim)
        root.setAttribute('width', w)
        root.setAttribute('height', h)

    for element in doc.getElementsByTagName('polygon'):
        # set other attributes for individual polygons
        element.setAttribute('fill', fill)
        element.setAttribute('stroke', stroke)
        element.setAttribute('strokewidth', str(strokewidth))

    return doc.toxml()
