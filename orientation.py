import re

import numpy as np
from fastai.vision.all import PILImage
from PIL import Image
from skimage import transform, util


VIEWS = ("frontal", "lateral")
ANGLES = ("0", "180", "270", "90")
FLIPROT_CLASSES = tuple(
    f"{view}_{angle}{flip}"
    for view in VIEWS
    for angle in ANGLES
    for flip in ("", "F")
) + ("none",)
_FLIPROT_LABEL = re.compile(r"^(?:frontal|lateral)_(0|90|180|270)(F?)$")


def prepare_fliprot_prediction_image(image):
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = util.img_as_ubyte(arr)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape for prediction: {arr.shape}")
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        raise ValueError(f"Unsupported image shape for prediction: {arr.shape}")
    return PILImage.create(Image.fromarray(arr))


def decode_fliprot_label(label):
    if label == "none":
        return None
    match = _FLIPROT_LABEL.fullmatch(label)
    if match is None:
        raise ValueError(f"Unexpected fliprot prediction label: {label}")
    return "".join(match.groups())


def apply_fliprot_label(image, label):
    action = decode_fliprot_label(label)
    match action:
        case "0F":
            image = np.fliplr(image)
        case "90":
            image = transform.rotate(image, 270, resize=True)
        case "270":
            image = transform.rotate(image, 90, resize=True)
        case "180":
            image = transform.rotate(image, 180, resize=True)
        case "90F":
            image = np.fliplr(image)
            image = transform.rotate(image, 270, resize=True)
        case "180F":
            image = np.flipud(image)
        case "270F":
            image = np.fliplr(image)
            image = transform.rotate(image, 90, resize=True)
        case "0" | None:
            return False, image
    return True, image
