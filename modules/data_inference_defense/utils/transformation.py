import random

import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageOps


def img_zoom(img):
    pad_len = random.randint(8, 16)
    img1 = transforms.RandomCrop(32+pad_len, padding=pad_len)(img)
    img2 = transforms.Resize(32)(img1)
    img2 = np.array(img2)
    mask = (img2 == 0) * 1
    img2 = (1 - mask) * img2 + mask * (np.random.random(img2.shape) * 255).astype(np.uint8)
    img2 = Image.fromarray(img2.astype('uint8')).convert('RGB')
    return img2

def img_warp(img):
    img1 = transforms.RandomPerspective(distortion_scale=0.3, p=1)(img)
    return img1

ranges = {
    "shearX": np.linspace(0, 0.3, 10),
    "shearY": np.linspace(0, 0.3, 10),
    "translateX": np.linspace(0, 150 / 331, 10),
    "translateY": np.linspace(0, 150 / 331, 10),
    "rotate": np.linspace(0, 30, 10),
    "color": np.linspace(0.0, 0.5, 10),
    "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
    "solarize": np.linspace(256, 0, 10),
    "contrast": np.linspace(0.0, 0.5, 10), #
    "sharpness": np.linspace(0.0, 0.5, 10), #
    "brightness": np.linspace(0.0, 0.5, 10), #
    "autocontrast": [0] * 10,
    "equalize": [0] * 10,
    "invert": [0] * 10,
    "zoom" : [0] * 10,
    "warp": [0] * 10,
}

# from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
def rotate_with_fill(img, magnitude):
    rot = img.convert("RGBA").rotate(magnitude)
    return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)


class Transform(object):
    def __init__(self, p1, operation1, magnitude_idx1, fillcolor=(0, 0, 0)):
        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img),
            "zoom": lambda img, magnitude: img_zoom(img),
            "warp": lambda img, magnitude: img_warp(img),
        }
        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]


    def __call__(self, img):
        img = self.operation1(img, self.magnitude1)
        return img

transformations = [
    Transform(0.1, "invert", 7),
    Transform(0.2, "contrast", 6),
    Transform(0.7, "rotate", 2),
    Transform(0.3, "translateX", 9),
    Transform(0.8, "sharpness", 1),

    Transform(0.9, "sharpness", 3),
    Transform(0.5, "shearY", 2),
    Transform(0.7, "translateY", 2) ,
    Transform(0.5, "autocontrast", 5),
    Transform(0.9, "equalize", 2), #

    Transform(0.2, "shearY", 5),
    Transform(0.3, "posterize", 5), #
    Transform(0.4, "color", 3),
    Transform(0.6, "brightness", 5), #
    Transform(0.3, "sharpness", 9),

    Transform(0.7, "brightness", 9),
    Transform(0.6, "equalize", 5),
    Transform(0.5, "equalize", 1),
    Transform(0.6, "contrast", 7),
    Transform(0.6, "sharpness", 5),

    Transform(0.7, "color", 5),
    Transform(0.5, "translateX", 5), #
    Transform(0.3, "equalize", 7),
    Transform(0.4, "autocontrast", 8),
    Transform(0.4, "translateY", 3),
    Transform(0.2, "sharpness", 6),
    Transform(0.9, "brightness", 6),
    Transform(0.2, "color", 8),
    Transform(0.5, "solarize", 0),
    Transform(0.0, "invert", 0), #
    Transform(0.2, "equalize", 0),
    Transform(0.6, "autocontrast", 0), #
    Transform(0.2, "equalize", 8),
    Transform(0.6, "equalize", 4),
    Transform(0.9, "color", 5),
    Transform(0.6, "equalize", 5), #
    Transform(0.8, "autocontrast", 4),
    Transform(0.2, "solarize", 4), #
    Transform(0.1, "brightness", 3),
    Transform(0.7, "color", 0),
    Transform(0.4, "solarize", 1),
    Transform(0.9, "autocontrast", 0), #
    Transform(0.9, "translateY", 3),
    Transform(0.7, "translateY", 3), #
    Transform(0.9, "autocontrast", 1),
    Transform(0.8, "solarize", 1), #
    Transform(0.8, "equalize", 5),
    Transform(0.1, "invert", 0),  #
    Transform(0.7, "translateY", 3),
    Transform(0.9, "autocontrast", 1),
]

if __name__ == '__main__':
    print(len(transformations))
