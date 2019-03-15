import numpy as np
from imageio import imread, imwrite
import skimage.color as color

def shift_hue(img, amt):
    """
    Shift the hue of image.

    Inputs:
        img - Numpy-like image with three color channels.
        amt - The amount of hue shift to apply, in range [0.0, 1.0].
    Outputs:
        img - The output RGB image.
    """

    if not img.ndim == 3:
        raise ValueError('img must have 3 dimensions (has {}).'.format(img.ndim))

    if not img.shape[2] == 3:
        raise ValueError('Size of channel dimension must be 3 (shape of img: {})'.format(img.shape))

    img = color.rgb2hsv(img)
    img[:, :, 0] = (img[:, :, 0] + amt) / 1.0
    return color.hsv2rgb(img)

def shift_lightness(img, amt):
    """
    Shift the lightness of image.

    Inputs:
        img - Numpy-like image with three color channels.
        amt - The amount of lightness shift to apply, in range [-1.0, 1.0].
    Outputs:
        img - The output RGB image.
    """

    if not img.ndim == 3:
        raise ValueError('img must have 3 dimensions (has {}).'.format(img.ndim))

    if not img.shape[2] == 3:
        raise ValueError('Size of channel dimension must be 3 (shape of img: {})'.format(img.shape))

    img = color.rgb2hsv(img)
    img[:, :, 2] = np.clip(img[:, :, 2] + amt, 0.0, 1.0)
    return color.hsv2rgb(img)

def crop(img, left, top, right, bottom):
    """
    Crop rectangle from image.

    Inputs:
        img - The image to crop.
        left - The leftmost index to crop the image.
        top - The topmost index.
        right - The rightmost index.
        bottom - The bottommost index.
    Outputs:
        img - The cropped image.
    """

    return img[left:right, top:bottom]

def crop_center(img):
    """
    Crop the largest possible square from center of image.

    Inputs:
        img - The image to crop.
    Outputs:
        img - The cropped image.
    """
    
    dims = img.shape[0:2]
    short_axis, long_axis = np.argsort(dims)
    short_width = dims[short_axis]
    long_width = dims[long_axis]
    begin = (long_width - short_width) // 2
    
    img = img.take(range(begin, begin + short_width), axis=long_axis)
    return img

def convert_image(img_path, output_path, write_kwargs={}):
    """
    Load image in img_path and rewrite it to output_path, converting type if needed.

    Inputs:
        img_path - Path to image.
        output_path - Path to write image to (extension determines format).
        write_kwargs - kwargs to pass to the write function
    Outputs:
        None
    """

    img = imread(img_path)

    try: 
        imwrite(output_path, img, **write_kwargs)
        print("Image saved: " + output_path)
    except IOError:
        print("Conversion failed: " + img_path)
