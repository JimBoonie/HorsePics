# HorsePics

A simple image processing library for equines (and people too)

![Stitching](assets/Bojack.png)

This includes a handful of simple functions that can:

* Crop large images into a grid of (overlapping) images
* Stitch a grid of cropped images back into one large image
* Adjust brightness and hue of image
* Crop image
* Center crop image
* Convert file type

## Installation

First, clone the repo. Then, simply navigate to that folder and run:

```bash
# install repo
pip install .

# (optional) install tqdm for progress bars
pip install tqdm
```

Then you can use these utilities with a simple import:

```python
import horsepics
```
