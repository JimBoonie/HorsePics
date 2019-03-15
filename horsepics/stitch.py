import os, warnings, errno
import numpy as np
from imageio import imread, imwrite

VALID_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp',)

def _list_files(loc, return_dirs=False, return_files=True, recursive=False, valid_exts=None):
    """
    Return a list of all filenames within a directory loc.

    Inputs:
        loc - Path to directory to list files from.
        return_dirs - If true, returns directory names in loc. (default: False)
        return_files - If true, returns filenames in loc. (default: True)
        recursive - If true, searches directories recursively. (default: False)
        valid_exts - If a list, only returns files with extensions in list. If None,
            does nothing. (default: None)

    Outputs:
        files - List of names of all files and/or directories in loc.
    """
    
    files = [os.path.join(loc, x) for x in os.listdir(loc)]

    if return_dirs or recursive:
        # check if file is directory and add it to output if so
        is_dir = [os.path.isdir(x) for x in files]
        found_dirs = [files[x] for x in range(len(files)) if is_dir[x]]
    else:
        found_dirs = []

    if return_files:
        # check if file is not directory and add it to output
        is_file = [os.path.isfile(x) for x in files]
        found_files = [files[x] for x in range(len(files)) if is_file[x]]
    else:
        found_files = []

    if recursive and not return_dirs:
        new_dirs = []
    else:
        new_dirs = found_dirs

    deeper_files = []
    if recursive:
        for d in found_dirs:
            deeper_files.extend(_list_files(d, 
                                           return_dirs=return_dirs, 
                                           return_files=return_files,
                                           recursive=recursive))

    if isinstance(valid_exts, (list, tuple)):
        concat_files = found_files + deeper_files
        new_files = []
        for e in valid_exts:
            new_files.extend([f for f in concat_files if f.endswith(e)])
    else:
        new_files = found_files + deeper_files

    return new_dirs + new_files

def get_nested_dirs(locs=['.']):
    """
    Return a list of all folders and subfolders within the directories in the list
    locs.

    Inputs:
        locs - String of path or list of paths to directories to search.

    Outputs:
        dirs - List of paths to all subfolders of locs.
    """

    # if locs is not a list, place it in one
    if not isinstance(locs, (list, tuple)):
        locs = [locs]

    dirs = []
    for loc in locs:
        # get directories with full path
        new_dirs = _list_files(loc, return_files=False, return_dirs=True)

        # traverse directories to the bottom
        nested_dirs = get_nested_dirs(new_dirs)
        dirs.extend(new_dirs)
        dirs.extend(nested_dirs)

    return dirs

def grid_crop(img, crop_dims, stride_size=None, include_excess=True):
    """
    Split the image into tiles of smaller images.

    Since the image dimensions may not be perfect multiples of the crop_dims, the 
    edge portions of the image are included as crops positioned at img.size - 
    crop_size. This results in the edge images overlapping slightly by default. This
    behavior can be overridden by setting 'include_excess' to False.

    Inputs:
        img - Numpy-like image.
        crop_dims - The dimensions of each cropped image.
        stride_size - The offset between each cropped image, in pixels. (Default: 
                    crop_dims)
        include_excess - When true, includes extra crops to include edges that would
                         exceed the largest multiple of crop_dims within the image.
    Outputs:
        crop_imgs - List of crop dicts containing img and other keys.
    """

    def _grid_crop_corners(im_dims, crop_size, stride_size=None, include_excess=True):
        if stride_size is None:
            stride_size = crop_size

        assert(len(crop_size) == 2 
               and len(stride_size) == 2)

        r_indices = list(range(0, im_dims[0] - crop_size[0], stride_size[0]))
        c_indices = list(range(0, im_dims[1] - crop_size[1], stride_size[1]))
        if include_excess:
            r_indices.append(im_dims[0] - crop_size[0])
            c_indices.append(im_dims[1] - crop_size[1])

        crop_corners = []
        for r in r_indices:
            for c in c_indices:
                crop_corners.append((r, c))
        return crop_corners

    img_dims = img.shape # NOTE: (rows, cols)
    crop_corners = _grid_crop_corners(img_dims, crop_dims, stride_size, include_excess)

    # loop through crop_corners and create crop for each
    crop_imgs = []
    for corner in crop_corners:
        idxs = (corner[0], corner[0] + crop_dims[0],
                corner[1], corner[1] + crop_dims[1])
        try:
            crop_img = img[idxs[0]:idxs[1], idxs[2]:idxs[3], :]
        except IndexError:
            crop_img = img[idxs[0]:idxs[1], idxs[2]:idxs[3]]
        crop = {'img': crop_img,
                'corner': corner}
        crop_imgs.append(crop)

    return crop_imgs

def stitch_crops(crop_imgs, method='linear_average', method_args={}):
    """
    Merge a list of regularly-spaced cropped images into one single image.

    Inputs:
        crop_imgs - List of crop dicts containing the following keys:
            'img' - Numpy array with image.
            'corner' - Tuple specifying the location of the upper-left
                       corner of the image.
        method - Blend method to combine two images. Options are:
            'linear_average' (Default)
            'sigmoid_average'
            'max'
            'average'
            'and'
            'or'
            'xor'
        method_args - Dict containing keyword args to use with method. Options are:
            'sigmoid_average':
                'k' - Sets spread of sigmoid function along x-axis. (Default: 12)
    Outputs:
        img - Numpy array with stitched image.
    """

    def _stitch_row(crop_row, method, method_args={}):
        row_img_shape = [crop_row[-1]['img'].shape[0],
                         crop_row[-1]['corner'][1] + crop_row[-1]['img'].shape[1]]
        try:
            n_channels = crop_row[-1]['img'].shape[2]
        except IndexError:
            n_channels = 1

        if n_channels == 1:
            row_img = np.zeros((row_img_shape[0], row_img_shape[1]), dtype=np.float64)
        else: 
            row_img = np.zeros((row_img_shape[0], row_img_shape[1], n_channels), dtype=np.float64)

        prev_width = 0
        for i in range(len(crop_row) - 1):
            # create overlap_img
            crop1 = crop_row[i]
            crop2 = crop_row[i + 1]
            overlap = [
                crop2['corner'][1], 
                crop1['corner'][1] + crop1['img'].shape[1]
            ]
            overlap_width = overlap[1] - overlap[0]

            x_vec = np.linspace(0, 1, overlap_width)
            if method == 'linear_average':
                weights = x_vec
            elif method == 'sigmoid_average':
                try:
                    k = method_args['k']
                except KeyError:
                    k = 12
                weights = 1.0 / (1.0 + np.exp(-k * (x_vec - 0.5)))
            else:
                weights = x_vec

            # add singleton dimension for broadcasting
            if n_channels == 1:
                weights = weights[None, :]
            else:
                weights = weights[None, :, None]

            overlap_img = (weights * crop2['img'][:, 0:overlap_width] 
                         + weights[:, ::-1] * crop1['img'][:, -overlap_width:])

            # blend images into row_img
            crop1_rng = (crop1['corner'][1] + prev_width, overlap[0])
            crop2_rng = (overlap[1], crop2['corner'][1] + crop2['img'].shape[1])
            row_img[:, crop1_rng[0]:crop1_rng[1]] = crop1['img'][:, prev_width:-overlap_width]
            row_img[:, crop2_rng[0]:crop2_rng[1]] = crop2['img'][:, overlap_width:]
            row_img[:, overlap[0]:overlap[1]] = overlap_img
            prev_width = overlap_width

        span = [
            crop_row[-1]['corner'][0], 
            crop_row[-1]['corner'][0] + crop_row[-1]['img'].shape[0]
        ]
        return row_img, span

    # determine type and number of channel of crops
    mode = crop_imgs[0]['img'].dtype
    try:
        n_channels = crop_imgs[0]['img'].shape[2]
    except IndexError:
        n_channels = 1

    # check that all crops have same number of channels
    for crop in crop_imgs:
        try:
            crop_n_channel = crop['img'].shape[2]
        except IndexError:
            crop_n_channel = 1

        if crop_n_channel != n_channels:
            raise Exception("Number of channels is not consistent between images.")

    # check that all crops are of same type
    for crop in crop_imgs:
        if crop['img'].dtype != mode:
            warnings.warn("Data types of images are not consistent. May produce \
                           unpredictable results.")
            break

    # find dimensions of original image
    max_dims = [0, 0]
    max_corner = [0, 0]
    for crop in crop_imgs:
        if crop['img'].shape[0] > max_dims[0]:
            max_dims[0] = crop['img'].shape[0]

        if crop['img'].shape[1] > max_dims[1]:
            max_dims[1] = crop['img'].shape[1]

        if crop['corner'][0] > max_corner[0]:
            max_corner[0] = crop['corner'][0]

        if crop['corner'][1] > max_corner[1]:
            max_corner[1] = crop['corner'][1]
    img_dims = (max_corner[0] + max_dims[0], max_corner[1] + max_dims[1])

    # create numpy array to hold crops
    try:
        if n_channels == 1:
            img = np.zeros((img_dims[0], img_dims[1]), dtype=int)
        else:
            img = np.zeros((img_dims[0], img_dims[1], n_channels), dtype=int)
    except MemoryError:
        raise MemoryError('Failed to create image with dimensions [{}, {}, {}]'.format(
            img_dims[0], img_dims[1], n_channels))

    # stitch image into numpy array
    if method == 'linear_average' or method == 'sigmoid_average':
        # sort crops by corner position
        crop_imgs = sorted(crop_imgs, key=lambda x: x['corner'][0])
        unique_row_idxs, crop_idxs = np.unique(
            [i['corner'][0] for i in crop_imgs], 
            return_index=True
        )

        # group crops into rows by corner position
        n_rows = len(unique_row_idxs)
        crop_rows = []
        for i in range(n_rows):
            i_first = crop_idxs[i]
            if i + 1 < n_rows:
                i_last = crop_idxs[i + 1]
                crop_rows.append(crop_imgs[i_first:i_last])
            else:
                crop_rows.append(crop_imgs[i_first:])
            
        # fuse images, first by column, then by row
        prev_ht = 0
        for i in range(n_rows - 1):
            # get row imgs
            crop_row1 = crop_rows[i]
            crop_row2 = crop_rows[i + 1]
            row_img1, span1 = _stitch_row(crop_row1, method, method_args)
            row_img2, span2 = _stitch_row(crop_row2, method, method_args)

            # create overlap_img
            overlap = [span2[0], span1[1]]
            overlap_ht = overlap[1] - overlap[0]

            x_vec = np.linspace(0, 1, overlap_ht)
            if method == 'linear_average':
                weights = x_vec
            elif method == 'sigmoid_average':
                try:
                    k = method_args['k']
                except KeyError:
                    k = 12
                weights = 1.0 / (1.0 + np.exp(-k * (x_vec - 0.5)))
            else:
                weights = x_vec

            if n_channels == 1:
                weights = weights[:, None]
            else:
                weights = weights[:, None, None]

            overlap_img = (weights * row_img2[0:overlap_ht]
                         + weights[::-1] * row_img1[-overlap_ht:])

            # blend images into img
            row1_rng = [span1[0] + prev_ht, span2[0]]
            row2_rng = [span1[1], span2[1]]
            img[row1_rng[0]:row1_rng[1]] = row_img1[prev_ht:-overlap_ht]
            img[row2_rng[0]:row2_rng[1]] = row_img2[overlap_ht:]
            img[overlap[0]:overlap[1]] = overlap_img
            prev_ht = overlap_ht
    else:
        for crop in crop_imgs:
            idxs = (crop['corner'][0], crop['corner'][0] + crop['img'].shape[0], 
                    crop['corner'][1], crop['corner'][1] + crop['img'].shape[1])
            img_section = img[idxs[0]:idxs[1], idxs[2]:idxs[3], :]

            if len(crop['img'].shape) < 3:
                crop['img'] = np.expand_dims(crop['img'], 2)

            # blend method
            if method == 'or':
                crop_merged = np.bitwise_or(img_section, crop['img'])
            elif method == 'and':
                crop_merged = np.bitwise_and(img_section, crop['img'])
            elif method == 'xor':
                crop_merged = np.bitwise_xor(img_section, crop['img'])
            elif method == 'average':
                crop_merged = (img_section + crop['img']) / 2.0
            elif method == 'max' or method == 'maximum':
                crop_merged = np.maximum(img_section, crop['img'])
            else:
                warnings.warn("Invalid method: '%s'. Reverting to 'average'." 
                              % method, UserWarning)
                crop_merged = (img_section + crop['img']) / 2.0

            img[idxs[0]:idxs[1], idxs[2]:idxs[3], :] = crop_merged

    return img

def grid_crop_images(src_dir, dest_dir, crop_dims, recursive=True, stride_size=None, 
                   output_ext=".jpg", write_kwargs={}, verbose=False):
    """
    Apply grid_crop to all images within a given list.

    Inputs:
        src_dir - Directory where images are stored.
        dest_dir - Directory to output folders of subcropped images (preserves 
                   structure of source directory).
        crop_dims - The dimensions of each cropped image.
        recursive - If true, searches src_dir and all child folders.
        stride_size - The offset between each cropped image, in pixels. (Default: 
                      crop_dims)
        output_ext - Extension to save output files in.
        write_kwargs - kwargs for imageio.imwrite.
        verbose - If true, prints status updates.
    Outputs:
        None
    """

    # get all files
    file_list = _list_files(src_dir, recursive=recursive, valid_exts=VALID_EXTS)

    if verbose:
        print("number of imgs: %d" % len(file_list))

    # subcrop images to directory
    for img_path in file_list:
        # create destination if does not exist
        full_d = os.path.join(dest_dir, img_path.split('/')[-1])
        try: 
            os.makedirs(full_d)
        except OSError as e:
            print(e)
            if e.errno != errno.EEXIST:
                raise

        if verbose:
            print("Reading image: %s" % img_path)
        img = imread(img_path)

        crops = grid_crop(img, crop_dims, stride_size)

        if verbose:
            print("Saving %d subcrops in: %s" % (len(crops), full_d))
        for crop in crops:
            r = crop['corner'][0]
            c = crop['corner'][1]
            filename = "crop-" + str(r) + '-' + str(c) + output_ext
            filepath = os.path.join(full_d, filename)

            try:
                imwrite(filepath, crop['img'], **write_kwargs)
            except IOError:
                print("cannot convert: ", filepath)

def stitch_images(src_dir, dest_dir, recursive=True, output_ext='.jpg', method='linear_average', method_args={}, delimiter='-', zero_index=True, verbose=False):
    """
    Apply stitch_crops to all images grouped within a set of directories.

    Assumes that each directory in src_dir corresponds to its own image. Each image
    in the directories contain images with names of the form "crop-r-c.ext" where 
    "r" and "c" are integers giving the row-column indices of the image's upper-left
    corner in the original image. This input method is modeled to complement the 
    output of the grid_crop_images function.

    Inputs:
        src_dir - Directory where image directories (one directory -> one image).
        dest_dir - Directory to output stitched images (preserves structure of 
            source directory).
        recursive - If true, searches src_dir and all child folders.
        output_ext - Extension to save output files in.
        method - Stitching method (see stitch_crops).
        delimiter - Character that separates the row and column indices in the 
            titles of the images.
        zero_index - If true, assumes row-column indices start from 0. Otherwise,
            assumes 1.
        verbose - If true, prints status updates.
    Outputs:
        None
    """

    # get all directories within src_dir
    dir_list = _list_files(src_dir, 
                          return_dirs=True, 
                          return_files=False, 
                          recursive=recursive)

    if verbose:
        print("number of dirs: %d" % len(dir_list))

    # subcrop images to directory
    len_output_ext = len(output_ext)
    for d in dir_list:
        if verbose:
            print("Current dir: %s" % d)

        # create destination if does not exist
        imgname_noext = os.path.splitext(d.split('/')[-1])[0]
        filepath = os.path.join(dest_dir, imgname_noext)

        try: 
            os.makedirs(os.path.dirname(filepath))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        img_list = _list_files(d)
        if verbose:
            print("Number of imgs: %d" % len(img_list))

        if len(img_list) == 0:
            continue

        crop_list = []
        for img_path in img_list:
            img_name = os.path.basename(img_path)
            r = int(img_name.split(delimiter)[-2])
            c = int(img_name.split(delimiter)[-1][:-len_output_ext])

            if not zero_index:
                r = r - 1
                c = c - 1

            img = imread(img_path)
            crop_dict = {'img': img,
                         'corner': (r, c)}
            crop_list.append(crop_dict)

        img = stitch_crops(crop_list, method=method, method_args=method_args)
        
        out_path = filepath + output_ext
        try:
            imwrite(out_path, img)
        except IOError:
            print("cannot convert", img)
