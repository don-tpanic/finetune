import os
import warnings
import threading
import multiprocessing
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array, apply_affine_transform, apply_channel_shift, apply_brightness_shift


class Iterator(Sequence):
    """
        An iterator inherits the keras.utils.Sequence as recommended for
        multiprocessing safety
    """
    white_list_formats = ('npy', 'png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff')

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def compute_step_size(self):
        if self.index_array is None:
            self._set_index_array()   # QUESTION:effect?
        len_of_index_array = len(self.index_array)
        step_size = np.ceil(len_of_index_array / self.batch_size)
        return step_size


    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)
    
    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            if self.n == 0:
                current_index = 0
            else:
                current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def next(self):
        """For python 2.x.
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.
        # Arguments
            index_array: Array of sample indices to include in batch.
        # Returns
            A batch of transformed samples.
        """
        raise NotImplementedError


class DirectoryIterator(Iterator):
    """
        Custom DirectoryIterator work with CustomIterator
        (flow_from_directory uses this class, where flow_from_directory is
        a function under class ImageDataGenerator)
    """
    allowed_class_modes = {'binary_feat3', 'binary_feat4', 'categorical', 'binary', 'sparse', 'input', None}

    def __init__(self,
                 directory,
                 class_indices,
                 target_size=(224, 224),
                 classes=None,
                 class_mode='sparse',
                 batch_size=16,
                 shuffle=False,
                 seed=None,
                 follow_links=False,
                 subset=None,
                 validation_split=0.0,
                 dtype=None,
                 color_mode='rgb',
                 data_format='channels_last',
                 preprocessing_function=None,
                 interpolation='nearest',
                 ):

        self.classes = classes
        self.validation_split = validation_split
        self.directory = directory
        self.color_mode = color_mode
        self.data_format = data_format
        self.target_size = tuple(target_size)
        self.preprocessing_function = preprocessing_function
        self.interpolation = interpolation

        if self.color_mode == 'rgb':
            if self.data_format == 'channels_last':
                self.image_shape = self.target_size + (3,)

        if subset is not None:
            validation_split = self.validation_split
            if subset == 'validation':
                split = (0, validation_split)
            elif subset == 'training':
                split = (validation_split, 1)
            else:
                raise ValueError(
                    'Invalid subset name: %s;'
                    'expected "training" or "validation"' % (subset,))
        else:
            split = None
        self.split = split
        self.subset = subset

        if validation_split and not 0 < validation_split < 1:
            raise ValueError(
                '`validation_split` must be strictly between 0 and 1. '
                ' Received: %s' % validation_split)

        if class_mode not in self.allowed_class_modes:
            raise ValueError('Invalid class_mode: {}; expected one of: {}'
                             .format(class_mode, self.allowed_class_modes))
        self.class_mode = class_mode
        self.dtype = dtype
        if dtype is None:
            self.dtype = K.floatx()


        # First, count the number of samples and classes.
        self.samples = 0
        if not classes:
            classes = []
            for subdir in sorted(os.listdir(directory)):
                if os.path.isdir(os.path.join(directory, subdir)):
                    # classes here is a list of all wnids
                    classes.append(subdir)
        self.num_classes = len(classes)
        # mapping from wnid to class indices as integers
        # NOTE(ken): for heldout to work, we replace inferred using manual passing.
        # self.class_indices = dict(zip(classes, range(len(classes))))
        self.class_indices = class_indices


        pool = multiprocessing.pool.ThreadPool()
        # Second, build an index of the images
        # in the different class subfolders.
        results = []
        self.filenames = []
        i = 0
        for dirpath in (os.path.join(directory, subdir) for subdir in classes):
            results.append(
                pool.apply_async(_list_valid_filenames_in_directory,
                                 (dirpath, self.white_list_formats, self.split,
                                  self.class_indices, follow_links)))
        classes_list = []
        for res in results:
            classes, filenames = res.get()
            classes_list.append(classes)
            self.filenames += filenames
        self.samples = len(self.filenames)
        self.classes = np.zeros((self.samples,), dtype='int32')

        # now the self.classes is a list of integer class indices
        for classes in classes_list:
            self.classes[i:i + len(classes)] = classes
            i += len(classes)

        pool.close()
        pool.join()
        self._filepaths = [
            os.path.join(self.directory, fname) for fname in self.filenames
        ]

        super(DirectoryIterator, self).__init__(self.samples,
                                                    batch_size,
                                                    shuffle,
                                                    seed,
                                                    )

    @property
    def filepaths(self):
        return self._filepaths

    @property
    def labels(self):
        return self.classes

    @property  # mixin needs this property to work
    def sample_weight(self):
        # no sample weights will be returned
        return None

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.
        # Arguments
            index_array: Array of sample indices to include in batch.
        # Returns
            A batch of transformed samples.
        """
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=self.dtype)
        # build batch of image data
        # self.filepaths is dynamic, is better to call it once outside the loop
        filepaths = self.filepaths
        for i, j in enumerate(index_array):

            # NOTE(ken): if the images are augmented
            # but not preprocessed, like the .png we 
            # have for task6, then we preprocess here.
            if self.preprocessing_function:
                img = load_img(filepaths[j],
                           color_mode=self.color_mode,
                           target_size=self.target_size,
                           interpolation=self.interpolation)
                x = img_to_array(img, data_format=self.data_format)
                # Pillow images should be closed after `load_img`,
                # but not PIL images.
                if hasattr(img, 'close'):
                    img.close()
                x = self.preprocessing_function(x)
            # (ken) Otherwise, means we have the preprocessed
            # images saved as .npy
            # we had this branch due to `full_train` where we 
            # need to train the entire model for task1.
            # generator is really not needed if finetune 1 layer.
            else:
                x = np.load(filepaths[j])
            
            batch_x[i] = x

        # build batch of labels
        if self.class_mode == 'input':
            batch_y = batch_x.copy()

        # NOTE(ken): after maunual sub from label to binary feature
        # this will fail because batch_y expects ints but 
        # are given lists of 0/1
        elif self.class_mode in {'binary', 'sparse'}:
            batch_y = np.empty(len(batch_x), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]

        # NOTE(ken): hence we create another class_mode
        elif self.class_mode in {'binary_feat3', 'binary_feat4'}:
            # TODO: a bit ad hoc but good for now.
            batch_y = np.empty((len(batch_x), int(self.class_mode[-1])), dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i] = self.classes[n_observation]

        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), len(self.class_indices)),
                               dtype=self.dtype)
            for i, n_observation in enumerate(index_array):
                batch_y[i, self.classes[n_observation]] = 1.
        elif self.class_mode == 'multi_output':
            batch_y = [output[index_array] for output in self.labels]
        elif self.class_mode == 'raw':
            batch_y = self.labels[index_array]
        else:
            return batch_x
        
        if self.sample_weight is None:
            return batch_x, batch_y
        else:
            return batch_x, batch_y, self.sample_weight[index_array]


def _count_valid_files_in_directory(directory,
                                    white_list_formats,
                                    split,
                                    follow_links):
    num_files = len(list(
        _iter_valid_files(directory, white_list_formats, follow_links)))
    if split:
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
    else:
        start, stop = 0, num_files
    return stop - start


def _list_valid_filenames_in_directory(directory, white_list_formats, split,
                                       class_indices, follow_links):
    dirname = os.path.basename(directory)
    if split:
        num_files = len(list(
            _iter_valid_files(directory, white_list_formats, follow_links)))
        start, stop = int(split[0] * num_files), int(split[1] * num_files)
        valid_files = list(
            _iter_valid_files(
                directory, white_list_formats, follow_links))[start: stop]
    else:
        valid_files = _iter_valid_files(
            directory, white_list_formats, follow_links)

    classes = []
    filenames = []
    for root, fname in valid_files:
        classes.append(class_indices[dirname])
        absolute_path = os.path.join(root, fname)
        relative_path = os.path.join(
            dirname, os.path.relpath(absolute_path, directory))
        filenames.append(relative_path)

    return classes, filenames


def _iter_valid_files(directory, white_list_formats, follow_links):
    def _recursive_list(subpath):
        return sorted(os.walk(subpath, followlinks=follow_links),
                      key=lambda x: x[0])

    for root, _, files in _recursive_list(directory):
        for fname in sorted(files):
            for extension in white_list_formats:
                if fname.lower().endswith('.tiff'):
                    warnings.warn('Using \'.tiff\' files with multiple bands '
                                  'will cause distortion. '
                                  'Please verify your output.')
                if fname.lower().endswith('.' + extension):
                    yield root, fname