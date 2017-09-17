from keras.preprocessing.image import array_to_img, NumpyArrayIterator, Iterator, DirectoryIterator
import numpy as np
import os
import threading
import warnings
import multiprocessing.pool
from functools import partial
from keras.models import Sequential
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
import bcolz
from collections import deque
import time
import threading


from common.model.deeplearning.imagerec.optimization.CachedTrainingPair import CachedTrainingPair
from common.model.deeplearning.imagerec.optimization.TransparentDirectoryIterator import TransparentDirectoryIterator


class ConvCacheIterator(Iterator):
    def __init__(self, cache_directory: str, batches: DirectoryIterator, batch_id: str, conv_model: Sequential, batch_size=32,
                 shuffle=False, seed=None, steps_per_file=20):
        self.FILE_NUM_ARRAY = np.array([])
        self.batch_index = 0
        self.FILE_NUM_LOCK = threading.Lock()
        self.FILE_QUEUE_APPEND_LOCK = threading.Lock()
        self.FILE_QUEUE_SIZE = 3
        self.SHUFFLE = shuffle
        self.CONV_MODEL = conv_model
        self.SOURCE_BATCHES = batches
        self.CACHE_DIRECTORY = cache_directory
        ConvCacheIterator.__establish_directory_if_needed(cache_directory)
        self.BATCH_SIZE = batch_size
        self.LABELS = self.__onehot(batches.classes)
        self.BATCH_ID = batch_id
        self.STEPS_PER_FILE = steps_per_file
        self.NUM_ITEMS_IN_BATCHES = batches.samples
        self.n = self.NUM_ITEMS_IN_BATCHES
        self.NUM_CACHE_PARTS = int(np.ceil(self.NUM_ITEMS_IN_BATCHES / self.BATCH_SIZE / self.STEPS_PER_FILE))
        self.__generate_batch_data_cache_if_needed()
        self.CACHE_FILE_QUEUE = deque(maxlen=self.FILE_QUEUE_SIZE)
        self.__start_file_load_daemon_threads()
        super(ConvCacheIterator, self).__init__(0, batch_size=batch_size, shuffle=shuffle, seed=seed)

    def next(self):
        with self.lock:
            return next(self.index_generator)

    def __start_file_load_daemon_threads(self):
        for thread_num in range(self.FILE_QUEUE_SIZE):
            thread = threading.Thread(target=self.__file_queue_populator_thread, args=())
            thread.daemon = True
            thread.start()

        while len(self.CACHE_FILE_QUEUE) < self.FILE_QUEUE_SIZE:
            time.sleep(0.01)

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        self.reset()
        self.__advance_to_next_cache_file()
        num_entries_in_file = self.__get_num_entries_in_current_file()
        index_array = self.__generate_index_array(shuffle=shuffle, num_entries_in_file=num_entries_in_file, seed=seed)

        while 1:
            num_entries_in_file =  self.__get_num_entries_in_current_file()

            current_index = current_index if num_entries_in_file == 0 else (self.batch_index * batch_size) % num_entries_in_file
            if num_entries_in_file > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                self.batch_index = 0
                self.__advance_to_next_cache_file()
                num_entries_in_file = self.__get_num_entries_in_current_file()

                while num_entries_in_file == 0:
                    self.__advance_to_next_cache_file()
                    num_entries_in_file = self.__get_num_entries_in_current_file()

                current_batch_size = num_entries_in_file - current_index

            if self.batch_index == 0 or index_array is None or len(index_array) == 0:
                index_array = self.__generate_index_array(shuffle=shuffle, num_entries_in_file=num_entries_in_file, seed=seed)

            self.total_batches_seen += 1
            batch_index_array = index_array[current_index: current_index + current_batch_size]
            batch_x = self.CURRENT_FEATURE_ARRAY[batch_index_array]
            batch_y = self.CURRENT_LABEL_ARRAY[batch_index_array]

            if len(batch_x) == 0 or len(batch_y) == 0:
                continue

            yield (batch_x, batch_y)

    def __generate_index_array(self, shuffle: bool, num_entries_in_file: int, seed):
        if seed is not None:
            np.random.seed(seed + self.total_batches_seen)

        if shuffle:
            return np.random.permutation(num_entries_in_file)
        else:
            return np.arange(num_entries_in_file)

    #TODO:  make this smarter eventually
    def __cache_exists(self):
        cache_path = self.__generate_features_cache_path(file_num=0)
        return os.path.exists(cache_path)

    def __generate_batch_data_cache_if_needed(self):
        if self.__cache_exists():
            return

        batch_labels_list = []

        def record_labels(batch_x, batch_y):
            batch_labels_list.append(batch_y)

        num_samples_cached = 0
        transparent_batches = TransparentDirectoryIterator(self.SOURCE_BATCHES, record_labels)

        for cache_part_num in range(self.NUM_CACHE_PARTS):
            print('Caching model features for ' + self.BATCH_ID + ', part ' + str(cache_part_num+1) + ' out of ' + str(self.NUM_CACHE_PARTS))
            num_items_remaining = self.NUM_ITEMS_IN_BATCHES - num_samples_cached
            features_array_raw = self.CONV_MODEL.predict_generator(transparent_batches, self.STEPS_PER_FILE, max_queue_size=1)
            #only take the number of items needed to match total number of samples in source batch
            num_items_to_fetch = min(len(features_array_raw), num_items_remaining)
            features_array = features_array_raw[:num_items_to_fetch]
            features_cache_path = self.__generate_features_cache_path(cache_part_num)
            ConvCacheIterator.__save_array(features_cache_path, features_array)
            features_array_length = len(features_array)
            labels_array = np.zeros(tuple([features_array_length] + list(batch_labels_list[0].shape)[1:]), dtype=image.K.floatx())
            index = 0

            for batch_labels in batch_labels_list:
                for label_pair in batch_labels:
                    if index >= features_array_length:
                        break
                    labels_array[index]=label_pair
                    index=index+1
                    num_samples_cached=num_samples_cached+1

            print(self.BATCH_ID + ': Num cached: ' + str(num_samples_cached) + ' vs num samples: '
                  + str(self.NUM_ITEMS_IN_BATCHES) + ' vs steps per file: ' + str(self.STEPS_PER_FILE))
            batch_labels_list.clear()
            labels_cache_path = self.__generate_labels_cache_path(cache_part_num)
            ConvCacheIterator.__save_array(labels_cache_path, labels_array)
            transparent_batches.mark_last_batch_skipped()


    def __get_num_entries_in_current_file(self):
        return len(self.CURRENT_FEATURE_ARRAY)

    def __advance_to_next_cache_file(self):
        while len(self.CACHE_FILE_QUEUE) == 0:
            time.sleep(0.01)

        array_pair = self.CACHE_FILE_QUEUE.popleft()

        if(self.BATCH_ID == 'validation'):
            self.CURRENT_FEATURE_ARRAY = array_pair.get_feature_array()
            self.CURRENT_LABEL_ARRAY = array_pair.get_label_array()
            x = 1;
        else:
            self.CURRENT_FEATURE_ARRAY = array_pair.get_feature_array()
            self.CURRENT_LABEL_ARRAY = array_pair.get_label_array()


    def __file_queue_populator_thread(self):
        while True:
            file_num = self.__get_next_file_num()
            feature_array = self.__load_feature_array(file_num)
            label_array = self.__load_label_array(file_num)
            array_pair = CachedTrainingPair(feature_array=feature_array, label_array=label_array)

            with self.FILE_QUEUE_APPEND_LOCK:
                while len(self.CACHE_FILE_QUEUE) == self.CACHE_FILE_QUEUE.maxlen:
                    time.sleep(0.01)
                self.CACHE_FILE_QUEUE.append(array_pair)

    def __get_next_file_num(self):
        with self.FILE_NUM_LOCK:
            #TODO:  Is this causing miscounts of how many samples there are?  Not sure yet....

            if len(self.FILE_NUM_ARRAY) == 0:
                if self.SHUFFLE:
                    self.FILE_NUM_ARRAY = np.random.permutation(self.NUM_CACHE_PARTS)
                else:
                    self.FILE_NUM_ARRAY = np.arange(self.NUM_CACHE_PARTS)

            file_num, self.FILE_NUM_ARRAY = self.FILE_NUM_ARRAY[0], self.FILE_NUM_ARRAY[1:]
            return file_num

    def __load_feature_array(self, file_num: int):
        cache_path = self.__generate_features_cache_path(file_num=file_num)
        feature_array = self.__load_array(cache_path)
        return feature_array

    def __load_label_array(self, file_num: int):
        cache_path = self.__generate_labels_cache_path(file_num=file_num)
        label_array = self.__load_array(cache_path)
        return label_array

    def __onehot(self, x):
        return to_categorical(x)

    def __load_array(self, fname:str):
        return bcolz.open(fname)[:]

    @staticmethod
    def __save_array(fname, arr):
        c = bcolz.carray(arr, rootdir=fname, mode='w')
        c.flush()

    @staticmethod
    def __establish_directory_if_needed(directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def __generate_features_cache_path(self, file_num: int):
        return self.CACHE_DIRECTORY + '/' + self.BATCH_ID + '_convlayer_features_' + str(file_num) + '_.bc'

    def __generate_labels_cache_path(self, file_num: int):
        return self.CACHE_DIRECTORY+ '/' + self.BATCH_ID  +  ' _convlayer_labels_' + str(file_num) + '_.bc'