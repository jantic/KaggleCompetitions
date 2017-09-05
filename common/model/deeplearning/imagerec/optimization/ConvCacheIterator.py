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

from common.model.deeplearning.imagerec.optimization.TransparentDirectoryIterator import TransparentDirectoryIterator


class ConvCacheIterator(Iterator):
    def __init__(self, cache_directory: str, batches: DirectoryIterator, batch_id: str, conv_model: Sequential, batch_size=32, num_cache_parts=100):
        self.CONV_MODEL = conv_model
        self.SOURCE_BATCHES = batches
        self.CACHE_DIRECTORY = cache_directory
        ConvCacheIterator.__establish_directory_if_needed(cache_directory)
        self.BATCH_SIZE = batch_size
        self.LABELS = self.__onehot(batches.classes)
        self.BATCH_ID  = batch_id
        self.CURRENT_FILE_NUM = -1
        self.NUM_CACHE_PARTS = num_cache_parts
        self.STEPS_PER_EPOCH = int(np.ceil(batches.samples / batch_size / num_cache_parts))
        self.__generate_batch_data_cache_if_needed()
        self.__advance_to_next_cache_file()
        super(ConvCacheIterator, self).__init__(batches.samples, batch_size=batch_size, shuffle=False, seed=None)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
            return self.__get_batch_data_from_cache(index_array)

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

        transparent_batches = TransparentDirectoryIterator(self.SOURCE_BATCHES, record_labels)

        for cache_part_num in range(self.NUM_CACHE_PARTS):
            print('Caching model features for ' + self.BATCH_ID + ', part ' + str(cache_part_num+1) + ' out of ' + str(self.NUM_CACHE_PARTS))

            features_array = self.CONV_MODEL.predict_generator(transparent_batches, self.STEPS_PER_EPOCH, max_queue_size=1)
            features_cache_path = self.__generate_features_cache_path(cache_part_num)
            ConvCacheIterator.__save_array(features_cache_path, features_array)
            labels_array = np.zeros(tuple([len(features_array)] + list(batch_labels_list[0].shape)[1:]), dtype=image.K.floatx())
            index = 0

            #There's an extra batch pulled at the end, so don't use that as a set of labels.
            for i in range(len(batch_labels_list)-1):
                batch_labels = batch_labels_list[i]
                for label_pair in batch_labels:
                    labels_array[index]=label_pair
                    index=index+1

            batch_labels_list.clear()
            labels_cache_path = self.__generate_labels_cache_path(cache_part_num)
            ConvCacheIterator.__save_array(labels_cache_path, labels_array)



    def __get_batch_data_from_cache(self, index_array):
        batch_x = np.zeros(tuple([self.BATCH_SIZE] + list(self.CURRENT_FEATURE_ARRAY.shape)[1:]), dtype=image.K.floatx())
        batch_y = np.zeros(tuple([self.BATCH_SIZE] + list(self.CURRENT_LABEL_ARRAY.shape)[1:]), dtype=image.K.floatx())

        for i, absolute_batch_index in enumerate(index_array):
            if absolute_batch_index == 0:
                self.__reset_cache_reads()

            relative_batch_index = self.__calculate_relative_batch_index(absolute_batch_index)
            max_relative_index_current_array = len(self.CURRENT_FEATURE_ARRAY)-1

            while relative_batch_index > max_relative_index_current_array:
                self.__advance_to_next_cache_file()
                relative_batch_index = self.__calculate_relative_batch_index(absolute_batch_index)
                max_relative_index_current_array = len(self.CURRENT_FEATURE_ARRAY) - 1
                if relative_batch_index < 0:
                    print('\n---------------------wtf moment coming-----------------------')

                    print('\nbatch_id: ' + self.BATCH_ID + '; file_num:' + str(self.CURRENT_FILE_NUM) + '; max_relative_index_current_array: ' +\
                          str(max_relative_index_current_array) + '; absolute_batch_index: ' + str(absolute_batch_index) + '; total_entries_read_so_far: ' + \
                          str(self.TOTAL_ENTRIES_READ_SO_FAR) + '; current_feature_array_len: ' +  str(len(self.CURRENT_FEATURE_ARRAY)))


            batch_x[i] = self.CURRENT_FEATURE_ARRAY[relative_batch_index]
            batch_y[i] = self.CURRENT_LABEL_ARRAY[relative_batch_index]

        return batch_x, batch_y

    def __reset_cache_reads(self):
        self.CURRENT_FILE_NUM = -1
        self.__advance_to_next_cache_file()

    def __calculate_relative_batch_index(self, absolute_batch_index: int):
        return absolute_batch_index - (self.TOTAL_ENTRIES_READ_SO_FAR - len(self.CURRENT_FEATURE_ARRAY))

    def __advance_to_next_cache_file(self):
        self.CURRENT_FILE_NUM = self.__get_next_file_num(self.CURRENT_FILE_NUM)
        self.CURRENT_FEATURE_ARRAY = self.__load_feature_array(self.CURRENT_FILE_NUM)
        self.CURRENT_LABEL_ARRAY = self.__load_label_array(self.CURRENT_FILE_NUM)

        if self.CURRENT_FILE_NUM == 0:
            self.TOTAL_ENTRIES_READ_SO_FAR = 0

        self.TOTAL_ENTRIES_READ_SO_FAR =  self.TOTAL_ENTRIES_READ_SO_FAR + len(self.CURRENT_FEATURE_ARRAY)

    def __get_next_file_num(self, file_num: int):
        incremented_file_num = file_num + 1
        incremented_cache_path = self.__generate_features_cache_path(file_num=incremented_file_num)
        return incremented_file_num if os.path.exists(incremented_cache_path) else 0

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