from keras.preprocessing.image import DirectoryIterator

#path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'
class TransparentDirectoryIterator(DirectoryIterator):
    def __init__(self, source_directory_iterator: DirectoryIterator, next_callback):
        self.NEXT_CALL_BACKS = []
        self.NEXT_CALL_BACKS.append(next_callback)
        self.REUSE_PREVIOUS_BATCH = False
        self.PREVIOUS_BATCH = None


        super(TransparentDirectoryIterator, self).__init__(directory=source_directory_iterator.directory, target_size=source_directory_iterator.target_size,
                                                           color_mode = source_directory_iterator.color_mode, class_mode = source_directory_iterator.class_mode,
                                                           batch_size = source_directory_iterator.batch_size, shuffle = source_directory_iterator.shuffle,
                                                           data_format = source_directory_iterator.data_format, save_to_dir = source_directory_iterator.save_to_dir,
                                                           save_prefix = source_directory_iterator.save_prefix, save_format = source_directory_iterator.save_format,
                                                           seed=None, image_data_generator=source_directory_iterator.image_data_generator)


    def mark_last_batch_skipped(self):
        self.REUSE_PREVIOUS_BATCH = True

    def next(self):
        if self.REUSE_PREVIOUS_BATCH:
            self.REUSE_PREVIOUS_BATCH = False
            batch_x, batch_y =  self.PREVIOUS_BATCH
            self.__run_next_batch_callbacks(batch_x, batch_y)
            return batch_x, batch_y
        else:
            batch_x, batch_y = super(TransparentDirectoryIterator, self).next()
            self.__run_next_batch_callbacks(batch_x, batch_y)
            self.PREVIOUS_BATCH = (batch_x, batch_y)
            return batch_x, batch_y

    def __run_next_batch_callbacks(self, batch_x, batch_y):
        for callback in self.NEXT_CALL_BACKS:
            callback(batch_x, batch_y)



