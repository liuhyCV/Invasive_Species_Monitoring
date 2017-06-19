import tensorflow as tf

import numpy as np
from functools import reduce
import os
from tensorflow.python.platform import gfile

VGG_MEAN = [103.939, 116.779, 123.68]

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

class Train_Flags():
    def __init__(self):

        self.current_file_path = '/home/linze/liuhy/liuhy_github/Invasive_Species_Monitoring/model'

        self.max_step = 100000
        self.num_per_epoch = 10000
        self.num_epochs_per_decay = 30

        self.batch_size = 10
        # test_num should be divided by batch_size
        self.test_num = 1540

        self.initial_learning_rate = 0.01
        self.learning_rate_decay_factor = 0.9
        self.moving_average_decay = 0.999999

        self.output_summary_path = os.path.join(self.current_file_path, 'result','summary')
        self.output_check_point_path = os.path.join(self.current_file_path, 'result','check_point')
        self.output_test_result_path = os.path.join(self.current_file_path, 'result','test_result')

        self.dataset_path = '/home/linze/liuhy/liuhy_github/Invasive_Species_Monitoring/data'
        self.dataset_train_csv_file_path = '/home/linze/liuhy/liuhy_github/Invasive_Species_Monitoring/train.csv'
        self.dataset_test_csv_file_path = '/home/linze/liuhy/liuhy_github/Invasive_Species_Monitoring/test.csv'

        self.check_path_exist()


    def check_path_exist(self):
        if not gfile.Exists(self.output_summary_path):
            gfile.MakeDirs(self.output_summary_path)
        if not gfile.Exists(self.output_check_point_path):
            gfile.MakeDirs(self.output_check_point_path)
        if not gfile.Exists(self.output_test_result_path):
            gfile.MakeDirs(self.output_test_result_path)


class Vgg19:
    """
    A trainable version VGG19.
    """

    def __init__(self, vgg19_npy_path=None, dropout=0.5, train_test_mode=True):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.dropout = dropout
        self.train_test_mode = train_test_mode

    def build(self, rgb_scaled, test_rgb_scaled, train_test_mode):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param train_test_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        test_red, test_green, test_blue = tf.split(axis=3, num_or_size_splits=3, value=test_rgb_scaled)
        assert test_red.get_shape().as_list()[1:] == [224, 224, 1]
        assert test_green.get_shape().as_list()[1:] == [224, 224, 1]
        assert test_blue.get_shape().as_list()[1:] == [224, 224, 1]
        test_bgr = tf.concat(axis=3, values=[
            test_blue - VGG_MEAN[0],
            test_green - VGG_MEAN[1],
            test_red - VGG_MEAN[2],
        ])
        assert test_bgr.get_shape().as_list()[1:] == [224, 224, 3]


        data = tf.cond(train_test_mode, lambda: bgr, lambda: test_bgr)

        self.conv1_1 = self.conv_layer(data, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')


        '''
        
        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer_fine_tune(self.pool4, 25088, 2048, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)
        self.relu6 = tf.nn.dropout(self.relu6, self.dropout)



        self.fc7 = self.fc_layer_fine_tune(self.relu6, 2048, 1024, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        '''

        self.fc6 = self.fc_layer_fine_tune(self.pool4, 100352, 100, "fc6")
        self.relu6 = tf.nn.relu(self.fc6)
        self.relu6 = tf.cond(train_test_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)



        self.fc7 = self.fc_layer_fine_tune(self.relu6, 100, 2, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        #self.relu7 = tf.nn.dropout(self.relu7, self.dropout)




        '''

        self.fc8 = self.fc_layer_fine_tune(self.relu7, 4096, 1000, "fc8")


        '''

        self.data_dict = None



    # calculate function: the triplet batch output loss
    def calc_loss(self, logits, trains_labels):

        #cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=trains_labels))
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=trains_labels))

        tf.add_to_collection('losses', cost)


    def train_batch_inputs(self, dataset_csv_file_path, batch_size):

        with tf.name_scope('batch_processing'):
            if (os.path.isfile(dataset_csv_file_path) != True):
                raise ValueError('No data files found for this dataset')

            filename_queue = tf.train.string_input_producer([dataset_csv_file_path], shuffle=True)
            reader = tf.TextLineReader()
            _, serialized_example = reader.read(filename_queue)
            image_path, image_label = tf.decode_csv(
                serialized_example, [["train_image"], ["label"]])

            # input
            image = tf.read_file(image_path)
            img = tf.image.decode_jpeg(image, channels=3)
            img= tf.cast(img, dtype=tf.int16)

            resized_img = tf.image.resize_images(img, (IMAGE_HEIGHT, IMAGE_WIDTH))

            # generate batch
            imgs, images_label = tf.train.batch(
                [resized_img, image_label],
                batch_size=batch_size,
                num_threads=4,
                capacity=50 + 3 * batch_size
            )
            return imgs, images_label

    def test_batch_inputs(self, dataset_test_csv_file_path, batch_size):

        with tf.name_scope('test_batch_processing'):
            if (os.path.isfile(dataset_test_csv_file_path) != True):
                raise ValueError('No data files found for this test dataset')

            filename_queue = tf.train.string_input_producer([dataset_test_csv_file_path], shuffle=False)
            reader = tf.TextLineReader()
            _, serialized_example = reader.read(filename_queue)
            test_image_path, _ = tf.decode_csv(serialized_example, [["test_image"], ["none"]])

            # input
            test_image = tf.read_file(test_image_path)
            test = tf.image.decode_jpeg(test_image, channels=3)
            test = tf.cast(test, dtype=tf.int16)

            resized_test = tf.image.resize_images(test, (IMAGE_HEIGHT, IMAGE_WIDTH))

            # generate batch
            tests = tf.train.batch(
                [resized_test],
                batch_size=batch_size,
                num_threads=4,
                capacity=50 + 3 * batch_size
            )
            return tests

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


    def fc_layer_fine_tune(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var_fine_tune(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        filters = self.get_var(name, 0, name + "_filters")
        biases = self.get_var(name, 1, name + "_biases")

        return filters, biases

    def conv_layer_fine_tune(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var_fine_tune(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def get_conv_var_fine_tune(self, filter_size, in_channels, out_channels, name):

        filters = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        biases = tf.truncated_normal([out_channels], .0, .001)

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        weights = self.get_var(name, 0, name + "_weights")
        biases = self.get_var(name, 1, name + "_biases")

        return weights, biases

    def get_fc_var_fine_tune(self, in_size, out_size, name):
        weights = self._variable_with_weight_decay('weights', shape=[in_size, out_size],
            stddev=0.001, wd=0.0, trainable=True)
        biases = self._variable_on_gpu('biases', [out_size], tf.constant_initializer(0.001))

        return weights, biases

    def _variable_on_gpu(self, name, shape, initializer):
        var = tf.get_variable(name, shape, initializer=initializer)
        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd, trainable=True):
        var = self._variable_on_gpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
        return var

    def get_var(self, name, idx, var_name):

        var = tf.Variable(self.data_dict[name][idx], name=var_name)

        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
