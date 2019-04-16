# -*- coding: utf-8 -*-
# Copyright 2018 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import functools

import tensorflow as tf

from lmnet.blocks import lmnet_block
from lmnet.networks.classification.base import Base
VGG_MEAN = [103.939, 116.779, 123.68]
#change class name
class Vgg16Network(Base):
    """template for classification.
    """
    version = 1.0

    def __init__(
            self,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        self.activation = tf.nn.relu
        self.custom_getter = None

    def _get_lmnet_block(self, is_training, channels_data_format):
        return functools.partial(lmnet_block,
                                 activation=self.activation,
                                 custom_getter=self.custom_getter,
                                 is_training=is_training,
                                 is_debug=self.is_debug,
                                 use_bias=False,
                                 data_format=channels_data_format)

    def _space_to_depth(self, inputs=None, block_size=2, name=''):
        if self.data_format != 'NHWC':
            inputs = tf.transpose(inputs, perm=[self.data_format.find(d) for d in 'NHWC'])

        output = tf.space_to_depth(inputs, block_size=block_size, name=name)

        if self.data_format != 'NHWC':
            output = tf.transpose(output, perm=['NHWC'.find(d) for d in self.data_format])
        return output

    def base(self, images, is_training, *args, **kwargs):
        self.images = images
        print(images.shape)
        keep_prob = tf.cond(is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.0))

        self.images = images
        

        net = tf.layers.conv2d(self.images, 64, [3, 3], strides=[1,1], padding='SAME',name='conv1')
        net = tf.layers.conv2d(self.images, 64, [3, 3], strides=[1,1], padding='SAME',name='conv1_1')
        
        net = tf.layers.max_pooling2d(net,  2, 2,name="pool1")

        net = tf.layers.conv2d(net, 128, [3, 3], name='conv2',padding='SAME')
        net = tf.layers.conv2d(net, 128, [3, 3], name='conv21',padding='SAME')

        net = tf.layers.max_pooling2d(net,  2, 2,name="pool2")
    
        net = tf.layers.conv2d(net, 256, [3, 3], name='conv3',padding='SAME')
        net = tf.layers.conv2d(net, 256, [3, 3], name='conv31',padding='SAME')
        net = tf.layers.conv2d(net, 256, [3, 3], name='conv32',padding='SAME')

        net = tf.layers.max_pooling2d(net,  2, 2,name="pool3",padding='SAME')

        net = tf.layers.conv2d(net, 512, [3, 3], name='conv4',padding='SAME')
        net = tf.layers.conv2d(net, 512, [3, 3], name='conv41',padding='SAME')
        net = tf.layers.conv2d(net, 512, [3, 3], name='conv42',padding='SAME')

        net = tf.layers.max_pooling2d(net,  2, 2,name="pool4")

        net = tf.layers.conv2d(net, 512, [3, 3], name='conv5',padding='SAME')
        net = tf.layers.conv2d(net, 512, [3, 3], name='conv51',padding='SAME')
        net = tf.layers.conv2d(net, 512, [3, 3], name='conv52',padding='SAME')
        
        net = tf.layers.max_pooling2d(net,  2, 2,name="pool5",padding='SAME')
        
        print(net.shape)
        net = tf.layers.conv2d(net, 4092, [3, 3], padding='Valid',name='fc6')
        net = tf.layers.max_pooling2d(net,  2, 2,name="pool6")
        net = tf.layers.dropout(net, keep_prob,name='dropout6')
        net = tf.layers.conv2d(net, 4092, [2, 2], name='fc7')
        net = tf.layers.dropout(net, keep_prob, name='dropout7')
        net = tf.layers.conv2d(net, self.num_classes, [1, 1],name='fc8')
        net = tf.reshape(net, [-1, self.num_classes], name='pool7_reshape')
        return net

        '''net = tf.layers.conv2d(self.images,64,[3,3],name="conv1",padding="SAME",activation=tf.nn.relu)
        net = tf.layers.conv2d(net,64,[3,3],name="conv2",padding="SAME",activation=tf.nn.relu)
        print(net.shape)
        net = tf.layers.max_pooling2d(net,  2, 2,name="pool1")
        print(net.shape)
        net = tf.layers.conv2d(net, 128, [3,3],name="conv3",padding="SAME",activation=tf.nn.relu)
        net = tf.layers.conv2d( net,128, [3,3], name="conv4",padding="SAME" ,activation=tf.nn.relu)
        print(net.shape)
        net = tf.layers.max_pooling2d(net,2,2,name="pool2")
        print(net.shape)
        net = tf.layers.conv2d(net,256,  [3,3],name="conv5",padding="SAME",activation=tf.nn.relu)
        net = tf.layers.conv2d(net,256,  [3,3],name="conv6",padding="SAME",activation=tf.nn.relu)
        net = tf.layers.conv2d(net,256,  [3,3],name="conv7",padding="SAME",activation=tf.nn.relu)
        print(net.shape)
        net = tf.layers.max_pooling2d(net,2, 2,name="pool3")
        print(net.shape)
        net = tf.layers.conv2d(net,  512,  [3,3] ,name="conv8",padding="SAME",activation=tf.nn.relu)
        net = tf.layers.conv2d(net,  512,  [3,3], name="conv9",padding="SAME",activation=tf.nn.relu)
        net = tf.layers.conv2d(net,  512,  [3,3],name="conv10",padding="SAME",activation=tf.nn.relu)
        print(net.shape)
        net = tf.layers.max_pooling2d(net,2,2,name="pool4")
        print(net.shape)
        net = tf.layers.conv2d(net,   512,  [3,3],name="conv11",padding="SAME",activation=tf.nn.relu)
        net = tf.layers.conv2d(net,   512,  [3,3],name="conv12",padding="SAME",activation=tf.nn.relu)
        net = tf.layers.conv2d( net,  512,  [3,3],name="conv13",padding="SAME",activation=tf.nn.relu)

        net = tf.layers.max_pooling2d(net, 2,2,name="pool5")

        net = tf.layers.conv2d(net, 128, [7, 7], name='fc6',activation=tf.nn.relu)
        #net = self.fc_layer("fc14", net, filters=4096, activation=tf.nn.relu)
        net = tf.nn.dropout(net, keep_prob)
        net = tf.layers.conv2d(net, 128, [1, 1], name='fc7',activation=tf.nn.relu)
        #net = self.fc_layer("fc15", net, filters=4096, activation=tf.nn.relu)
        net = tf.nn.dropout(net, keep_prob)
        self._heatmap_layer = net
        net = tf.layers.conv2d(net, self.num_classes, [1, 1],name='fc8',activation=tf.nn.relu)
        print(net.shape)

        net = tf.reshape(net, [-1, self.num_classes], name='pool7_reshape')
        #self.fc16 = self.fc_layer("fc16", net, filters=self.num_classes, activation=None)
        #print(self.fc16.shape)

        return net'''

    def conv_layer(
        self,
        name,
        inputs,
        filters,
        kernel_size,
        strides=1,
        padding="SAME",
        activation=tf.nn.sigmoid,
        *args,
        **kwargs
    ):
        kernel_initializer = tf.contrib.layers.xavier_initializer()
        biases_initializer = tf.zeros_initializer()

        output = super(Vgg16Network, self).conv_layer(
            name=name,
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
            kernel_initializer=kernel_initializer,
            biases_initializer=biases_initializer,
            *args,
            **kwargs
        )

        return output
    def fc_layer(
            self,
            name,
            inputs,
            filters,
            *args,
            **kwargs
    ):
        kernel_initializer = tf.contrib.layers.xavier_initializer()
        biases_initializer = tf.zeros_initializer()

        output = super(Vgg16Network, self).fc_layer(
            name=name,
            inputs=inputs,
            filters=filters,
            kernel_initializer=kernel_initializer,
            biases_initializer=biases_initializer,
            *args,
            **kwargs
        )

        return output
    def convert_rbg_to_bgr(self, rgb_images):
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_images)

        bgr_images = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        return bgr_images
#Only change the LmnetV1 and the class name
class Vgg16NetworkQuantize(Vgg16Network):
    """Lmnet quantize network for classification, version 1.0

    Following `args` are used for inference: ``activation_quantizer``, ``activation_quantizer_kwargs``,
    ``weight_quantizer``, ``weight_quantizer_kwargs``.

    Args:
        activation_quantizer (callable): Weight quantizater. See more at `lmnet.quantizations`.
        activation_quantizer_kwargs (dict): Kwargs for `activation_quantizer`.
        weight_quantizer (callable): Activation quantizater. See more at `lmnet.quantizations`.
        weight_quantizer_kwargs (dict): Kwargs for `weight_quantizer`.
    """
    version = 1.0

    def __init__(
            self,
            activation_quantizer=None,
            activation_quantizer_kwargs=None,
            weight_quantizer=None,
            weight_quantizer_kwargs=None,
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )

        assert weight_quantizer
        assert activation_quantizer

        activation_quantizer_kwargs = activation_quantizer_kwargs if activation_quantizer_kwargs is not None else {}
        weight_quantizer_kwargs = weight_quantizer_kwargs if weight_quantizer_kwargs is not None else {}

        self.activation = activation_quantizer(**activation_quantizer_kwargs)
        weight_quantization = weight_quantizer(**weight_quantizer_kwargs)
        self.custom_getter = functools.partial(self._quantized_variable_getter,
                                               weight_quantization=weight_quantization)

    @staticmethod
    def _quantized_variable_getter(getter, name, weight_quantization=None, *args, **kwargs):
        """Get the quantized variables.

        Use if to choose or skip the target should be quantized.

        Args:
            getter: Default from tensorflow.
            name: Default from tensorflow.
            weight_quantization: Callable object which quantize variable.
            args: Args.
            kwargs: Kwargs.
        """
        assert callable(weight_quantization)
        var = getter(name, *args, **kwargs)
        with tf.variable_scope(name):
            # Apply weight quantize to variable whose last word of name is "kernel".
            if "kernel" == var.op.name.split("/")[-1]:
                return weight_quantization(var)
        return var
