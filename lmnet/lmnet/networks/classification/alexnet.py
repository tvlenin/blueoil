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

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

#change class name
class Alexnet(Base):
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


    def base(self,images,is_training, dropout_keep_prob = 0.5,
                        spatial_squeeze=True,
                        scope='alexnet_v2',
                        global_pool=False,*args, **kwargs,):
        self.images = images
        
        #with tf.variable_scope(scope, 'alexnet_v2', [self.images]) as sc:
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        #with tf.variable_scope('base',[tf.layers.conv2d,tf.layers.average_pooling2d]):
        net = tf.layers.conv2d(self.images, 64, [3, 3], strides=[1,1], padding='SAME',name='conv1')
        net = self._space_to_depth(net,4,'pool1')
        print(net.shape)
        net = self._space_to_depth(net,2,'pool11')
        #net = tf.layers.average_pooling2d(net, [3, 3], strides=[2,2],padding='SAME', name='pool1')
        net = tf.layers.conv2d(net, 192, [3, 3], name='conv2',padding='SAME')
        #net = tf.layers.average_pooling2d(net, [3, 3], strides=[2,2],padding='SAME', name='pool2')
        net = self._space_to_depth(net,2,'pool2')
        print(net.shape)
        net = tf.layers.conv2d(net, 384, [3, 3], name='conv3')
        net = tf.layers.conv2d(net, 384, [3, 3], name='conv4')
        net = tf.layers.conv2d(net, 256, [3, 3], name='conv5')
        #net = tf.layers.average_pooling2d(net,  [3, 3], strides=[2,2],padding='SAME', name='pool5')
        net = self._space_to_depth(net,2,'pool5')
        
        print(net.shape)
        net = tf.layers.conv2d(net, 384, [3, 3], padding='Valid',name='fc6')
        net = tf.layers.dropout(net, dropout_keep_prob,name='dropout6')
        net = tf.layers.conv2d(net, 128, [2, 2], name='fc7')
        net = tf.layers.dropout(net, dropout_keep_prob, name='dropout7')
        net = tf.layers.conv2d(net, self.num_classes, [1, 1],name='fc8')
        net = tf.reshape(net, [-1, self.num_classes], name='pool7_reshape')
        return net

#Only change the LmnetV1 and the class name
class AlexnetQuantize(Alexnet):
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
