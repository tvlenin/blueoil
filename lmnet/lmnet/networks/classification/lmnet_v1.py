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


class LmnetV1(Base):
    """Lmnet v1 for classification.
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

        #output = tf.space_to_depth(inputs, block_size=block_size, name=name)
        output = tf.layers.max_pooling2d(inputs,  2, 2,name=name)

        if self.data_format != 'NHWC':
            output = tf.transpose(output, perm=['NHWC'.find(d) for d in self.data_format])
        return output
    def RNN(self,is_training,inputs,depth,filters,reuse=tf.AUTO_REUSE,scope=None):
        channels_data_format = 'channels_last' if self.data_format == 'NHWC' else 'channels_first'
        _lmnet_block = self._get_lmnet_block(is_training, channels_data_format)
        with tf.variable_scope(scope, 'rnn'):
            x = inputs
            for i in range(0, depth):
                with tf.variable_scope('block{}'.format(i)):
                    x = _lmnet_block('conv1R', x, filters, 3)
                    x = tf.add(inputs, x,name='add1')
        return x

    def base(self, images, is_training, *args, **kwargs):
        """Base network.

        Args:
            images: Input images.
            is_training: A flag for if is training.
        Returns:
            tf.Tensor: Inference result.
        """

        channels_data_format = 'channels_last' if self.data_format == 'NHWC' else 'channels_first'
        _lmnet_block = self._get_lmnet_block(is_training, channels_data_format)

        self.images = images
        

        #t=0
        x1f = _lmnet_block('conv1', images, 32, 3)
        x2f = _lmnet_block('conv2', x1f, 64, 3)
        p2f = self._space_to_depth(name='pool2', inputs=x2f)
        #x3f = _lmnet_block('conv3', p2f, 128, 3)
        #x4f = _lmnet_block('conv4', x3f, 64, 3)
        p4f = self._space_to_depth(name='pool4', inputs=p2f)
        #x5f = _lmnet_block('conv5', p4f, 128, 3)
        p5f = self._space_to_depth(name='pool5', inputs=p4f)
        #x6f = _lmnet_block('conv6', p5f, 64, 3)
        #x6f = tf.layers.dropout(x6f, training=is_training)
        #x7f = _lmnet_block('conv7', p5f, 10, 3)

       
        #t=1
        x1r = _lmnet_block('conv1', images, 32, 3) + _lmnet_block('conv1r', x1f, 32, 3)
        x2r = _lmnet_block('conv2r', x2f, 64, 3)
        p2r = self._space_to_depth(name='pool2', inputs=x2r)
        #x3r = _lmnet_block('conv3', p2r, 128, 3) + _lmnet_block('conv3r', x3f, 128, 3)
        #x4r = _lmnet_block('conv4', x3r, 64, 3) + _lmnet_block('conv4r', x4f, 64, 3)
        p4r = self._space_to_depth(name='pool4', inputs=p2r)
        #x5r = _lmnet_block('conv5', p4r, 128, 3) + _lmnet_block('conv5r', x5f, 128, 3)
        p5r = self._space_to_depth(name='pool5', inputs=p4r)
        #x6r = _lmnet_block('conv6', p5r, 64, 3) + _lmnet_block('conv6r', x6f, 64, 3)
        x7r = _lmnet_block('conv7', p5r, 10, 3) + _lmnet_block('conv7r', p5f, 10, 3)
        '''
        #t=2
        x1r = _lmnet_block('conv11', images, 32, 3) + _lmnet_block('conv1r', x1r, 32, 3)
        x2r = _lmnet_block('conv22', x1r, 64, 3) + _lmnet_block('conv2r', x2r, 64, 3)
        p2r = self._space_to_depth(name='pool2', inputs=x2r)
        x3r = _lmnet_block('conv22', p2r, 128, 3) + _lmnet_block('conv3r', x3r, 128, 3)
        x4r = _lmnet_block('conv4', x3r, 64, 3) + _lmnet_block('conv4r', x4r, 64, 3)
        p4r = self._space_to_depth(name='pool4', inputs=x4r)
        x5r = _lmnet_block('conv5', p4r, 128, 3) + _lmnet_block('conv5r', x5r, 128, 3)
        p5r = self._space_to_depth(name='pool5', inputs=x5r)
        x6r = _lmnet_block('conv6', p5r, 64, 3) + _lmnet_block('conv6r', x6r, 64, 3)
        x7r = _lmnet_block('conv7', x6r, 10, 3) + _lmnet_block('conv7r', x7r, 10, 3)

        #t=3
        x1r = _lmnet_block('conv11', images, 32, 3) + _lmnet_block('conv1r', x1r, 32, 3)
        x2r = _lmnet_block('conv22', x1r, 64, 3) + _lmnet_block('conv2r', x2r, 64, 3)
        p2r = self._space_to_depth(name='pool2', inputs=x2r)
        x3r = _lmnet_block('conv22', p2r, 128, 3) + _lmnet_block('conv3r', x3r, 128, 3)
        x4r = _lmnet_block('conv4', x3r, 64, 3) + _lmnet_block('conv4r', x4r, 64, 3)
        p4r = self._space_to_depth(name='pool4', inputs=x4r)
        x5r = _lmnet_block('conv5', p4r, 128, 3) + _lmnet_block('conv5r', x5r, 128, 3)
        p5r = self._space_to_depth(name='pool5', inputs=x5r)
        x6r = _lmnet_block('conv6', p5r, 64, 3) + _lmnet_block('conv6r', x6r, 64, 3)
        x7r = _lmnet_block('conv7', x6r, 10, 3) + _lmnet_block('conv7r', x7r, 10, 3)
        '''
        h = x7r.get_shape()[1].value if self.data_format == 'NHWC' else x7r.get_shape()[2].value
        w = x7r.get_shape()[2].value if self.data_format == 'NHWC' else x7r.get_shape()[3].value
        x = tf.layers.average_pooling2d(name='pool7',
                                        inputs=x7r,
                                        pool_size=[h, w],
                                        padding='VALID',
                                        strides=1,
                                        data_format=channels_data_format)

        self.base_output = tf.reshape(x, [-1, self.num_classes], name='pool7_reshape')


        return self.base_output


class LmnetV1Quantize(LmnetV1):
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
