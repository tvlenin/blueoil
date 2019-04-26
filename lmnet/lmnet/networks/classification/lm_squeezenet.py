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
class SqueezeNet(Base):
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
        channels_data_format = 'channels_last' if self.data_format == 'NHWC' else 'channels_first'
        _lmnet_block = self._get_lmnet_block(is_training, channels_data_format)

        self.images = images
        print(images.shape)
        keep_prob = tf.cond(is_training, lambda: tf.constant(0.5), lambda: tf.constant(1.0))

        self.images = images
        
        four_letter_data_format = 'NHWC'
        #
        net = _lmnet_block('conv1',self.images, 32, 3)
        #
        net = tf.layers.max_pooling2d(net,  2, 2,name="pool1")
        #
        net = _lmnet_block('conv2',net, 32, 3)
        #
        net = tf.layers.max_pooling2d(net,  2, 2,name="pool1_1")
        #
        print(net.shape)
        #squeeze layer
        squeeze1 = _lmnet_block('squeze1',net, 32, 1)
        #expand layer
        k1_relu = _lmnet_block('squeze11',squeeze1, 64, 1)
        k3_relu = _lmnet_block('squeze13',squeeze1, 64, 3)
        net = tf.concat([k1_relu,k3_relu],axis=3)
        #squeeze layer
        squeeze2 = _lmnet_block('squeze2',net, 32, 1)
        #expand layer
        k1_relu = _lmnet_block('squeze21',squeeze2, 64, 1)
        k3_relu = _lmnet_block('squeze23',squeeze2, 64, 3)
        net = tf.concat([k1_relu,k3_relu],axis=3)
        
        #
        net = tf.layers.max_pooling2d(net,  2, 2,name="pool2")
        print(net.shape)
        #squeeze layer
        #squeeze layer
        squeeze3 = _lmnet_block('squeze3',net, 32, 1)
        #expand layer
        k1_relu = _lmnet_block('squeze31',squeeze3, 128, 1)
        k3_relu = _lmnet_block('squeze33',squeeze3, 128, 3)
        net = tf.concat([k1_relu,k3_relu],axis=3)
        #
        #squeeze layer
        squeeze4 = _lmnet_block('squeze4',net, 32, 1)
        #expand layer
        k1_relu = _lmnet_block('squeze41',squeeze4, 128, 1)
        k3_relu = _lmnet_block('squeze43',squeeze4, 128, 3)
        net = tf.concat([k1_relu,k3_relu],axis=3)
        #
        net = tf.layers.max_pooling2d(net,  2, 2,name="pool3")   
        #
        print(net.shape)
        #squeeze layer
        squeeze5 = _lmnet_block('squeze5',net, 64, 1)
        #expand layer
        k1_relu = _lmnet_block('squeze51',squeeze5, 128, 1)
        k3_relu = _lmnet_block('squeze53',squeeze5, 128, 3)
        net = tf.concat([k1_relu,k3_relu],axis=3)
        #
        #squeeze layer
        squeeze6 = _lmnet_block('squeze6',net, 64, 1)
        #expand layer
        k1_relu = _lmnet_block('squeze61',squeeze6, 128, 1)
        k3_relu = _lmnet_block('squeze63',squeeze6, 128, 3)
        net = tf.concat([k1_relu,k3_relu],axis=3)
        #
        #squeeze layer
        squeeze7 = _lmnet_block('squeze7',net, 128, 1)
        #expand layer
        k1_relu = _lmnet_block('squeze71',squeeze7, 256, 1)
        k3_relu = _lmnet_block('squeze73',squeeze7, 256, 3)
        net = tf.concat([k1_relu,k3_relu],axis=3)
        #
        #squeeze layer
        squeeze8 = _lmnet_block('squeze8',net, 128, 1)
        #expand layer
        k1_relu = _lmnet_block('squeze81',squeeze8, 256, 1)
        k3_relu = _lmnet_block('squeze83',squeeze8, 256, 3)
        net = tf.concat([k1_relu,k3_relu],axis=3)
        #
        print(net.shape)
        net = tf.layers.dropout(net,rate=0.5,training=is_training)

        #w_init = tf.truncated_normal_initializer(mean=0.0,stddev=(1.0/int(net.shape[2])))
        #net = tf.layers.conv2d(net,10,[3,3],[1,1],padding='valid',kernel_initializer=w_init)
        net = _lmnet_block('full',net, 10, 3)
        #batch_norm = tf.contrib.layers.batch_norm(net,decay=0.99,scale=True,center=True,updates_collections=None,is_training=is_training,data_format=four_letter_data_format)
        #net =self.activation(batch_norm)
        print(net.shape)
        net = tf.layers.average_pooling2d(net,pool_size=14,strides=1,name="pool_end")
        #
        print(net.shape)
        pool_shape = tf.shape(net)
        net = tf.reshape(net,shape=(pool_shape[0],pool_shape[3])) 
        #tf.reshape(net, [-1, self.num_classes], name='pool7_reshape')
        return net


    
#Only change the LmnetV1 and the class name
class squeezenetV1Quantize(SqueezeNet):
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
