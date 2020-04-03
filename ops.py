import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
from utils import *

def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=tf.compat.v1.GraphKeys.UPDATE_OPS, epsilon=1e-5, scale=True, scope=name)

def batchnormSR(inputs, is_training=True):
    return slim.batch_norm(inputs, decay=0.9, epsilon=0.001, updates_collections=tf.compat.v1.GraphKeys.UPDATE_OPS,
                        scale=False, fused=True, is_training=is_training)

def denselayer(inputs, output_size, name="dense_layer"):
    with tf.compat.v1.variable_scope(name):
        return slim.fully_connected(inputs, output_size, activation_fn=None)
        
def instance_norm(input, name="instance_norm"):
    with tf.compat.v1.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset

def conv2d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv2d"):
    with tf.compat.v1.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding, activation_fn=None)

def deconv2d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv2d"):
    with tf.compat.v1.variable_scope(name):
        return slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None)
        
def conv3d(input_, output_dim, ks=4, s=2, stddev=0.02, padding='SAME', name="conv3d"):
    with tf.compat.v1.variable_scope(name):
        return slim.conv3d(input_, output_dim, ks, s, padding=padding, activation_fn=None)

def deconv3d(input_, output_dim, ks=4, s=2, stddev=0.02, name="deconv3d"):
    with tf.compat.v1.variable_scope(name):
        return slim.conv3d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None, biases_initializer=None)
'''
def conv3d(x, numFilters, kernelSize, stride, padding):
    x =  tf.keras.layers.Conv3D(filters=numFilters, kernel_size=kernelSize, strides=stride, padding=padding)(x)
    return x

def deconv3d(x, numFilters, ks=4, s=2, padding='SAME', name='deconv3d'):
    x =  tf.keras.layers.Conv3DTranspose(filters=numFilters, kernel_size=ks, strides=s, padding=padding)(x)
    return x
''' 

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)
    
def preluLayer(_x):
    _x=tf.keras.layers.PReLU(shared_axes=[1, 2])(_x)
    return _x
#def preluLayer(_x, name='alpha'):
#    """
#    Parametric ReLU
#    """
#    alphas = tf.get_variable(name, _x.get_shape()[-1],
#                       initializer=tf.constant_initializer(0.1),
#                        dtype=tf.float32, trainable=True)
#    pos = tf.nn.relu(_x)
#    neg = alphas * (_x - abs(_x)) * 0.5

#return pos + neg

def prelu(x):
    x=preluLayer(x)
    return x
    
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

    with tf.compat.v1.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
    
def bicubic_kernel(x, a=0): # use hermite resampling a=0 to avoid black white inversion
  """https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic"""
  if abs(x) <= 1:
    return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
  elif 1 < abs(x) and abs(x) < 2:
    return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a 
  else:
    return 0

def build_filter(factor):
  size = factor*4
  k = np.zeros((size))
  for i in range(size):
    x = (1/factor)*(i- np.floor(size/2) +0.5)
    k[i] = bicubic_kernel(x)
  k = k / np.sum(k)
  # make 2d
  k = np.outer(k, k.T)
  k = tf.constant(k, dtype=tf.float32, shape=(size, size, 1, 1))
  return tf.concat([k, k, k], axis=2)

def apply_bicubic_downsample(x, filter, factor):
  """Downsample x by a factor of factor, using the filter built by build_filter()
  x: a rank 4 tensor with format NHWC
  filter: from build_filter(factor)
  factor: downsampling factor (ex: factor=2 means the output size is (h/2, w/2))
  """
  # using padding calculations from https://www.tensorflow.org/api_guides/python/nn#Convolution
  #x = (x+1)*127.5
  filter_height = factor*4
  filter_width = factor*4
  strides = factor
  pad_along_height = max(filter_height - strides, 0)
  pad_along_width = max(filter_width - strides, 0)
  # compute actual padding values for each side
  pad_top = pad_along_height // 2
  pad_bottom = pad_along_height - pad_top
  pad_left = pad_along_width // 2
  pad_right = pad_along_width - pad_left
  # apply mirror padding
  x = tf.pad(x, [[0,0], [pad_top,pad_bottom], [pad_left,pad_right], [0,0]], mode='REFLECT')
  # downsampling performed by strided conv
  x = tf.nn.depthwise_conv2d(x, filter=filter, strides=[1,strides,strides,1], padding='VALID')
  #x = x/127.5 - 1
  return x
