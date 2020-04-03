from __future__ import division
import tensorflow as tf
from ops import *
from utils import *
from tensorlayer.layers import * # use TL for this implementation, and keras for the tf2 implementation
def generator_resnetYDW(image, options, reuse=False, name="generator", g3Flag=False): # this has been modified to suit c2gan and acgan for poisson noise removal and downsampling
#TODO: checkerboarding is quite extreme, switched to subpixel conv
    factor = 2
    with tf.compat.v1.variable_scope(name):
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        #else:
            #assert tf.compat.v1.get_variable_scope().reuse is False

        def residualBlock(x, dim, ks=3, s=1, name='res'): # the srresnet block
            y = x
            y = conv2d(y, dim, ks, s, padding='SAME', name=name+'_c1')
            y = tf.nn.relu(y)
            y = conv2d(y, dim, ks, s, padding='SAME', name=name+'_c2')
            return y + x
        # dont change the image size, leave it alone
        c0 = image
        c0 = tf.nn.relu(conv2d(c0, options.gf_dim, 9, 1, name='g_e1_c'))
        c1 = c0#tf.nn.space_to_depth(c0, 2, name='g_e1_subpix')
        c0 = tf.nn.relu(conv2d(c0, options.gf_dim, 7, 1, name='g_e2_c'))
        c0 = c0#tf.nn.space_to_depth(c0, 2, name='g_e2_subpix')
        c0 = tf.nn.relu(conv2d(c0, options.gf_dim, 5, 1, name='g_e3_c'))
        # convert the image by resnets
        # define G network with 9 resnet blocks
        if not g3Flag: # if this is symetric, use the full network
            r = residualBlock(c0, options.gf_dim, name='g_r1')
            r = residualBlock(r, options.gf_dim, name='g_r2')
            r = residualBlock(r, options.gf_dim, name='g_r3')
            r = residualBlock(r, options.gf_dim, name='g_r4')
            r = residualBlock(r, options.gf_dim, name='g_r5')
            r = residualBlock(r, options.gf_dim, name='g_r6')
            r = residualBlock(r, options.gf_dim, name='g_r7')
            r = residualBlock(r, options.gf_dim, name='g_r8')
            r = residualBlock(r, options.gf_dim, name='g_r9')
            r = r + c0
            d = tf.nn.relu(conv2d(r, options.gf_dim, 3, 1, name='g_d1_c'))
            #d = tf.nn.relu(conv2d(d, options.gf_dim, 3, 1, name='g_d2_c'))
        elif g3Flag: # if asymmetric, skip the resblocks, do a skip to reduce artefacts, and downsample
            d = tf.nn.relu(tf.nn.space_to_depth(c0+c1, factor, name='g_d1_subpix')) 
            d = conv2d(d, options.gf_dim, 3, 1, name='g_d1_c')
            #add nins here or befpre
            d = tf.nn.relu(tf.nn.space_to_depth(d, factor, name='g_d2_subpix'))
            d = tf.nn.relu(conv2d(d, options.gf_dim, 3, 1, name='g_d2_c'))
        pred = tf.nn.tanh(conv2d(d, options.output_c_dim, 3, 1, padding='SAME', name='g_pred_c'))
        return pred

def generator_unetYDW(image, options, reuse=False, name="generator"):
    #TODO: the phase shifted unet is much nore information dense. should optimise the filternumbers per layer
    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.compat.v1.variable_scope(name):
        # image is 256 x 256 x c
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        #else:
            #assert tf.compat.v1.get_variable_scope().reuse is False

        # image is (256 x 256 x c) 
        # encode the image
        e1 = instance_norm(conv2d(image, options.gf_dim, 4, 1, name='g_e1_conv'))
        # space to depth
        e1 = tf.nn.space_to_depth(e1, 2, name='g_e1_subpix')
        # e1 is (128 x 128 x 4nf)
        e2 = instance_norm(conv2d(lrelu(e1), options.gf_dim*2, 4, 1, name='g_e2_conv'), 'g_bn_e2')
        e2 = tf.nn.space_to_depth(e2, 2, name='g_e2_subpix')
        # e2 is (64 x 64 x 8nf)
        e3 = instance_norm(conv2d(lrelu(e2), options.gf_dim*4, 4, 1, name='g_e3_conv'), 'g_bn_e3')
        e3 = tf.nn.space_to_depth(e3, 2, name='g_e3_subpix')
        # e3 is (32 x 32 x 16nf)
        e4 = instance_norm(conv2d(lrelu(e3), options.gf_dim*8, 4, 1, name='g_e4_conv'), 'g_bn_e4')
        e4 = tf.nn.space_to_depth(e4, 2, name='g_e4_subpix')
        # e4 is (16 x 16 x 32nf)
        e5 = instance_norm(conv2d(lrelu(e4), options.gf_dim*8, 4, 1, name='g_e5_conv'), 'g_bn_e5')
        e5 = tf.nn.space_to_depth(e5, 2, name='g_e5_subpix')
        # e5 is (8 x 8 x 32nf)
        e6 = instance_norm(conv2d(lrelu(e5), options.gf_dim*8, 4, 1, name='g_e6_conv'), 'g_bn_e6')
        e6 = tf.nn.space_to_depth(e6, 2, name='g_e6_subpix')
        # e6 is (4 x 4 x 32nf)
        e7 = instance_norm(conv2d(lrelu(e6), options.gf_dim*8, 4, 1, name='g_e7_conv'), 'g_bn_e7')
        e7 = tf.nn.space_to_depth(e7, 2, name='g_e7_subpix')
        # e7 is (2 x 2 x 32nf)
        e8 = instance_norm(conv2d(lrelu(e7), options.gf_dim*8, 4, 1, name='g_e8_conv'), 'g_bn_e8')
        e8 = tf.nn.space_to_depth(e8, 2, name='g_e8_subpix')
        # e8 is (1 x 1 x 32nf)
        e8 = instance_norm(conv2d(lrelu(e8), options.gf_dim*8, 4, 1, name='g_bottom_conv'), 'g_bn_bottom')
        #go back up
        e8 = tf.nn.depth_to_space(e8, 2, name='g_bottom_subpix')
        d1 = conv2d(tf.nn.relu(e8), options.gf_dim*8, 4, 1, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e7], 3)
        # d1 is (2 x 2 x 32nf*2)
        d1 = tf.nn.depth_to_space(d1, 2, name='g_d1_subpix')
        d2 = conv2d(tf.nn.relu(d1), options.gf_dim*8, 4, 1, name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e6], 3)
        # d2 is (4 x 4 x 32nf*2)
        d2 = tf.nn.depth_to_space(d2, 2, name='g_d2_subpix')
        d3 = conv2d(tf.nn.relu(d2), options.gf_dim*8, 4, 1, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e5], 3)
        # d3 is (8 x 8 x 32nf*2)
        d3 = tf.nn.depth_to_space(d3, 2, name='g_d3_subpix')
        d4 = conv2d(tf.nn.relu(d3), options.gf_dim*8, 4, 1, name='g_d4')
        d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)
        # d4 is (16 x 16 x 32nf*2)
        d4 = tf.nn.depth_to_space(d4, 2, name='g_d4_subpix')
        d5 = conv2d(tf.nn.relu(d4), options.gf_dim*4, 4, 1, name='g_d5')
        d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)
        # d5 is (32 x 32 x 16nf*2)
        d5 = tf.nn.depth_to_space(d5, 2, name='g_d5_subpix')
        d6 = conv2d(tf.nn.relu(d5), options.gf_dim*2, 4, 1, name='g_d6')
        d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)
        # d6 is (64 x 64 x 8nf*2)
        d6 = tf.nn.depth_to_space(d6, 2, name='g_d6_subpix')
        d7 = conv2d(tf.nn.relu(d6), options.gf_dim, 4, 1, name='g_d7')
        d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3)
        # d7 is (128 x 128 x 4nf*2)
        d7 = tf.nn.depth_to_space(d7, 2, name='g_d7_subpix')
        d8 = conv2d(tf.nn.relu(d7), options.output_c_dim, 4, 1, name='g_d8')
        # d8 is (256 x 256 x c)

        return tf.nn.tanh(d8)

def edsrYDW(image, options, reuse=False, name='EDSRYDW', numResBlocks=16):
    with tf.compat.v1.variable_scope(name):
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        #else:
           #assert tf.compat.v1.get_variable_scope().reuse is False

        def residualBlock(x, dim, ks=3, s=1, name='resEDSR'): # the edsr block
            y = x
            y = conv2d(y, dim, ks, s, padding='SAME', name=name+'_c1')
            y = tf.nn.relu(y)
            y = conv2d(y, dim, ks, s, padding='SAME', name=name+'_c2')
            return y + x
        # encode the image with an initial conv layer
        c0 = conv2d(image, options.srf_dim, 3, 1, name='g_e_shallow_c')
        shallow=c0
        # pass through the residual blocks
        for i in range(1, numResBlocks+1):
            c0 = residualBlock(c0, options.srf_dim, name='g_residual_%d'%(i))
        # output conv
        deep = conv2d(c0, options.srf_dim, 3, 1, name='g_e_deep_c')
        # skip connection edsr
        c0 = deep + shallow
        # super resolve the image
        c0 = conv2d(c0, options.srf_dim*4, 3, 1, name='g_presubconv_1')
        c0 = tf.nn.depth_to_space(c0, 2, name='g_d1_subpix')
        c0 = tf.nn.relu(c0)#where should the activation go?
        
        c0 = conv2d(c0, options.srf_dim*4, 3, 1, name='g_presubconv_2')
        c0 = tf.nn.depth_to_space(c0, 2, name='g_d2_subpix')
        c0 = tf.nn.relu(c0)#where should the activation go?
        # do the wdsr skip
        wideOut=tf.nn.relu(tf.nn.depth_to_space(conv2d(shallow, options.output_c_dim*16, 5, 1, name='wideSkip'), 4, name='wideSubPix'))
        c0 = tf.nn.tanh(conv2d(c0, options.output_c_dim, 3, 1, padding='SAME', name='g_pred_c')+wideOut)
        return c0
        

def discriminator(image, options, reuse=False, name="discriminator", d2Flag=False): # patchGAN doesnt work well with minibatches
    # pools the LR input by a factor of 8, and the SR by a factor of 32
    # instanceNorm works very poorly with SR. it was designed for style transfer, so doesnt preserve accuracy as well
    with tf.compat.v1.variable_scope(name):
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        #else:
            #assert tf.compat.v1.get_variable_scope().reuse is False
        s=2
        if d2Flag:
            s=4 # reduce the 4x images appropriately
        h = lrelu(conv2d(image, options.df_dim, ks=4, s=s, name='d_h0_conv')) # 2 or 4
        h = lrelu(instance_norm(conv2d(h, options.df_dim*2, ks=4, s=s, name='d_h1_conv'), 'd_bn1')) # 2 or 4
        h = lrelu(instance_norm(conv2d(h, options.df_dim*4, ks=4, s=2, name='d_h2_conv'), 'd_bn2')) # 2
        h = lrelu(instance_norm(conv2d(h, options.df_dim*8, ks=4, s=1, name='d_h3_conv'), 'd_bn3')) # 1
        h = conv2d(h, 1, ks=4, s=1, name='d_h3_pred')
        return h

def discriminatorSR(image, options, reuse=False, name="discriminator", d2Flag=False): # the SRGAN discriminator modified to fit acgan
    with tf.compat.v1.variable_scope(name):
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        #else:
            #assert tf.compat.v1.get_variable_scope().reuse is False
        if d2Flag:
            s=2
            numDiscBlocks=3
        else:
            s=1
            numDiscBlocks=2
        h = lrelu(conv2d(image, options.df_dim, ks=3, s=1, name='dInitConv'))
        h = lrelu(batchnormSR(conv2d(h, options.df_dim, ks=3, s=s, name='dUpConv')))
        for i in range(numDiscBlocks):
            expon=2**(i+1)
            h = lrelu(batchnormSR(conv2d(h, options.df_dim*expon, ks=3, s=1, name=f'dBlock{i+1}Conv')))
            h = lrelu(batchnormSR(conv2d(h, options.df_dim*expon, ks=3, s=2, name=f'dBlock{i+1}UpConv')))
        h = conv2d(h, 1, ks=3, s=1, name='d_h3_pred')
        #h = lrelu(denselayer(slim.flatten(h), 1024, name="dFC1"))
        #h = denselayer(h, 1, name="dFCout")
        return h


def generator_unet(image, options, reuse=False, name="generator"): # this unet doesnt seem to have cross connections
    #TODO: convert to phase-shifted unet
    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.compat.v1.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        #else:
            #assert tf.compat.v1.get_variable_scope().reuse is False

        # image is (256 x 256 x input_c_dim)
        e1 = instance_norm(conv2d(image, options.gf_dim, name='g_e1_conv'))
        # e1 is (128 x 128 x self.gf_dim)
        e2 = instance_norm(conv2d(lrelu(e1), options.gf_dim*2, name='g_e2_conv'), 'g_bn_e2')
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = instance_norm(conv2d(lrelu(e2), options.gf_dim*4, name='g_e3_conv'), 'g_bn_e3')
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = instance_norm(conv2d(lrelu(e3), options.gf_dim*8, name='g_e4_conv'), 'g_bn_e4')
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = instance_norm(conv2d(lrelu(e4), options.gf_dim*8, name='g_e5_conv'), 'g_bn_e5')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = instance_norm(conv2d(lrelu(e5), options.gf_dim*8, name='g_e6_conv'), 'g_bn_e6')
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = instance_norm(conv2d(lrelu(e6), options.gf_dim*8, name='g_e7_conv'), 'g_bn_e7')
        # e7 is (2 x 2 x self.gf_dim*8)
        e8 = instance_norm(conv2d(lrelu(e7), options.gf_dim*8, name='g_e8_conv'), 'g_bn_e8')
        # e8 is (1 x 1 x self.gf_dim*8)

        d1 = deconv2d(tf.nn.relu(e8), options.gf_dim*8, name='g_d1')
        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = tf.concat([instance_norm(d1, 'g_bn_d1'), e7], 3)
        # d1 is (2 x 2 x self.gf_dim*8*2)

        d2 = deconv2d(tf.nn.relu(d1), options.gf_dim*8, name='g_d2')
        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = tf.concat([instance_norm(d2, 'g_bn_d2'), e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        d3 = deconv2d(tf.nn.relu(d2), options.gf_dim*8, name='g_d3')
        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = tf.concat([instance_norm(d3, 'g_bn_d3'), e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        d4 = deconv2d(tf.nn.relu(d3), options.gf_dim*8, name='g_d4')
        d4 = tf.concat([instance_norm(d4, 'g_bn_d4'), e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = deconv2d(tf.nn.relu(d4), options.gf_dim*4, name='g_d5')
        d5 = tf.concat([instance_norm(d5, 'g_bn_d5'), e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = deconv2d(tf.nn.relu(d5), options.gf_dim*2, name='g_d6')
        d6 = tf.concat([instance_norm(d6, 'g_bn_d6'), e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        d7 = deconv2d(tf.nn.relu(d6), options.gf_dim, name='g_d7')
        d7 = tf.concat([instance_norm(d7, 'g_bn_d7'), e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        d8 = deconv2d(tf.nn.relu(d7), options.output_c_dim, name='g_d8')
        # d8 is (256 x 256 x output_c_dim)

        return tf.nn.tanh(d8)


def generator_resnet(image, options, reuse=False, name="generator", g3Flag=False): 
#TODO: checkerboarding is quite extreme, switch to subpixel conv

    with tf.compat.v1.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        #else:
            #assert tf.compat.v1.get_variable_scope().reuse is False

        def residualBlock(x, dim, ks=3, s=1, name='res'):
            #p = int((ks - 1) / 2)
            #y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(x, dim, ks, s, padding='SAME', name=name+'_c1'), name+'_bn1')
            #y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='SAME', name=name+'_c2'), name+'_bn2')
            return y + x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        #c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(image, options.gf_dim, 7, 1, padding='SAME', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        # define G network with 9 resnet blocks
        r1 = residualBlock(c3, options.gf_dim*4, name='g_r1')
        r2 = residualBlock(r1, options.gf_dim*4, name='g_r2')
        r3 = residualBlock(r2, options.gf_dim*4, name='g_r3')
        r4 = residualBlock(r3, options.gf_dim*4, name='g_r4')
        r5 = residualBlock(r4, options.gf_dim*4, name='g_r5')
        r6 = residualBlock(r5, options.gf_dim*4, name='g_r6')
        r7 = residualBlock(r6, options.gf_dim*4, name='g_r7')
        r8 = residualBlock(r7, options.gf_dim*4, name='g_r8')
        r9 = residualBlock(r8, options.gf_dim*4, name='g_r9')
        stride=2
        if g3Flag:
            stride=1
        d1 = deconv2d(r9, options.gf_dim*2, 3, stride, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim, 3, stride, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        #d2 = tf.pad(d2, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d2, options.output_c_dim, 7, 1, padding='SAME', name='g_pred_c'))
        return pred

def segNetYDW(image, options, reuse=False, name='segNetYDW', numResBlocks=16): # the original segnet
    with tf.compat.v1.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        dropout_rate = 0.5 if options.is_training else 1.0
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        else:
            assert tf.compat.v1.get_variable_scope().reuse is False
              
        x = tf.nn.lrn(image, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')
        
        x1 = tf.nn.max_pool(tf.nn.dropout(tf.nn.relu(batchnormSR(conv2d(x, options.gf_dim, 7, 1, padding='SAME', name='g_e1_c'))), dropout_rate), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        x2 = tf.nn.max_pool(tf.nn.dropout(tf.nn.relu(batchnormSR(conv2d(x1, options.gf_dim*2, 7, 1, padding='SAME', name='g_e2_c'))), dropout_rate), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        x3 = tf.nn.max_pool(tf.nn.dropout(tf.nn.relu(batchnormSR(conv2d(x2, options.gf_dim*4, 7, 1, padding='SAME', name='g_e3_c'))), dropout_rate), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
        x4 = tf.nn.max_pool(tf.nn.dropout(tf.nn.relu(batchnormSR(conv2d(x3, options.gf_dim*8, 7, 1, padding='SAME', name='g_e4_c'))), dropout_rate), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
        
        x = deconv2d(x4, options.gf_dim, 2, 2, name='g_e5_dc')
        x = tf.nn.dropout(tf.nn.relu(batchnormSR(conv2d(x, options.gf_dim*8, 7, 1, padding='SAME', name='g_e5_c'))), dropout_rate)
        
        x = deconv2d(x, options.gf_dim, 2, 2, name='g_e6_dc')
        x = tf.nn.dropout(tf.nn.relu(batchnormSR(conv2d(x, options.gf_dim*4, 7, 1, padding='SAME', name='g_e6_c'))), dropout_rate)
        
        x = deconv2d(x, options.gf_dim, 2, 2, name='g_e7_dc')
        x = tf.nn.dropout(tf.nn.relu(batchnormSR(conv2d(x, options.gf_dim*2, 7, 1, padding='SAME', name='g_e7_c'))), dropout_rate)
        
        x = deconv2d(x, options.gf_dim, 2, 2, name='g_e8_dc')
        x = tf.nn.dropout(tf.nn.relu(batchnormSR(conv2d(x, options.gf_dim, 7, 1, padding='SAME', name='g_e8_c'))), dropout_rate)
        
        pred = conv2d(x, options.num_classes, 1, 1, padding='SAME', name='nin_pix_class')
        
        return pred
        
def segNetYDW2(image, options, reuse=False, name='segNetYDW', numResBlocks=16): # this is resnet segnet
    with tf.compat.v1.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        else:
            assert tf.compat.v1.get_variable_scope().reuse is False
        dropout_rate = 0.5 if options.is_training else 1.0
        def residualBlock(x, dim, ks=7, s=1, name='resBlock', dropout_rate=0.5):
            y = x
            y = batchnormSR(conv2d(y, dim, ks, s, padding='SAME', name=name+'_c1'))
            y = tf.nn.relu(y)
            y = tf.nn.dropout(batchnormSR(conv2d(y, dim, ks, s, padding='SAME', name=name+'_c2')), dropout_rate)
            return y + x
        # encode the image with an initial conv layer
        c0 = conv2d(image, options.srf_dim, 7, 1, name='g_e_shallow_c')
        shallow=c0
        # pass through the residual blocks
        for i in range(1, numResBlocks+1):
            c0 = residualBlock(c0, options.srf_dim, name='g_residual_%d'%(i), dropout_rate=dropout_rate)
        # output conv
        deep = tf.nn.relu(batchnormSR(conv2d(c0, options.srf_dim, 7, 1, name='g_e_deep_c')))

        c0 = deep# + shallow
        
        c0 = conv2d(c0, options.num_classes, 1, 1, padding='SAME', name='g_pred_c')
        return c0
        
        
def segNetYDW3(image, options, reuse=False, name="segNetYDW", numResBlocks=16): #  a Unet segnet with k=4

    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.compat.v1.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        else:
            assert tf.compat.v1.get_variable_scope().reuse is False

        # image is (256 x 256 x input_c_dim)
        e1 = batchnormSR(conv2d(image, options.gf_dim, name='g_e1_conv'))
        # e1 is (128 x 128 x self.gf_dim)
        e2 = batchnormSR(conv2d(lrelu(e1), options.gf_dim*2, name='g_e2_conv'))
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = batchnormSR(conv2d(lrelu(e2), options.gf_dim*4, name='g_e3_conv'))
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = batchnormSR(conv2d(lrelu(e3), options.gf_dim*8, name='g_e4_conv'))

        d4 = deconv2d(tf.nn.relu(e4), options.gf_dim*8, name='g_d4')
        d4 = tf.nn.dropout(d4, dropout_rate)
        d4 = tf.concat([batchnormSR(d4), e3], 3)


        d5 = deconv2d(tf.nn.relu(d4), options.gf_dim*4, name='g_d5')
        d5 = tf.nn.dropout(d5, dropout_rate)
        d5 = tf.concat([batchnormSR(d5), e2], 3)


        d6 = deconv2d(tf.nn.relu(d5), options.gf_dim*2, name='g_d6')
        d6 = tf.nn.dropout(d6, dropout_rate)
        d6 = tf.concat([batchnormSR(d6), e1], 3)


        d8 = deconv2d(tf.nn.relu(d6), options.num_classes, name='g_d8')


        return d8
        
def segNetYDW4(image, options, reuse=False, name="segNetYDW", numResBlocks=16): #  a Res-Unet-segnet with k=4
    resSkip=options.resSkip
    Uskip=options.Uskip
    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.compat.v1.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        else:
            assert tf.compat.v1.get_variable_scope().reuse is False
        
        def residualBlock(x, dim, ks=4, s=1, name='resBlock', dropout_rate=0.5, resSkip=False):
            if resSkip:
                x = tf.nn.relu(conv2d(x, dim, 1, 1, padding='SAME', name=name+'_c0'))
            y = tf.nn.relu(batchnormSR(conv2d(x, dim, ks, s, padding='SAME', name=name+'_c1')))
            if resSkip:
                y = tf.nn.dropout(tf.nn.relu(batchnormSR(conv2d(y, dim, ks, s, padding='SAME', name=name+'_c2'))), dropout_rate)
                return y + x    
            else:
                return tf.nn.dropout(y, dropout_rate)
        
        x = image
        # pool, conv, conv, save
        x1 = residualBlock(x, options.gf_dim, ks=4, s=1, name='resBlock1', dropout_rate=dropout_rate, resSkip=False)
        
        x2 = residualBlock(tf.nn.max_pool(x1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1'), options.gf_dim*2, ks=4, s=1, name='resBlock2', dropout_rate=dropout_rate, resSkip=resSkip)
        
        x3 = residualBlock(tf.nn.max_pool(x2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2'), options.gf_dim*4, ks=4, s=1, name='resBlock3', dropout_rate=dropout_rate, resSkip=resSkip)

        x4 = residualBlock(tf.nn.max_pool(x3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3'), options.gf_dim*8, ks=4, s=1, name='resBlock4', dropout_rate=dropout_rate, resSkip=resSkip)
   
        # trough conv conv
        x5 = residualBlock(tf.nn.max_pool(x4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4'), options.gf_dim*16, ks=4, s=1, name='resBlockT', dropout_rate=dropout_rate, resSkip=resSkip)
               
        # updeconv, concat, conv, conv
        y1 = batchnormSR(deconv2d(x5, options.gf_dim*8, ks=4, name='g_d4'))
        if Uskip:
            y1 = tf.concat([y1, x4], 3)
        y1 = residualBlock(y1, options.gf_dim*8, ks=4, s=1, name='resBlock5', dropout_rate=dropout_rate, resSkip=resSkip)
        
        y2 = batchnormSR(deconv2d(y1, options.gf_dim*4, ks=4, name='g_d5'))
        if Uskip:
            y2 = tf.concat([y2, x3], 3)
        y2 = residualBlock(y2, options.gf_dim*4, ks=4, s=1, name='resBlock6', dropout_rate=dropout_rate, resSkip=resSkip)
        
        y3 = batchnormSR(deconv2d(y2, options.gf_dim*2, ks=4, name='g_d6'))
        if Uskip:
            y3 = tf.concat([y3, x2], 3)
        y3 = residualBlock(y3, options.gf_dim*2, ks=4, s=1, name='resBlock7', dropout_rate=dropout_rate, resSkip=resSkip)
        
        y4 = batchnormSR(deconv2d(y3, options.gf_dim, ks=4, name='g_d7'))
        if Uskip:
            y4 = tf.concat([y4, x1], 3)
        y4 = residualBlock(y4, options.gf_dim, ks=4, s=1, name='resBlock8', dropout_rate=dropout_rate, resSkip=resSkip)
        
        #output
        d8 = conv2d(y4, options.num_classes, 1, 1, padding='SAME', name='g_pred_c')
        return d8


def segNetYDW3D(image, options, reuse=False, name="segNetYDW3D", numResBlocks=16): #  a Res-Unet-segnet with k=4
    resSkip=options.resSkip
    Uskip=options.Uskip
    dropout_rate = 0.5 #if options.is_training else 1.0 # there is a bug associated with label 0 and the dropout
    with tf.compat.v1.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        else:
            assert tf.compat.v1.get_variable_scope().reuse is False
        
        def residualBlock(x, dim, ks=4, s=1, name='resBlock', dropout_rate=0.5, resSkip=False):
            if resSkip:
                x = tf.nn.relu(conv3d(x, dim, 1, 1, padding='SAME', name=name+'_c0'))
            y = tf.nn.relu(batchnormSR(conv3d(x, dim, ks, s, padding='SAME', name=name+'_c1')))
            if resSkip:
                y = tf.nn.dropout(tf.nn.relu(batchnormSR(conv3d(y, dim, ks, s, padding='SAME', name=name+'_c2'))), dropout_rate)
                return y + x    
            else:
                return tf.nn.dropout(y, dropout_rate)
        
        x = image
        # pool, conv, conv, save
        x1 = residualBlock(x, options.gf_dim, ks=4, s=1, name='resBlock1', dropout_rate=dropout_rate, resSkip=False)
        
        x2 = residualBlock(tf.nn.max_pool3d(x1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1'), options.gf_dim*2, ks=4, s=1, name='resBlock2', dropout_rate=dropout_rate, resSkip=resSkip)
        
        x3 = residualBlock(tf.nn.max_pool3d(x2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool2'), options.gf_dim*4, ks=4, s=1, name='resBlock3', dropout_rate=dropout_rate, resSkip=resSkip)

        x4 = residualBlock(tf.nn.max_pool3d(x3, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool3'), options.gf_dim*8, ks=4, s=1, name='resBlock4', dropout_rate=dropout_rate, resSkip=resSkip)
   
        # trough conv conv
        x5 = residualBlock(tf.nn.max_pool3d(x4, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool4'), options.gf_dim*16, ks=4, s=1, name='resBlockT', dropout_rate=dropout_rate, resSkip=resSkip)
               
        # updeconv, concat, conv, conv
        y1 = batchnormSR(deconv3d(x5, options.gf_dim*8, ks=4, name='g_d4'))
        if Uskip:
            y1 = tf.concat([y1, x4], 4)
        y1 = residualBlock(y1, options.gf_dim*8, ks=4, s=1, name='resBlock5', dropout_rate=dropout_rate, resSkip=resSkip)
        
        y2 = batchnormSR(deconv3d(y1, options.gf_dim*4, ks=4, name='g_d5'))
        if Uskip:
            y2 = tf.concat([y2, x3], 4)
        y2 = residualBlock(y2, options.gf_dim*4, ks=4, s=1, name='resBlock6', dropout_rate=dropout_rate, resSkip=resSkip)
        
        y3 = batchnormSR(deconv3d(y2, options.gf_dim*2, ks=4, name='g_d6'))
        if Uskip:
            y3 = tf.concat([y3, x2], 4)
        y3 = residualBlock(y3, options.gf_dim*2, ks=4, s=1, name='resBlock7', dropout_rate=dropout_rate, resSkip=resSkip)
        
        y4 = batchnormSR(deconv3d(y3, options.gf_dim, ks=4, name='g_d7'))
        if Uskip:
            y4 = tf.concat([y4, x1], 4)
        y4 = residualBlock(y4, options.gf_dim, ks=4, s=1, name='resBlock8', dropout_rate=dropout_rate, resSkip=resSkip)
        
        #output
        d8 = conv3d(y4, options.num_classes, 1, 1, padding='SAME', name='g_pred_c')

        return d8   
        '''
def segNetYDW3D(image, options, reuse=False, name="segNetYDW3D", numResBlocks=16): #  a Res-Unet-segnet with k=4

    resSkip=options.resSkip
    Uskip=options.Uskip
    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.compat.v1.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        else:
            assert tf.compat.v1.get_variable_scope().reuse is False

        def residualBlock(x, dim_in, dim, ks=4, s=1, name='resBlock', dropout_rate=0.5, resSkip=False, is_train=True):
            if resSkip:
                x = Conv3dLayer(x, shape = [1, 1, 1, dim_in, dim], strides=[1, s, s, s, 1], padding='SAME', act=tf.nn.relu, name=name+'ConvResIn')
                #x = tf.nn.relu(conv3d(x, dim, 1, 1, padding='SAME'))
                dim_in=dim
            y = Conv3dLayer(x, shape = [ks, ks, ks, dim_in, dim], strides=[1, s, s, s, 1], padding='SAME', act=None, name=name+'ConvRes1')
            y = BatchNormLayer(y, act=tf.nn.relu, is_train=is_train, name=name+'BN1')
            if resSkip:
                y = Conv3dLayer(y, shape = [ks, ks, ks, dim_in, dim], strides=[1, s, s, s, 1], padding='SAME', act=None, name=name+'ConvRes2')
                y = BatchNormLayer(y, act=tf.nn.relu, is_train=is_train, name=name+'BN2')
                y = DropoutLayer(y, keep=1-dropout_rate, is_train=is_train, name=name+'Drop')
            
                #y = tf.nn.dropout(tf.nn.relu(batchnormSR(conv3d(y, dim, ks, s, padding='SAME'))), dropout_rate)
                return ElementwiseLayer([x, y], tf.add, name=name+'ShortSkip')    
            else:
                return DropoutLayer(y, keep=1-dropout_rate, is_train=is_train, name=name+'Drop')
        
        x = InputLayer(image, name='Input Tensor')
        # pool, conv, conv, save
        x1 = residualBlock(x, 1, options.gf_dim, ks=4, s=1, name='resBlock1', dropout_rate=dropout_rate, resSkip=False, is_train=options.is_training)
        x2 = PoolLayer(x1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool3d, name='Pool1')
        
        x2 = residualBlock(x2, options.gf_dim, options.gf_dim*2, ks=4, s=1, name='resBlock2', dropout_rate=dropout_rate, resSkip=resSkip, is_train=options.is_training)
        x3 = PoolLayer(x2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool3d,name='Pool2')
        
        x3 = residualBlock(x3, options.gf_dim*2, options.gf_dim*4, ks=4, s=1, name='resBlock3', dropout_rate=dropout_rate, resSkip=resSkip, is_train=options.is_training)
        x4 = PoolLayer(x3, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool3d,name='Pool3')

        x4 = residualBlock(x4, options.gf_dim*4, options.gf_dim*8, ks=4, s=1, name='resBlock4', dropout_rate=dropout_rate, resSkip=resSkip, is_train=options.is_training)
        x5 = PoolLayer(x4, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', pool=tf.nn.max_pool3d,name='Pool4')
   
        # trough conv conv
        x5 = residualBlock(x5, options.gf_dim*8, options.gf_dim*16, ks=4, s=1, name='resBlockT', dropout_rate=dropout_rate, resSkip=resSkip, is_train=options.is_training)
               
        # updeconv, concat, conv, conv (if deconv doesnt work, use keras upsampling)
        y1 = DeConv3dLayer(x5, act=None, shape=[4, 4, 4, options.gf_dim*8, options.gf_dim*16], output_shape = [None,None,None,None,None], strides=[1, 2, 2, 2, 1], name='Deconv1')
        y1 = BatchNormLayer(y1, act=None, is_train=options.is_training, name='Deconv1BN2')
        in_dims=options.gf_dim*8
        #y1 = batchnormSR(deconv2d(x5, options.gf_dim*8, ks=4, name='g_d4'))
        if Uskip:
            y1 = ConcatLayer([y1, x4], 4, name='concat1')
            in_dims=in_dims*2
        y1 = residualBlock(y1, in_dims, options.gf_dim*8, ks=4, s=1, name='resBlock5', dropout_rate=dropout_rate, resSkip=resSkip, is_train=options.is_training)
        
        y2 = DeConv3dLayer(y1, act=None, shape=[4, 4, 4, options.gf_dim*4, options.gf_dim*8], strides=[1, 2, 2, 2, 1], name='Deconv2')
        y2 = BatchNormLayer(y2, act=None, is_train=options.is_training, name='Deconv2BN2')
        in_dims=options.gf_dim*4
        #y1 = batchnormSR(deconv2d(x5, options.gf_dim*8, ks=4, name='g_d4'))
        if Uskip:
            y2 = ConcatLayer([y2, x3], 4, name='concat2')
            in_dims=in_dims*2
        y2 = residualBlock(y2, in_dims, options.gf_dim*4, ks=4, s=1, name='resBlock6', dropout_rate=dropout_rate, resSkip=resSkip, is_train=options.is_training)
        
        y3 = DeConv3dLayer(y2, act=None, shape=[4, 4, 4, options.gf_dim*2, options.gf_dim*4], strides=[1, 2, 2, 2, 1], name='Deconv3')
        y3 = BatchNormLayer(y3, act=None, is_train=options.is_training, name='Deconv3BN2')
        in_dims=options.gf_dim*2
        #y1 = batchnormSR(deconv2d(x5, options.gf_dim*8, ks=4, name='g_d4'))
        if Uskip:
            y3 = ConcatLayer([y3, x2], 4, name='concat3')
            in_dims=in_dims*2
        y3 = residualBlock(y3, in_dims, options.gf_dim*2, ks=4, s=1, name='resBlock7', dropout_rate=dropout_rate, resSkip=resSkip, is_train=options.is_training)
        
        
        
        y4 = DeConv3dLayer(y3, act=None, shape=[4, 4, 4, options.gf_dim, options.gf_dim*2], strides=[1, 2, 2, 2, 1], name='Deconv4')
        y4 = BatchNormLayer(y4, act=None, is_train=options.is_training, name='Deconv4BN2')
        in_dims=options.gf_dim
        #y1 = batchnormSR(deconv2d(x5, options.gf_dim*8, ks=4, name='g_d4'))
        if Uskip:
            y4 = ConcatLayer([y4, x1], 4, name='concat4')
            in_dims=in_dims*2
        y4 = residualBlock(y4, in_dims, options.gf_dim, ks=4, s=1, name='resBlock8', dropout_rate=dropout_rate, resSkip=resSkip, is_train=options.is_training)
        
        #output
        d8 = Conv3dLayer(y4, shape = [1, 1, 1, options.gf_dim, options.num_classes], strides=[1, 1, 1, 1, 1], padding='SAME', act=None, name=name+'LastLayer')
        #d8 = conv2d(y4, options.num_classes, 1, 1, padding='SAME', name='g_pred_c')
        return d8.outputs
'''

def uResNetp2p(image, options, reuse=False, name="UResNetYDW", numResBlocks=16): #  a Res-Unet-segnet with k=4
    resSkip=options.resSkip
    Uskip=options.Uskip
    dropout_rate = 0.5 if options.is_training else 1.0
    with tf.compat.v1.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        else:
            assert tf.compat.v1.get_variable_scope().reuse is False
        
        def residualBlock(x, dim, ks=4, s=1, name='resBlock', dropout_rate=0.5, resSkip=False):
            if resSkip:
                x = tf.nn.relu(conv2d(x, dim, 1, 1, padding='SAME', name=name+'_c0'))
            y = tf.nn.relu(batchnormSR(conv2d(x, dim, ks, s, padding='SAME', name=name+'_c1')))
            if resSkip:
                y = tf.nn.dropout(tf.nn.relu(batchnormSR(conv2d(y, dim, ks, s, padding='SAME', name=name+'_c2'))), dropout_rate)
                return y + x    
            else:
                return tf.nn.dropout(y, dropout_rate)
        
        x = image
        # pool, conv, conv, save
        x1 = residualBlock(x, options.gf_dim, ks=4, s=1, name='resBlock1', dropout_rate=dropout_rate, resSkip=False)
        
        x2 = residualBlock(tf.nn.max_pool(x1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1'), options.gf_dim*2, ks=4, s=1, name='resBlock2', dropout_rate=dropout_rate, resSkip=resSkip)
        
        x3 = residualBlock(tf.nn.max_pool(x2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2'), options.gf_dim*4, ks=4, s=1, name='resBlock3', dropout_rate=dropout_rate, resSkip=resSkip)

        x4 = residualBlock(tf.nn.max_pool(x3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3'), options.gf_dim*8, ks=4, s=1, name='resBlock4', dropout_rate=dropout_rate, resSkip=resSkip)
   
        # trough conv conv
        x5 = residualBlock(tf.nn.max_pool(x4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4'), options.gf_dim*16, ks=4, s=1, name='resBlockT', dropout_rate=dropout_rate, resSkip=resSkip)
               
        # updeconv, concat, conv, conv
        y1 = batchnormSR(deconv2d(x5, options.gf_dim*8, ks=4, name='g_d4'))
        if Uskip:
            y1 = tf.concat([y1, x4], 3)
        y1 = residualBlock(y1, options.gf_dim*8, ks=4, s=1, name='resBlock5', dropout_rate=dropout_rate, resSkip=resSkip)
        
        y2 = batchnormSR(deconv2d(y1, options.gf_dim*4, ks=4, name='g_d5'))
        if Uskip:
            y2 = tf.concat([y2, x3], 3)
        y2 = residualBlock(y2, options.gf_dim*4, ks=4, s=1, name='resBlock6', dropout_rate=dropout_rate, resSkip=resSkip)
        
        y3 = batchnormSR(deconv2d(y2, options.gf_dim*2, ks=4, name='g_d6'))
        if Uskip:
            y3 = tf.concat([y3, x2], 3)
        y3 = residualBlock(y3, options.gf_dim*2, ks=4, s=1, name='resBlock7', dropout_rate=dropout_rate, resSkip=resSkip)
        
        y4 = batchnormSR(deconv2d(y3, options.gf_dim, ks=4, name='g_d7'))
        if Uskip:
            y4 = tf.concat([y4, x1], 3)
        y4 = residualBlock(y4, options.gf_dim, ks=4, s=1, name='resBlock8', dropout_rate=dropout_rate, resSkip=resSkip)
        
        #output
        d8 = (conv2d(y4, options.num_classes, 1, 1, padding='SAME', name='g_pred_c'))
        return d8


def uResNetp2p3D(image, options, reuse=False, name="uResNetp2p3D", numResBlocks=16): #  a Res-Unet-segnet with k=4
    resSkip=options.resSkip
    Uskip=options.Uskip
    dropout_rate = 0.5 #if options.is_training else 1.0 # there is a bug associated with label 0 and the dropout
    with tf.compat.v1.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.compat.v1.get_variable_scope().reuse_variables()
        else:
            assert tf.compat.v1.get_variable_scope().reuse is False
        
        def residualBlock(x, dim, ks=4, s=1, name='resBlock', dropout_rate=0.5, resSkip=False):
            if resSkip:
                x = tf.nn.relu(conv3d(x, dim, 1, 1, padding='SAME', name=name+'_c0'))
            y = tf.nn.relu(batchnormSR(conv3d(x, dim, ks, s, padding='SAME', name=name+'_c1')))
            if resSkip:
                y = tf.nn.dropout(tf.nn.relu(batchnormSR(conv3d(y, dim, ks, s, padding='SAME', name=name+'_c2'))), dropout_rate)
                return y + x    
            else:
                return tf.nn.dropout(y, dropout_rate)
        
        x = image
        # pool, conv, conv, save
        x1 = residualBlock(x, options.gf_dim, ks=4, s=1, name='resBlock1', dropout_rate=dropout_rate, resSkip=False)
        
        x2 = residualBlock(tf.nn.max_pool3d(x1, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool1'), options.gf_dim*2, ks=4, s=1, name='resBlock2', dropout_rate=dropout_rate, resSkip=resSkip)
        
        x3 = residualBlock(tf.nn.max_pool3d(x2, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool2'), options.gf_dim*4, ks=4, s=1, name='resBlock3', dropout_rate=dropout_rate, resSkip=resSkip)

        x4 = residualBlock(tf.nn.max_pool3d(x3, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool3'), options.gf_dim*8, ks=4, s=1, name='resBlock4', dropout_rate=dropout_rate, resSkip=resSkip)
   
        # trough conv conv
        x5 = residualBlock(tf.nn.max_pool3d(x4, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME', name='pool4'), options.gf_dim*16, ks=4, s=1, name='resBlockT', dropout_rate=dropout_rate, resSkip=resSkip)
               
        # updeconv, concat, conv, conv
        y1 = batchnormSR(deconv3d(x5, options.gf_dim*8, ks=4, name='g_d4'))
        if Uskip:
            y1 = tf.concat([y1, x4], 4)
        y1 = residualBlock(y1, options.gf_dim*8, ks=4, s=1, name='resBlock5', dropout_rate=dropout_rate, resSkip=resSkip)
        
        y2 = batchnormSR(deconv3d(y1, options.gf_dim*4, ks=4, name='g_d5'))
        if Uskip:
            y2 = tf.concat([y2, x3], 4)
        y2 = residualBlock(y2, options.gf_dim*4, ks=4, s=1, name='resBlock6', dropout_rate=dropout_rate, resSkip=resSkip)
        
        y3 = batchnormSR(deconv3d(y2, options.gf_dim*2, ks=4, name='g_d6'))
        if Uskip:
            y3 = tf.concat([y3, x2], 4)
        y3 = residualBlock(y3, options.gf_dim*2, ks=4, s=1, name='resBlock7', dropout_rate=dropout_rate, resSkip=resSkip)
        
        y4 = batchnormSR(deconv3d(y3, options.gf_dim, ks=4, name='g_d7'))
        if Uskip:
            y4 = tf.concat([y4, x1], 4)
        y4 = residualBlock(y4, options.gf_dim, ks=4, s=1, name='resBlock8', dropout_rate=dropout_rate, resSkip=resSkip)
        
        #output
        d8 = tf.nn.tanh(conv3d(y4, options.num_classes, 1, 1, padding='SAME', name='g_pred_c'))

        return d8   

def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))

def mse_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    
def tot_var_criterion(image): # this is just the image gradient loss. it must be combined with the abs or mae criterion for balance
    return tf.reduce_mean(tf.image.total_variation(image))

def seg_criterion(logits, labels):
    #segWeights =  np.array([0.0921, 0.1081, 0.6430, 0.1573, 0.0035, 0.0023]) # class occurences, weigh them inversely
    segWeights =  np.array([1, 1, 1, 1, 1, 1]) # class occurences, weigh them inversely
    nDims = len(logits.shape)
    labels = tf.reshape(tf.one_hot(labels, logits.shape[nDims-1]), [-1, logits.shape[nDims-1]]) # flatten in batch and space dims
    logits = tf.reshape(logits, (-1,logits.shape[nDims-1]))
    epsilon = tf.constant(value=1e-10)

    logits = logits + epsilon
    softmax = tf.nn.softmax(logits)

    cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), 1.0/segWeights), axis=[1])   
    
    #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels, name='cross_entropy_per_example')
    return tf.reduce_mean(cross_entropy)

def psnr_metric(in_, target, max_val=2.0):
    return tf.image.psnr(in_, target, max_val=max_val)




