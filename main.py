import argparse
import os
import tensorflow as tf
import pdb
import numpy as np
tf.compat.v1.set_random_seed(19)
from cyclegan import cyclegan
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
        
def str2int(v):
    if v=='M':
        return v
    try:
        v = int(v)
    except:
        raise argparse.ArgumentTypeError('int value expected.')
    return v
    
def str2float(v):
    if v=='M':
        return v
    try:
        v = float(v)
    except:
        raise argparse.ArgumentTypeError('float value expected.')
    return v
        

# TODO modularise this into concentricGAN noise->clean->SR->seg->vels
# TODO change semseg to scale data from -1 to 1?
parser = argparse.ArgumentParser(description='')
#training data arguments
parser.add_argument('--gpuIDs', dest='gpuIDs', type=str, default='1', help='IDs for the GPUs. Empty for CPU. Nospaces')
parser.add_argument('--dataset_dir', dest='dataset_dir', default='segSimonRock_BIN', help='path of the dataset')
parser.add_argument('--load2ram', dest='load2ram', type=bool, default=False, help='load dataset into ram')
parser.add_argument('--epoch', dest='epoch', type=int, default=1000, help='# of epoch')
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=25, help='# of epoch to decay lr')
parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='momentum term of adam')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='# images in batch')
parser.add_argument('--train_size', dest='train_size', type=int, default=8000, help='# images used to train per epoch')
parser.add_argument('--iterNum', dest='iterNum', type=str2int, default=1000, help='# iterations per epoch')
parser.add_argument('--load_size', dest='load_size', type=int, default=286, help='scale images to this size') # only active if SC1GAN, this is turned off for C2GAN and ACGAN
parser.add_argument('--fine_size', dest='fine_size', type=str2int, default=256, help='then crop to this size')
parser.add_argument('--nDims', dest='nDims', type=str2int, default=2, help='2D or 3D inputs and outputs')
parser.add_argument('--ngf', dest='ngf', type=int, default=32, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=8, help='# of discri filters in first conv layer')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='# of image channels for A')# 1 for 3D,m 3 for 2D colour
parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of image channels for B') # 1 for seg, 3 for SR etc etc

parser.add_argument('--phase', dest='phase', default='train', help='train, test')
parser.add_argument('--save_freq', dest='save_freq', type=int, default=10, help='save a model every save_freq epochs')
parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the validation images every X epochs')
parser.add_argument('--continue_train', dest='continue_train', type=str2bool, default=True, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--save_dir', dest='save_dir', default=None, help='models are saved here, if none, will generate based on some params')
parser.add_argument('--model_dir', dest='model_dir', default=None, help='models are loaded from here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='samples are saved here')

parser.add_argument('--use_resnet', dest='use_resnet', type=bool, default=True, help='generation network using residual block')
parser.add_argument('--use_lsgan', dest='use_lsgan', type=str2bool, default=True, help='gan loss defined in lsgan') # patchGAN plays poorly with scgan
# symcycleganFlags
parser.add_argument('--c1ganFlag', dest='c1ganFlag', type=bool, default=False, help='flag for training a symmetric type cyclegan network')
# SRGAN arguments
parser.add_argument('--srganFlag', dest='srganFlag', type=bool, default=True, help='flag for training a feed forward network')
# asymCGAN arguments
parser.add_argument('--ACGANFlag', dest='ACGANFlag', type=bool, default=False, help='flag for training the Asymetric cyclegan network')
# C2GAN arguments
parser.add_argument('--c2ganFlag', dest='c2ganFlag', type=bool, default=False, help='flag for training the c2gan network')
# asymmetric models
parser.add_argument('--acType', dest='acType', type=str, default='semSeg', help='which model is asymetric, semSeg, or superRes')
parser.add_argument('--segRes', dest='segRes', type=str2bool, default=False, help='segnet has res skips')
parser.add_argument('--segU', dest='segU', type=str2bool, default=False, help='segnet has u skips')
parser.add_argument('--numClasses', dest='numClasses', type=str2int, default=6, help='number of semantic classes for segmentation')
parser.add_argument('--use_gan', dest='use_gan', type=str2bool, default=False, help='if srgan has gan active')
# SR arguments
parser.add_argument('--nsrf', dest='nsrf', type=int, default=64, help='# of SR filters in first conv layer')
parser.add_argument('--numResBlocks', dest='numResBlocks', type=int, default=16, help='# of resBlocks in EDSR')
parser.add_argument('--sr_nc', dest='sr_nc', type=int, default=3, help='# of image channels for C') #add this for hyperspectral support
# loss coefficients
parser.add_argument('--L1_lambda', dest='L1_lambda', type=str2float, default=10.0, help='weight on L1 term for normal cycle')
parser.add_argument('--idt_lambda', dest='idt_lambda', type=str2float, default=0.0, help='weight assigned to the a2b identity loss function') # b2b should give b
parser.add_argument('--tv_lambda', dest='tv_lambda', type=str2float, default=0.0, help='weight assigned to the a2b total variation loss function')
parser.add_argument('--L1_sr_lambda', dest='L1_sr_lambda', type=str2float, default=10.0, help='weight on L1 term in the SR cycle') # low since patchGAN doesnt have dense summation?
parser.add_argument('--glcm_sr_lambda', dest='glcm_sr_lambda', type=str2float, default=0.0, help='weight on glcm term in the SR cycle')
parser.add_argument('--idt_sr_lambda', dest='idt_sr_lambda', type=str2float, default=0.0, help='weight assigned to the SR identity loss function')
parser.add_argument('--tv_sr_lambda', dest='tv_sr_lambda', type=str2float, default=0.0, help='weight assigned to the SR total variation loss function') # this is a crutch. avoid it. if needed, tune it carefuly. div2k accepts 1-2e-4, amd fails at 1e-3 vs 10


# testing arguments
parser.add_argument('--testInputs', dest='testInputs', default='./datasets/grey2seg/testA', help='test input images are here')
parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
args = parser.parse_args()

gpuList=args.gpuIDs
args.numGPUs = len(gpuList.split(','))
if args.gpuIDs == '':
    os.environ["CUDA_VISIBLE_DEVICES"]=gpuList
def main():
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    gpu_options = tf.compat.v1.GPUOptions(visible_device_list=gpuList)
    tfconfig = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    with tf.compat.v1.Session(config=tfconfig) as sess:
        model = cyclegan(sess, args)
        #pdb.set_trace()
        model.train(args) if args.phase == 'train' else model.test(args)

if __name__ == '__main__':
    main()
