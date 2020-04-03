from __future__ import division
import os
import time
import pdb
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple
from sys import stdout
from module import *
from utils import *
import datetime
from ops import *
from tifffile import imsave
import random
#from skimage.measure import compare_psnr as psnr
from glcmLosses import *
'''
with modifications by YDW

this implementation is really dumb. make build model, train, and test on a cngan framework, and define each sectoin better

this script has mutated into an out of control monster

TODO: add asymmetric semantic segmentation routines to this as well as c3gan kek the final form should be something like concentricGAN noise->clean->SR->seg->vels->phase

'''

# define the model as an object
class cyclegan(object):
    # define the initial state of the object
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.L1_lambda = args.L1_lambda
        self.dataset_dir = args.dataset_dir
        # from module, choose resnet im2im or unet im2im
        if args.c1ganFlag: #these are the symmetric generators
            if args.use_resnet:
                self.generator = generator_resnet
            else:
                self.generator = generator_unet
            self.discriminator = discriminator # the c1gan style type
        self.criterionGEN = abs_criterion # thus far, all reverse cycles use L1
        if args.use_lsgan: 
            print('LSGAN is active')
            self.criterionGAN = mse_criterion # LSGAN
        else:
            print('SigmoidGAN is active')
            self.criterionGAN = sce_criterion # SigmoidGAN
        # load the sr generator 
        if args.c2ganFlag or args.ACGANFlag or args.srganFlag: # these are the asymmetric generators with varying degrees of cyclicity
            if args.acType == 'superRes':
                self.generator = generator_resnetYDW # asymmetric/non-pooling resnet
                self.generatorSR = edsrYDW # SRCNN
                self.discriminator = discriminatorSR # SRGAN disc type
                self.criterionGENAB = abs_criterion
            elif args.acType == 'semSeg':
                self.generator = generator_resnet # the BA direction should resemble pix2pix
                if args.nDims == 2:
                    self.generatorSR = segNetYDW4 # the AB direction is segnet
                elif args.nDims == 3:
                    self.generatorSR = segNetYDW3D # the AB direction is segnet
                self.discriminator = discriminatorSR # SRGAN disc type for p2p
                self.criterionGENAB = seg_criterion # segnet uses softmax cross entropy
            elif args.acType == 'p2p':
                self.generator = generator_resnet # the BA direction should resemble pix2pix
                if args.nDims == 2:
                    self.generatorSR = uResNetp2p # the AB direction is segnet
                elif args.nDims == 3:
                    self.generatorSR = uResNetp2p3D # the AB direction is segnet
                self.discriminator = discriminatorSR # SRGAN disc type for p2p
                self.criterionGENAB = mse_criterion # segnet uses softmax cross entropy
            self.sr_c_dim = args.sr_nc
            self.idt_lambda=args.idt_lambda
            self.tv_lambda=args.tv_lambda
            self.L1_sr_lambda=args.L1_sr_lambda
            self.idt_sr_lambda=args.idt_sr_lambda
            self.tv_sr_lambda=args.tv_sr_lambda
            self.numResBlocks=args.numResBlocks
        # pass arguments into options object
        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size nDims gf_dim df_dim srf_dim output_c_dim output_sr_dim num_classes is_training is_c2gan is_acgan acType is_srgan is_gan is_c1gan glcmRatio resSkip Uskip')
        self.options = OPTIONS._make((args.batch_size, args.fine_size, args.nDims, args.ngf, args.ndf, args.nsrf, args.output_nc, args.sr_nc, args.numClasses, args.phase == 'train', args.c2ganFlag, args.ACGANFlag, args.acType, args.srganFlag, args.use_gan, args.c1ganFlag, args.glcm_sr_lambda, args.segRes, args.segU))

    def initialiseInputPlaceholders(self):
        if self.options.nDims == 2:
            inputShape=[None, None, None, self.input_c_dim + self.output_c_dim]
            inputShapeA=[None, None, None, self.input_c_dim]
            inputShapeB=[None, None, None, self.output_c_dim]
            inputShapeC=[None, None, None, self.sr_c_dim]
        elif self.options.nDims == 3:
            inputShape=[None, None, None, None, self.input_c_dim + self.output_c_dim]
            inputShapeA=[None, None, None, None, self.input_c_dim]
            inputShapeB=[None, None, None, None, self.output_c_dim]
            inputShapeC=[None, None, None, None, self.sr_c_dim] 
        if self.options.is_c1gan or self.options.is_c2gan: # for cases with AB same
            self.real_data = tf.compat.v1.placeholder(tf.float32, inputShape, name='real_A_and_B_images')
            if self.options.is_c2gan:
                self.real_C = tf.compat.v1.placeholder(tf.float32, inputShapeC, name='real_C_images')# the SR cycle
        elif self.options.is_acgan or self.options.is_srgan: # for cases AB different
            self.real_A = tf.compat.v1.placeholder(tf.float32, inputShapeA, name='real_A_images')
            if self.options.acType == 'superRes' or self.options.acType == 'p2p':
                self.real_B = tf.compat.v1.placeholder(tf.float32, inputShapeB, name='real_B_images_SR')
            elif self.options.acType == 'semSeg':
                self.real_B = tf.compat.v1.placeholder(tf.int32, inputShapeB, name='real_B_images_SR')
            
    def _build_model(self): # get graph, separate the conditionals into more coherent sectionS PLEASE 
        self.k = build_filter(factor=2)
        self.initialiseInputPlaceholders()
        # the AB section
        dBRatio=0
        if self.options.is_acgan or self.options.is_srgan: # in this case, we 
            #self.real_A.set_shape([self.options.batch_size, self.options.image_size, self.options.image_size, self.input_c_dim])
            #self.real_B.set_shape([self.options.batch_size, self.options.image_size*4, self.options.image_size*4, self.output_c_dim])
            # define the fake data as outputs of the Asymmetric generators. set the generator to reuse the same variables
            self.fake_B = self.generatorSR(self.real_A, self.options, False, name="generatorA2B_SR", numResBlocks=self.numResBlocks)# first half of cycle A2B
            if self.options.is_acgan:
                self.fake_A_ = self.generator(self.fake_B, self.options, reuse=False, name='generatorB_SR2A', g3Flag=True)# cycle back to A using g3
            elif self.options.is_srgan: # overwrite the ABA cycle
                self.fake_A_ = self.real_A
            if self.options.is_gan: # the B discrims
                # define the discriminators, these check the halfway outputs by classification accuracy
                self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB_SR", d2Flag = True)
                self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB_SR", d2Flag = True)
            else:
                self.DB_fake = tf.convert_to_tensor(0.0, name = 'discriminatorDummy')
                self.DB_real = tf.convert_to_tensor(0.0, name = 'discriminatorDummy2')
                dBRatio=0
        else: # for c1 and c2gans, the ab section is c1gan-like
            # pass the real A and B images to their separate containers
            self.real_A = self.real_data[:, :, :, :self.input_c_dim]
            self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
            #self.real_A.set_shape([self.options.batch_size, self.options.image_size, self.options.image_size, self.input_c_dim])
            #self.real_B.set_shape([self.options.batch_size, self.options.image_size, self.options.image_size, self.output_c_dim])
            # define the fake data as outputs of the symmetric generator. set the generator to reuse the same variables
            self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")# first half of cycle 
            self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")# second half of cycle, forming A2B2A
            # define the discriminators, these check the halfway outputs by classification accuracy
            self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
            self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.g_loss = 0
        self.d_loss = 0
        #g losses for the A2B2A cycle TODO: add support for L2 losses 
        self.g_loss_A_pixelwise = self.L1_lambda * self.criterionGEN(self.real_A, self.fake_A_)
        self.g_adv_B_loss = dBRatio*self.criterionGAN(self.DB_fake, tf.ones_like(self.DB_fake))
        self.g_loss = self.g_loss + self.g_loss_A_pixelwise + self.g_adv_B_loss
        #metrics
        if self.options.acType == 'superRes':
            self.gAPSNR = tf.image.psnr(self.real_A, self.fake_A_, max_val=2)
        else:
            self.gAPSNR = tf.convert_to_tensor(0.0)
        #d losses for the B outputs of the ABA cycle
        self.db_loss_real = dBRatio*self.criterionGAN(self.DB_real, tf.ones_like(self.DB_real))
        self.db_loss_fake = dBRatio*self.criterionGAN(self.DB_fake, tf.zeros_like(self.DB_fake))
        self.db_loss = (self.db_loss_real + self.db_loss_fake) / 2 #a completely fooled discrimnator would give back 0.5 per cycle
        self.d_loss = self.d_loss + self.db_loss
        # the BCA section
        if self.options.is_c2gan:
            self.g2_loss = 0
            self.d2_loss = 0
            #self.real_C.set_shape([self.options.batch_size, self.options.image_size*4, self.options.image_size*4, self.sr_c_dim])
            self.fake_C = self.generatorSR(self.fake_B, self.options, reuse=False, name='generatorB2C', numResBlocks=self.numResBlocks)# SR the A2B output to make A2B2C
            self.fake_A__ = self.generator(self.fake_C, self.options, reuse=False, name='generatorC2A', g3Flag=True)# cycle back to A, forming A2B2C2A
            #TODO: add support for C2A2B2C cycle to get the reverse cycle pixelwise loss, and feed discB the halfway outputs
            self.DC_fake = self.discriminator(self.fake_C, self.options, reuse=False, name="discriminatorC", d2Flag = True)
            self.DC_real = self.discriminator(self.real_C, self.options, reuse=True, name="discriminatorC", d2Flag = True)
            # discrimC losses
            self.dc_loss_real = self.criterionGAN(self.DC_real, tf.ones_like(self.DC_real))
            self.dc_loss_fake = self.criterionGAN(self.DC_fake, tf.zeros_like(self.DC_fake))
            self.dc_loss = (self.dc_loss_real + self.dc_loss_fake) / 2.0
            self.d2_loss = self.d2_loss + self.dc_loss
            #special losses for c2gan (image denoising specific)
            self.identity = self.generator(self.real_B, self.options, True, name="generatorA2B") # a clean image should come out clean
            self.identity_loss = self.idt_lambda * abs_criterion(self.identity, self.real_B)
            self.total_variation_loss = self.tv_lambda * tot_var_criterion(self.fake_B)
            self.g_loss = self.g_loss + self.total_variation_loss + self.identity_loss 
            # special losses that are SR specific to c2gan
            self.g_adv_C_loss = self.criterionGAN(self.DC_fake, tf.ones_like(self.DC_fake))
            self.g_loss_C_pixelwise = self.L1_sr_lambda * self.criterionGEN(self.fake_A__, self.real_A)
            self.real_C_idt = apply_bicubic_downsample(apply_bicubic_downsample(self.real_C, filter=self.k, factor=2), filter=self.k, factor=2)
            # gan identity loss
            if self.idt_sr_lambda>0:
                advRatio=1.0
                self.identitySR = self.generatorSR(self.real_C_idt, self.options, True, name="generatorB2C", numResBlocks=self.numResBlocks)
                self.DC_fake_idt = self.discriminator(self.identitySR, self.options, reuse=True, name="discriminatorC", d2Flag = True)
                self.DC_real_idt = self.discriminator(self.real_C, self.options, reuse=True, name="discriminatorC", d2Flag = True)
            else:
                advRatio=0
                self.DC_fake_idt = self.DC_fake
                self.DC_real_idt = self.DC_real
                self.identitySR = self.real_C
            self.identity_loss_SR = self.idt_sr_lambda * self.criterionGEN(self.identitySR, self.real_C)
            self.g_loss_C_variation_loss = self.tv_sr_lambda * tot_var_criterion(self.fake_C)
            
            self.g_adv_Cidt_loss = advRatio*self.criterionGAN(self.DC_fake_idt, tf.ones_like(self.DC_fake_idt))
            self.dc_loss_fake_idt = advRatio*self.criterionGAN(self.DC_fake_idt, tf.zeros_like(self.DC_fake_idt))
            self.dc_loss_real_idt = advRatio*self.criterionGAN(self.DC_real_idt, tf.zeros_like(self.DC_real_idt))
            self.d_loss = self.d_loss + self.dc_loss_fake_idt*0.5 +  self.dc_loss_real_idt*0.5

            # c2gan metrics
            self.gA2PSNR = tf.image.psnr(self.real_A, self.fake_A__, max_val=2.0)
            self.gCidtPSNR = tf.image.psnr(self.real_C, self.identitySR, max_val=2.0)
            #the total g2 loss
            self.g2_loss = self.g_adv_C_loss + self.g_loss_C_pixelwise + self.identity_loss_SR + self.g_loss_C_variation_loss + self.g_adv_Cidt_loss
            #self.testC = self.generatorSR(self.testB, self.options, True, name="generatorB2C", numResBlocks=self.numResBlocks)
            
            # train as an ensemble for now, split this out into separate optimisers later
            self.g_loss = self.g_loss + self.g2_loss
            self.d_loss = self.d_loss + self.d2_loss
        #the BA section            
        elif not self.options.is_c2gan:
            if self.options.is_acgan or self.options.is_srgan:
                if self.options.is_srgan: # overwrite cycles
                    self.fake_A = self.real_B
                    self.fake_B_ = self.fake_A
                else:
                    self.fake_A = self.generator(self.real_B, self.options, reuse=True, name='generatorB_SR2A', g3Flag=True)# reverse B2A2B cycle
                    self.fake_B_ = self.generatorSR(self.fake_A, self.options, reuse=True, name="generatorA2B_SR", numResBlocks=self.numResBlocks)# reverse half of cycle A2B using EDSR            
                #identity loss (enforce original SRGAN performance)

                if self.idt_sr_lambda>0: # this is for ACGAN (sr mode)
                    advRatio=1.0
                    self.real_B_idt = apply_bicubic_downsample(apply_bicubic_downsample(self.real_B, filter=self.k, factor=2), filter=self.k, factor=2)
                    self.identitySR = self.generatorSR(self.real_B_idt, self.options, True, name="generatorA2B_SR", numResBlocks=self.numResBlocks)
                    self.DB_fake_idt = self.discriminator(self.identitySR, self.options, reuse=True, name="discriminatorB_SR", d2Flag = True)
                    self.DB_real_idt = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB_SR", d2Flag = True)
                else:
                    self.real_B_idt = self.real_B
                    self.identitySR = self.real_B
                    self.DB_fake_idt = self.DB_fake
                    self.DB_real_idt = self.DB_real
                    advRatio=0
                if self.options.acType == 'superRes':
                    self.identity_loss_SR = self.idt_sr_lambda * self.criterionGENAB(self.identitySR, self.real_B)
                    self.gBidtPSNR = tf.image.psnr(self.real_B, self.identitySR, max_val=2.0)
                else:
                    self.identity_loss_SR = tf.convert_to_tensor(0.0)#self.idt_sr_lambda * self.criterionGENAB(self.real_A, self.real_B) # this is such stupid bullshit
                    self.gBidtPSNR = tf.convert_to_tensor(0.0)
                # gan identity loss
                self.g_adv_Bidt_loss = advRatio*self.criterionGAN(self.DB_fake_idt, tf.ones_like(self.DB_fake_idt))
                self.db_loss_fake_idt = advRatio*self.criterionGAN(self.DB_fake_idt, tf.zeros_like(self.DB_fake_idt))
                self.db_loss_real_idt = advRatio*self.criterionGAN(self.DB_real_idt, tf.zeros_like(self.DB_real_idt))
                
                self.d_loss = self.d_loss + self.db_loss_fake_idt*0.5 + self.db_loss_real_idt*0.5
                if self.tv_sr_lambda>0:
                    self.g_loss_B_variation_loss = self.tv_sr_lambda * tot_var_criterion(self.fake_B)
                else:
                    self.g_loss_B_variation_loss = tf.convert_to_tensor(0.0)
                self.g_loss = self.g_loss + self.identity_loss_SR + self.g_loss_B_variation_loss + self.g_adv_Bidt_loss
                if self.options.is_srgan: # configure the B loss as either AB loss or BAB loss
                    self.g_loss_B_pixelwise = self.L1_sr_lambda * self.criterionGENAB(self.fake_B, self.real_B)
                    if self.options.glcmRatio>0:
                        print('GLCM Loss function is active')
                        numLevels = 8
                        span = 1#scaleFactor
                        glcmLoss = self.options.glcmRatio*tf.reduce_mean(tf.abs(compute8WayGLCM(self.real_B, numLevels, span) - compute8WayGLCM(self.fake_B, numLevels, span)), name = 'GLCMGeneratorLoss')
                    else:
                        glcmLoss = tf.zeros(1,tf.float32)
                    # calculate gradients to calculate the gradent discrepancy (guides the GAN)
                    self.g_loss = self.g_loss + glcmLoss
                else:
                    self.g_loss_B_pixelwise = self.L1_sr_lambda * self.criterionGENAB(self.real_B, self.fake_B_)        
            elif self.options.is_c1gan:
                self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")# this reverse cycle is inactive during c2gan because we dont care about B2A
                self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")
                self.g_loss_B_pixelwise = self.L1_lambda * self.criterionGEN(self.real_B, self.fake_B_)        
            
            if self.options.is_srgan: # the A discrims are inactive
                self.DA_fake = self.DB_fake
                self.DA_real = self.DB_real
                dRatio=0
            else:
                self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")# discrimination of A is not required for c2gan 1.0
                self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
                dRatio=1
            #g losses for the B2A cycle
            self.g_adv_A_loss = dRatio*self.criterionGAN(self.DA_fake, tf.ones_like(self.DA_fake))           
            self.g_loss = self.g_loss + self.g_loss_B_pixelwise + self.g_adv_A_loss 
            #metrics
            if self.options.is_srgan:
                if self.options.acType == 'superRes' or self.options.acType == 'p2p':
                    self.gBPSNR = tf.image.psnr(self.real_B, self.fake_B, max_val=2.0)
                elif self.options.acType == 'semSeg': # make this weighted 
                    self.gBPSNR = 1-tf.reduce_mean(tf.abs(tf.squeeze(tf.one_hot(self.real_B, self.options.num_classes))-tf.one_hot(tf.argmax(self.fake_B, self.options.nDims+1), self.options.num_classes)))
                self.d_loss = self.d_loss*2
            else:
                self.gBPSNR = tf.image.psnr(self.real_B, self.fake_B_, max_val=2.0)
            #d losses for the A outputs of the B cycle
            self.da_loss_real = dRatio*self.criterionGAN(self.DA_real, tf.ones_like(self.DA_real))
            self.da_loss_fake = dRatio*self.criterionGAN(self.DA_fake, tf.zeros_like(self.DA_fake))
            self.da_loss = (self.da_loss_real + self.da_loss_fake) / 2.0
            self.d_loss = self.d_loss + self.da_loss

        # get the list of network variables 
        self.t_vars = tf.compat.v1.trainable_variables()
        self.d_vars = [var for var in self.t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in self.t_vars if 'generator' in var.name]

        gParams=summarise_model(self.g_vars)
        dParams=summarise_model(self.d_vars)
        total_parameters = dParams+gParams
        print(f'Total Network Parameters: {total_parameters}')

    def train(self, args):
        #with tf.device('/cpu:0'): # do some basic checkpoint computations with the cpu
        """Train cyclegan""" 
        # create the optimisation routines
        self.lr = tf.compat.v1.placeholder(tf.float32, None, name='learning_rate')
#        if self.options.is_c2gan:
#            self.d2_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1).minimize(self.d_loss, var_list=self.d_vars)
#            self.g2_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1).minimize(self.g_loss, var_list=self.g_vars)
        
        counter = 1
        self._build_model()
        self.saver = tf.compat.v1.train.Saver(self.t_vars)
          
        rightNow=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        trainingDir=f"./outputs/{rightNow}-cycleGAN-{self.dataset_dir}"
        os.mkdir(trainingDir)
        # save the input arguments for recordkeeping
        with open(os.path.join(trainingDir, 'args.txt'), 'w') as f:
            for k, v in sorted(args.__dict__.items()):
                f.write(f'{k}={v}\n')
        
        trainOutputDir=f'./trainingOutputs/{rightNow}-cycleGAN-{self.dataset_dir}'
        os.mkdir(trainOutputDir)
        self.opt = tf.compat.v1.train.AdamOptimizer(self.lr, beta1=args.beta1)
        #towerGradients=[]
        #for i in range(args.numGPUs):
        #with tf.device(f'/gpu:{i}'): # run epochs with the gpus, duplicate the graph and 
        # build the network, checkpoint IO, and data IO
        if self.options.is_gan:
            self.d_optim = self.opt.minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = self.opt.minimize(self.g_loss, var_list=self.g_vars)
        # grab compute resources
        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)
        # load checkpoints
        if args.continue_train:
            self.load(args.checkpoint_dir, args.model_dir)
        start_time = time.time()

        writer = tf.summary.FileWriter("output", self.sess.graph)

        writer.close()
        # start training
        for epoch in range(args.epoch):
            # get list of all files in the dataset, shuffle the list, find total batches in dataset
            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
            if not self.options.is_srgan:
                np.random.shuffle(dataA)
                np.random.shuffle(dataB)            
            else:
                dataA=np.sort(dataA)
                dataB=np.sort(dataB)
            if self.options.is_c2gan:
                dataC = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainC'))
                np.random.shuffle(dataC)
                batch_idxs = min(min(len(dataA), len(dataB), len(dataC)), args.train_size) // self.batch_size
            else:
                batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size

            if args.load2ram:
                trainA, trainB = loadDataset2Ram(dataA, dataB, args) # this shouldnt be in the epoch loop, whatever...
                if self.options.is_c2gan:
                    trainC = loadDatasetC2RRam(dataC, args) # havent written this one up yet
                    
                        
            lr = args.lr*(0.5**(epoch/args.epoch_step)) #if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
            tempVec=np.zeros(args.iterNum)
            gLossVec=np.zeros(args.iterNum)
            dLossVec=np.zeros(args.iterNum)
            gAPSNRVec=np.zeros(args.iterNum) 
            gBPSNRVec=np.zeros(args.iterNum)
            gA2PSNRVec=np.zeros(args.iterNum)
            gCidtPSNRVec=np.zeros(args.iterNum)
            gBidtPSNRVec=np.zeros(args.iterNum)
            # for each batch 
            for idx in range(0, args.iterNum):
                batchInd = np.mod(idx,batch_idxs)
                if not args.load2ram: # load subsection from disk
                    if self.options.is_acgan:
                        batch_files = list(zip(dataA[batchInd * self.batch_size:(batchInd + 1) * self.batch_size]))
                        batch_A = [load_c2train_data(batch_file, args.fine_size) for batch_file in batch_files]
                        
                        batch_files = list(zip(dataB[batchInd * self.batch_size:(batchInd + 1) * self.batch_size]))
                        if args.acType == 'superRes':
                            batch_B = [load_c2train_data(batch_file, args.fine_size*4) for batch_file in batch_files]
                        elif args.acType == 'semSeg' or self.options.acType == 'p2p':
                            batch_B = [load_c2train_data(batch_file, args.fine_size) for batch_file in batch_files]

                        batch_A = np.array(batch_A).astype(np.float32)
                        batch_B = np.array(batch_B).astype(np.float32)
                    elif self.options.is_srgan:
                        batch_filesA = list(zip(dataA[batchInd * self.batch_size:(batchInd + 1) * self.batch_size]))
                        batch_filesB = list(zip(dataB[batchInd * self.batch_size:(batchInd + 1) * self.batch_size]))
                        if args.acType == 'superRes':
                            batch_A, batch_B = loadSrganTrainData(batch_filesA, batch_filesB, args)
                        elif args.acType == 'semSeg':
                            batch_A, batch_B = loadSemSegTrainData(batch_filesA, batch_filesB, args)
                        elif args.acType == 'p2p':
                            batch_A, batch_B = loadp2pTrainData(batch_filesA, batch_filesB, args)
                        batch_A = np.array(batch_A).astype(np.float32)
                        batch_B = np.array(batch_B).astype(np.float32)
#                        import matplotlib.pyplot as plt 
#                        plt.figure(1)  
#                        plt.imshow(np.squeeze(batch_A[0,:,:,:])) 
#                        plt.figure(2)
#                        plt.imshow(np.squeeze(batch_B[0,:,:,5]))    
#                        plt.show()
#                        pdb.sdaklghdfsklgjh
                    else: 
                        batch_files = list(zip(dataA[batchInd * self.batch_size:(batchInd + 1) * self.batch_size],
                                               dataB[batchInd * self.batch_size:(batchInd + 1) * self.batch_size]))
                        batch_images = [load_train_data(batch_file, args.load_size, args.fine_size, aA=1/127.5, bA = -1., aB=1/127.5, bB = -1.) for batch_file in batch_files]
                        batch_images = np.array(batch_images).astype(np.float32)
                        
                    if self.options.is_c2gan:
                        batch_files = list(zip(dataC[batchInd * self.batch_size:(batchInd + 1) * self.batch_size]))
                        batch_C = [load_c2train_data(batch_file, args.fine_size*4) for batch_file in batch_files]
                        batch_C = np.array(batch_C).astype(np.float32)
                # get subsection from dataset blocks                        
                elif args.load2ram:
                    batch_A=trainA[batchInd * self.batch_size:(batchInd + 1) * self.batch_size]
                    batch_B=trainB[batchInd * self.batch_size:(batchInd + 1) * self.batch_size]
                    if self.options.is_c2gan:
                        batch_C=trainC[batchInd * self.batch_size:(batchInd + 1) * self.batch_size]
                
                if self.options.is_c2gan:
                    fake_C, fake_B, _, gLoss, gABLoss, gABADVLoss, gABTVLoss, gABIDTLoss, gBCADVLoss, gBCLoss, gBCIDTLoss, gBCTVLoss, aCyclePSNR, a2CyclePSNR, cidtCyclePSNR = self.sess.run([self.fake_C, self.fake_B, self.g_optim, self.g_loss, self.g_loss_A_pixelwise, self.g_adv_B_loss, self.total_variation_loss, self.identity_loss, self.g_adv_C_loss, self.g_loss_C_pixelwise, self.identity_loss_SR, self.g_loss_C_variation_loss, self.gAPSNR, self.gA2PSNR, self.gCidtPSNR], feed_dict={self.real_data: batch_images, self.lr: lr, self.real_C: batch_C})
                    aCyclePSNR=np.mean(aCyclePSNR)
                    a2CyclePSNR=np.mean(a2CyclePSNR)
                    cidtCyclePSNR=np.mean(cidtCyclePSNR)
                    # Update D network
                    if self.options.is_gan:
                        _, dLoss, dABR, dABF, dBAR, dBAF = self.sess.run(
                            [self.d_optim, self.d_loss, self.db_loss_real, self.db_loss_fake, self.dc_loss_real, self.dc_loss_fake],
                            feed_dict={self.real_data: batch_images,
                                       self.real_C: batch_C,
                                       self.fake_C: fake_C,
                                       self.fake_B: fake_B,
                                       self.lr: lr})
                    else:
                        dLoss, dABR, dABF, dBAR, dBAF = 0, 0, 0, 0, 0
                    counter += 1
                    stdout.write("\rLR: %.4e Epoch: [%2d/%2d]] [%4d/%4d] time: %4.4f gLoss: %4.4f [%4.4f | %4.4f | %4.4f | %4.4f | %4.4f | %4.4f | %4.4f | %4.4f] dLoss: %4.4f [%4.4f | %4.4f | %4.4f | %4.4f]" % (lr, epoch+1, args.epoch, idx+1, args.iterNum, time.time() - start_time, gLoss, gABLoss, gABADVLoss, gABTVLoss, gABIDTLoss, gBCLoss, gBCADVLoss, gBCTVLoss, gBCIDTLoss, dLoss, dABR, dABF, dBAR, dBAF))
                    stdout.flush()
                    gLossVec[idx]=gLoss
                    dLossVec[idx]=dLoss
                    gA2PSNRVec[idx]=a2CyclePSNR
                    gCidtPSNRVec[idx]=cidtCyclePSNR
                else:
                    if self.options.is_c1gan:  
                        # Update G network and record halfway fake outputs
                        fake_A, fake_B, _, gLoss, aCyclePSNR, bCyclePSNR = self.sess.run([self.fake_A, self.fake_B, self.g_optim, self.g_loss, self.gAPSNR, self.gBPSNR], feed_dict={self.real_data: batch_images, self.lr: lr})
                        aCyclePSNR=np.mean(aCyclePSNR)
                        bCyclePSNR=np.mean(bCyclePSNR)

                        # Update D network
                        if self.options.is_gan:
                            _, dLoss, dABR, dABF, dBAR, dBAF = self.sess.run(
                                [self.d_optim, self.d_loss, self.db_loss_real, self.db_loss_fake, self.da_loss_real, self.da_loss_fake],
                                feed_dict={self.real_data: batch_images,
                                           self.fake_A: fake_A,
                                           self.fake_B: fake_B,
                                           self.lr: lr})
                        else:
                            dLoss, dABR, dABF, dBAR, dBAF = 0, 0, 0, 0, 0
                        stdout.write(("\rLR: %.4e Epoch: [%2d/%2d]] [%4d/%4d] time: %4.4f gLoss: %4.4f [A-PSNR: %4.4f B-PSNR: %4.4f] dLoss: %4.4f [%4.4f | %4.4f | %4.4f | %4.4f]" % (lr, epoch+1, args.epoch, idx+1, args.iterNum, time.time() - start_time, gLoss, aCyclePSNR, bCyclePSNR, dLoss, dABR, dABF, dBAR, dBAF)))
                    else:
                        #pdbt.set_trace()
                        # Update G network and record halfway fake outputs
                        fake_A, fake_B, _, gLoss, aCyclePSNR, bCyclePSNR, bidtCyclePSNR = self.sess.run([self.fake_A, self.fake_B, self.g_optim, self.g_loss, self.gAPSNR, self.gBPSNR, self.gBidtPSNR], feed_dict={self.real_A: batch_A, self.real_B: batch_B, self.lr: lr})
                        aCyclePSNR=np.mean(aCyclePSNR)
                        bCyclePSNR=np.mean(bCyclePSNR)
                        bidtCyclePSNR=np.mean(bidtCyclePSNR)
                        # Update D network
                        if self.options.is_gan:
                            _, dLoss, dABR, dABF, dBAR, dBAF = self.sess.run(
                                [self.d_optim, self.d_loss, self.db_loss_real, self.db_loss_fake, self.da_loss_real, self.da_loss_fake],
                                feed_dict={self.real_A: batch_A,
                                           self.real_B: batch_B,
                                           self.fake_A: fake_A,
                                           self.fake_B: fake_B,
                                           self.lr: lr})
                        else:
                            dLoss, dABR, dABF, dBAR, dBAF = 0, 0, 0, 0, 0
                        gBidtPSNRVec[idx]=bidtCyclePSNR
                        stdout.write("\rLR: %.4e Epoch: [%2d/%2d]] [%4d/%4d] time: %4.4f gLoss: %4.4f [A-PSNR: %4.4f B-PSNR: %4.4f Bidt-PSNR: %4.4f] dLoss: %4.4f [%4.4f | %4.4f | %4.4f | %4.4f]" % (lr, epoch+1, args.epoch, idx+1, args.iterNum, time.time() - start_time, gLoss, aCyclePSNR, bCyclePSNR, bidtCyclePSNR, dLoss, dABR, dABF, dBAR, dBAF))
                    counter += 1
                    stdout.flush()
                    gLossVec[idx]=gLoss
                    dLossVec[idx]=dLoss
                    gBPSNRVec[idx]=bCyclePSNR
                gAPSNRVec[idx]=aCyclePSNR
            stdout.write("\n")
            
            if self.options.is_c2gan:
                print('Mean Losses G: %4.4f D: %4.4f A-PSNR: %4.4f A2-PSNR: %4.4f Cidt-PSNR: %4.4f' %(np.mean(gLossVec), np.mean(dLossVec), np.mean(gAPSNRVec), np.mean(gA2PSNRVec), np.mean(gCidtPSNRVec)))
            elif self.options.is_acgan or self.options.is_srgan:
                print('Mean Losses G: %4.4f D: %4.4f A-PSNR: %4.4f B-PSNR: %4.4f Bidt-PSNR: %4.4f' %(np.mean(gLossVec), np.mean(dLossVec), np.mean(gAPSNRVec), np.mean(gBPSNRVec, dtype=np.float64), np.mean(gBidtPSNRVec, dtype=np.float64)))
            else:
                print('Mean Losses G: %4.4f D: %4.4f A-PSNR: %4.4f B-PSNR: %4.4f' %(np.mean(gLossVec), np.mean(dLossVec), np.mean(gAPSNRVec), np.mean(gBPSNRVec)))
       
            # run validation
            if np.mod(epoch+1, args.print_freq) == 0 or epoch==0:
                # get the validation dataset, always sort them for ease of reading
                sample_filesA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
                sample_filesB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
                sample_filesA=np.sort(sample_filesA)
                sample_filesB=np.sort(sample_filesB)
                if self.options.is_c2gan:
                    sample_filesC = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testC'))
                    sample_filesC=np.sort(sample_filesC)
                    numValImgs=np.min([len(sample_filesA), len(sample_filesB), len(sample_filesC)])
                    valCidtPSNR=np.zeros(numValImgs)
                    valA2PSNR=np.zeros(numValImgs)
                else:
                    numValImgs=np.min([len(sample_filesA), len(sample_filesB)])
                    valBPSNR=np.zeros(numValImgs)
                if self.options.is_acgan or self.options.is_srgan:
                    valBidtPSNR=np.zeros(numValImgs)
                valAPSNR=np.zeros(numValImgs)
                ind=0
                #make the output folder
                valOutputDir=f'{trainOutputDir}/epoch-{(epoch+1):04}/'
                os.mkdir(valOutputDir)
                if self.options.is_c2gan:
                    for sample_fileA, sample_fileB, sample_fileC in zip(sample_filesA, sample_filesB, sample_filesC):
                        sample_imageA = [load_test_data(sample_fileA)]
                        sample_imageB = [load_test_data(sample_fileB)]
                        sample_imageC = [load_test_data(sample_fileC)]
                        sample_imageA = np.array(sample_imageA).astype(np.float32)
                        sample_imageB = np.array(sample_imageB).astype(np.float32)
                        sample_imageC = np.array(sample_imageC).astype(np.float32)
                        nx=np.min([sample_imageA.shape[1], sample_imageB.shape[1]])
                        ny=np.min([sample_imageA.shape[2], sample_imageB.shape[2]])
                        nx=nx//4*4
                        ny=ny//4*4
                        sample_imageA=sample_imageA[:,0:nx,0:ny,:]
                        sample_imageB=sample_imageB[:,0:nx,0:ny,:]
                        sample_imageC=sample_imageC[:,0:nx*2,0:ny*2,:]
                        fake_C, real_C_down, realCidt, fake_B, aCyclePSNR, a2CyclePSNR, cidtCyclePSNR= self.sess.run([self.fake_C, self.real_C_idt, self.identitySR, self.fake_B, self.gAPSNR, self.gA2PSNR, self.gCidtPSNR], feed_dict={self.real_data: np.concatenate((sample_imageA, sample_imageB), axis=3), self.real_C: sample_imageC})
                        valAPSNR[ind]=aCyclePSNR
                        valA2PSNR[ind]=a2CyclePSNR
                        valCidtPSNR[ind]=cidtCyclePSNR
                        ind=ind+1
                        stdout.write('\rProcessing image: [%4d/%4d], A-PSNR: %4.4f A2-PSNR: %4.4f C-PSNR: %4.4f' %(ind, numValImgs, aCyclePSNR, a2CyclePSNR, cidtCyclePSNR))
                        stdout.flush()            

#                        image_path = os.path.join(valOutputDir,'{0}_{1}{2}'.format('AtoB', os.path.basename(sample_fileA[:-4]), '.png'))
#                        save_images(sample_imageA, [1, 1], image_path)

#                        image_path = os.path.join(valOutputDir,'{0}_{1}{2}'.format('AtoB', os.path.basename(sample_fileB[:-4]), '.png'))
#                        save_images(sample_imageB, [1, 1], image_path)

#                        image_path = os.path.join(valOutputDir,'{0}_{1}{2}'.format('AtoB', os.path.basename(sample_fileC[:-4]), '.png'))
#                        save_images(sample_imageC, [1, 1], image_path)
                        
                        image_path = os.path.join(valOutputDir,'{0}_{1}{2}'.format('AtoB', os.path.basename(sample_fileA[:-4]), '.tif'))
                        #save_images(fake_B, [1, 1], image_path)
                        fake_B=(fake_B+1)*127.5
                        imsave(image_path, np.array(np.squeeze(fake_B.astype('uint8')), dtype='uint8'))
                        
                        
                        image_path = os.path.join(valOutputDir,'{0}_{1}{2}'.format('BtoC', os.path.basename(sample_fileB[:-4]), '.tif'))
                        #save_images(fake_C, [1, 1], image_path)
                        fake_C=(fake_C+1)*127.5
                        imsave(image_path, np.array(np.squeeze(fake_C.astype('uint8')), dtype='uint8'))
#                        image_path = os.path.join(valOutputDir,'{0}_{1}{2}'.format('CRealDown', os.path.basename(sample_fileC[:-4]), '.png'))
#                        save_images(real_C_down, [1, 1], image_path)
                        
                        image_path = os.path.join(valOutputDir,'{0}_{1}{2}'.format('CIDT', os.path.basename(sample_fileC[:-4]), '.tif'))
                        #save_images(realCidt, [1, 1], image_path)
                        realCidt=(realCidt+1)*127.5
                        
                        imsave(image_path, np.array(np.squeeze(realCidt.astype('uint8')), dtype='uint8'))
                        #pdr.set_trace()
                    stdout.write("\n")
                    print('Mean Validation Metrics: A-PSNR: %4.4f A2-PSNR: %4.4f C-PSNR: %4.4f' %(np.mean(valAPSNR), np.mean(valA2PSNR), np.mean(valCidtPSNR)))
                else:
                    for sample_fileA, sample_fileB in zip(sample_filesA, sample_filesB):
                        if self.options.is_acgan or self.options.is_srgan:  
                            if self.options.acType == 'superRes':
                                sample_imageA = [load_test_data(sample_fileA, self.options.is_c1gan, a=1/127.5, b=-1.)]
                                sample_imageB = [load_test_data(sample_fileB, self.options.is_c1gan, a=1/127.5, b=-1.)]
                                sample_imageA = np.array(sample_imageA).astype(np.float32)
                                sample_imageB = np.array(sample_imageB).astype(np.float32)
                                nx=np.min([sample_imageA.shape[1], sample_imageB.shape[1]])
                                ny=np.min([sample_imageA.shape[2], sample_imageB.shape[2]])
                                nx=nx//4*2
                                ny=ny//4*2
                                sample_imageA=sample_imageA[:,0:nx,0:ny,:]
                                sample_imageB=sample_imageB[:,0:nx*4,0:ny*4,:]
                            elif self.options.acType == 'semSeg':
                                sample_imageA = [load_test_data(sample_fileA, self.options.is_c1gan, a=1/255., b=0.0)]
                                sample_imageB = [load_test_data(sample_fileB, self.options.is_c1gan, a=1, b=0.0)]
                                sample_imageA = np.array(sample_imageA).astype(np.float32) 
                                sample_imageB = np.array(sample_imageB).astype(np.float32)
                                if args.nDims == 2:
                                    sample_imageB = np.expand_dims(sample_imageB[:,:,:,0], args.nDims+1) #we assume any channels are repeats
                                elif args.nDims == 3:
                                    sample_imageB = np.expand_dims(sample_imageB[:,:,:,:], args.nDims+1) # we assume single channel unput
                                    sample_imageA = np.expand_dims(sample_imageA, args.nDims+1)
                                nx=args.fine_size
                                ny=args.fine_size
                                sample_imageA=sample_imageA[:,0:nx,0:ny,:]
                                sample_imageB=sample_imageB[:,0:nx,0:ny,:]
                            elif self.options.acType == 'p2p':
                                sample_imageA = [load_test_data(sample_fileA, self.options.is_c1gan, a=1/127.5, b=-1.)]
                                sample_imageB = [load_test_data(sample_fileB, self.options.is_c1gan, a=1/127.5, b=-1.)]
                                sample_imageA = np.array(sample_imageA).astype(np.float32) 
                                sample_imageB = np.array(sample_imageB).astype(np.float32)
                                sample_imageA = np.expand_dims(sample_imageA, args.nDims+1)
                                nx=args.fine_size
                                ny=args.fine_size
                                sample_imageB=np.transpose(sample_imageB,[0,2,3,1])
                                sample_imageA=sample_imageA[:,0:nx,0:ny,:]
                                sample_imageB=sample_imageB[:,0:nx,0:ny,:]

                            fake_A, fake_B, realBidt, realBdown, aCyclePSNR, bCyclePSNR, bidtCyclePSNR = self.sess.run([self.fake_A, self.fake_B, self.identitySR, self.real_B_idt, self.gAPSNR, self.gBPSNR, self.gBidtPSNR], feed_dict={self.real_A: sample_imageA, self.real_B: sample_imageB})
                            #bidtCyclePSNR = psnr(realBidt, sample_imageB, data_range = 2)
                            valBidtPSNR[ind]=bidtCyclePSNR
                        else:
                            fake_A, fake_B, aCyclePSNR, bCyclePSNR = self.sess.run([self.fake_A, self.fake_B, self.gAPSNR, self.gBPSNR], feed_dict={self.real_A: sample_imageA, self.real_B: sample_imageB})
                        
                        valAPSNR[ind]=aCyclePSNR
                        valBPSNR[ind]=bCyclePSNR
                        ind=ind+1
                        if self.options.is_acgan:
                            stdout.write('\rProcessing image: [%4d/%4d], A-PSNR: %4.4f B-PSNR: %4.4f Bidt-PSNR: %4.4f' %(ind, numValImgs, aCyclePSNR, bCyclePSNR, bidtCyclePSNR))
                        else:  
                            stdout.write('\rProcessing image: [%4d/%4d], A-PSNR: %4.4f B-PSNR: %4.4f' %(ind, numValImgs, aCyclePSNR, bCyclePSNR))
                        stdout.flush()            
                        
                        image_path = os.path.join(valOutputDir,'{0}_{1}{2}'.format('AtoB', os.path.basename(sample_fileA[:-4]), '.tif'))
                        #save_images(fake_B, [1, 1], image_path)
                        if self.options.acType == 'superRes' or self.options.acType == 'p2p':
                            fake_B=(fake_B+1)*127.5
                        elif self.options.acType == 'semSeg':
                            fake_B = np.expand_dims(np.squeeze(np.argmax(fake_B, args.nDims+1)), args.nDims)/fake_B.max()*255.
                            if args.nDims == 2:
                                fake_B = np.concatenate((fake_B, fake_B, fake_B), 2)
                        imsave(image_path, np.array(np.squeeze(fake_B[:,:,:,0:3].astype('uint8')), dtype='uint8'))
                        if not self.options.is_srgan:
                            image_path = os.path.join(valOutputDir,'{0}_{1}{2}'.format('BtoA', os.path.basename(sample_fileB[:-4]), '.tif'))
                            #save_images(fake_C, [1, 1], image_path)
                            fake_A=(fake_A+1)*127.5
                            imsave(image_path, np.array(np.squeeze(fake_A.astype('uint8')), dtype='uint8'))
                        
                        if self.options.is_acgan:  
                            image_path = os.path.join(valOutputDir,'{0}_{1}{2}'.format('BIDT', os.path.basename(sample_fileB[:-4]), '.tif'))
                            #save_images(fake_C, [1, 1], image_path)
                            realBidt=(realBidt+1)*127.5
                            imsave(image_path, np.array(np.squeeze(realBidt.astype('uint8')), dtype='uint8'))
                            
                            image_path = os.path.join(valOutputDir,'{0}_{1}{2}'.format('Bbc', os.path.basename(sample_fileB[:-4]), '.tif'))
                            #save_images(fake_C, [1, 1], image_path)
                            realBdown=(realBdown+1)*127.5
                            imsave(image_path, np.array(np.squeeze(realBdown.astype('uint8')), dtype='uint8'))
                        
#                        image_path = os.path.join(valOutputDir,'{0}_{1}'.format('AtoB', os.path.basename(sample_fileA)))
#                        save_images(fake_B, [1, 1], image_path)
#                        
#                        image_path = os.path.join(valOutputDir,'{0}_{1}'.format('BtoA', os.path.basename(sample_fileB)))
#                        save_images(fake_A, [1, 1], image_path)
                    stdout.write("\n")
                    if self.options.is_acgan or self.options.is_srgan:
                        print('Mean Validation Metrics: A-PSNR: %4.4f B-PSNR: %4.4f Bidt-PSNR: %4.4f' %(np.mean(valAPSNR), np.mean(valBPSNR), np.mean(valBidtPSNR)))
                    else:
                        print('Mean Validation Metrics: A-PSNR: %4.4f B-PSNR: %4.4f' %(np.mean(valAPSNR), np.mean(valBPSNR)))
            if np.mod(epoch+1, args.save_freq) == 0 or epoch==0:
                self.save(args.checkpoint_dir, counter, model_dir=args.save_dir)

    def save(self, checkpoint_dir, step, model_dir = None):
        model_name = "cyclegan.model"
        if model_dir is None:
            model_dir = "%s_%s_L1-%s_sr-%s_c1-%s_ac-%s_c2-%s" % (self.dataset_dir, self.image_size, self.L1_lambda, self.options.is_srgan, self.options.is_c1gan, self.options.is_acgan, self.options.is_c2gan)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        print('Saving Checkpoints to ' + checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir, model_dir = None):
        print(" [*] Reading checkpoint...")
        if model_dir is None:
            model_dir = "%s_%s_L1-%s_sr-%s_c1-%s_ac-%s_c2-%s" % (self.dataset_dir, self.image_size, self.L1_lambda, self.options.is_srgan, self.options.is_c1gan, self.options.is_acgan, self.options.is_c2gan)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Loaded checkpoint " + ckpt_name)
            return True
        else:
            return False

    def test(self, args):
        """Test cyclegan"""
        self._build_model()
        self.saver = tf.compat.v1.train.Saver(self.t_vars)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
#        if args.which_direction == 'AtoB':
#            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
#        elif args.which_direction == 'BtoA':
#            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
#        else:
#            raise Exception('--which_direction must be AtoB or BtoA')
        sample_files=glob(args.testInputs+'/*')
        if self.load(args.checkpoint_dir, args.model_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        out_var, in_var = (self.fake_B, self.real_A) if args.which_direction == 'AtoB' else (
            self.fake_A, self.real_B)

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = load_test_image(sample_file)      

            if self.options.acType == 'superRes':
                sample_image = np.array(sample_image).astype(np.float32)/127.5 - 1 # semseg
            elif self.options.acType == 'semSeg':
                sample_image = np.array(sample_image).astype(np.float32)/255. # semseg

            fake_B = self.sess.run(out_var, feed_dict={in_var: sample_image})
            
            if self.options.acType == 'superRes':
                fake_B=(fake_B+1)*127.5
            elif self.options.acType == 'semSeg':
                fake_B = np.expand_dims(np.squeeze(np.argmax(fake_B, args.nDims+1)), args.nDims)#/fake_B.max()*255.
                if args.nDims == 2:
                    fake_B = np.concatenate((fake_B, fake_B, fake_B), 2)
            
            image_path = os.path.join(args.test_dir,'{0}_{1}'.format(args.which_direction, os.path.basename(sample_file.split('.')[0])+'.tif'))
            imsave(image_path, np.array(np.squeeze(fake_B.astype('uint8')), dtype='uint8'))

