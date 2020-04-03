# a tf and np version of matlab's GLCM
import numpy as np

#import matplotlib.pyplot as plt
import tensorflow as tf

#from skimage.feature import greycomatrix, greycoprops
#from skimage import data
# will transform a nb, x, y, c tensor into a glcm stack 
def compute8WayGLCM(x, numLevels, span):
    offset = np.linspace(1,span,span)
    offset = np.expand_dims(offset,1)
    offsets0 = np.vstack([np.concatenate([offset*-1, offset*-1],1), \
                          np.concatenate([np.zeros([span,1]), offset],1), \
                          np.concatenate([np.zeros([span,1]), offset*-1],1), \
                          np.concatenate([offset*-1, np.zeros([span,1])],1), \
                          np.concatenate([offset*-1, offset],1), \
                          np.concatenate([offset, offset],1), \
                          np.concatenate([offset, np.zeros([span,1])],1), \
                          np.concatenate([offset, offset*-1],1)]).astype('int32')
    GLCMs = computeGLCM(x, offsets0[0,:], numLevels)
    for i in range(offsets0.shape[0]-1):
        n=i+1
        GLCM = computeGLCM(x, offsets0[n,:], numLevels)
        GLCMs = tf.concat([GLCMs, GLCM],2) # change concat dim to 2 for true glcms
    return GLCMs

def computeGLCM(image, offset, numLevels):
    maxGrey = 2**numLevels-1
    # using operators for the quantisation lets both np and tf work
    imageLeveled = (image+1.0)
    imageLeveled = imageLeveled/2.0*maxGrey+1 
    nx = tf.shape(imageLeveled)[1] # no choice but to use function instead of object
    ny = tf.shape(imageLeveled)[2]
#    nx = np.shape(imageLeveled)[1] # no choice but to use function instead of object
#    ny = np.shape(imageLeveled)[2]
    # some pretty clever cropping imo
    start1 = -(np.abs(offset[0])-offset[0])//2
    start2 = -(np.abs(offset[1])-offset[1])//2 # this can be replaced with an offset kernel

    end1 = (np.abs(offset[0])+offset[0])//2
    end2 = (np.abs(offset[1])+offset[1])//2
    
    img1 = imageLeveled[:,-start1:nx-end1, -start2:ny-end2,:]
    img2 = imageLeveled[:,end1:nx+start1, end2:ny+start2,:]
    
    #convInds= img2 + (img1)*2**numLevels # convert the offsets into unique indexes
    #
    
    
    img1 = tf.reshape(img1,[-1]) - 1 # generalise for nD tensors
    img2 = tf.reshape(img2,[-1]) - 1
    # sub2ind
    inds = img2 + (img1)*2**numLevels # this is superior to the 6point gradient loss because it retains real values
    # if the inds are unsorted, its just pixelwise again. sorting is needed to enforce collocation
    inds = tf.contrib.framework.sort(inds) # this is very expensive to calculate
    #glcm = inds#tf.cast(inds, tf.float32) 
    
    indsint = tf.cast(tf.round(inds), tf.int32)
#     #the index list acts as a differentiable form of the GLCM information
#     #this segmentsum int casting is gradient incompatible, so do something clever
    glcm = tf.math.segment_sum(inds, indsint)

    pad = tf.zeros(2**numLevels*2**numLevels-tf.shape(glcm)[0],tf.float32)
    glcm = tf.concat([glcm,pad],0)
    glcm = glcm/np.linspace(1,2**numLevels*2**numLevels,2**numLevels*2**numLevels)
    glcm = tf.reshape(glcm, [2**numLevels, 2**numLevels])
    glcm = tf.expand_dims(glcm,2)

    glcm = glcm/tf.reduce_prod(tf.cast(tf.shape(image), tf.float32)) # tf will not minimise this properly without mse support - more testing needed
    return glcm #, img1, img2, inds

def testFunction():
    # image is a none tf tensor
    # open the camera image
    img = (data.camera().astype('float64')-127.5)/127.5
    img = np.expand_dims(img,0)
    img = np.expand_dims(img,3)
    inputShape=[None, None, None, 1]
    image = tf.placeholder('float32',inputShape, name='tempImagePlaceholder')
    image2 = tf.placeholder('float32',inputShape, name='tempImagePlaceholder')
    numLevels = 4

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 1
    session = tf.Session(config=config)

    GLCM = compute8WayGLCM(image, numLevels, 1)
    bt,row,col,ch= img.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = img*gauss/2

    GLCM2 = compute8WayGLCM(image2, numLevels, 5)
    glcmLoss = tf.reduce_mean(tf.abs(GLCM-GLCM2), name = 'GLCMGeneratorLoss')

    session.run(tf.initialize_all_variables())

    glcm, glcm2, glcm_loss = session.run([GLCM, GLCM2, glcmLoss], {image: img, image2: noisy})
    print(f'GLCMloss: {glcm_loss}')
    glcm=np.array(glcm)
    for i in range(40):
        plt.subplot(4,10,i+1)
        plt.imshow(np.squeeze(glcm[:,:,i]))
        plt.colorbar()
    plt.show()
    return
