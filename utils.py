"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import pprint
from PIL import Image
import imageio
import numpy as np
import copy
import pdb
from sys import stdout
import h5py
def load_test_image(img_path):
    ext=img_path.split('.')[1]
    if ext == 'png':
        img = Image.open(img_path)
        if img.mode != 'RGB': #makes it triple channel
            img = img.convert('RGB')
        img = np.array(img, dtype='uint8')
        img = img[0:512,0:512]
    elif ext == 'mat':
        arrays = {}
        f = h5py.File(img_path)
        for k, v in f.items():
            arrays[k] = np.array(v)
        img=arrays['temp'] 
        img=np.expand_dims(img,3)
        img = np.array(img, dtype='uint8')
    
    #img = np.array(Image.fromarray(img).resize([512, 512]))
    return np.expand_dims(img,0) # for consistency

def load_test_data(image_path, c1ganFlag, a=1/127.5, b=-1.): # this is for insitu training validation (loads a numpy)
    img = np.load(image_path)
    if c1ganFlag:
        img = np.array(Image.fromarray(img).resize([256, 256]))

    img = img*a+b
    return img

#load unpaired images of same size
def load_train_data(image_path, load_size=286, fine_size=256, aA=1/127.5, bA = -1., aB=1/127.5, bB = -1.):
    img_A = np.load(image_path[0], allow_pickle=True)
    img_B = np.load(image_path[1], allow_pickle=True)

    img_A = np.array(Image.fromarray(img_A).resize([load_size, load_size]))
    img_B = np.array(Image.fromarray(img_B).resize([load_size, load_size]))
    h1 = int(np.ceil(np.random.uniform(1e-2, img_A.shape[0]-fine_size)))
    w1 = int(np.ceil(np.random.uniform(1e-2, img_A.shape[1]-fine_size)))
    h2 = int(np.ceil(np.random.uniform(1e-2, img_B.shape[0]-fine_size)))
    w2 = int(np.ceil(np.random.uniform(1e-2, img_B.shape[1]-fine_size)))
    if load_size==fine_size:
        h1=0
        w1=0
        h2=0
        w2=0
    img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
    img_B = img_B[h2:h2+fine_size, w2:w2+fine_size]

    img_A = img_A*aA+bA
    img_B = img_B*aB+bB

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB
    
# load simgle image    
def load_c2train_data(image_path, fine_size=256, is_testing=False):
    img_C = np.load(image_path[0])
    nx=img_C.shape[0]
    ny=img_C.shape[1]
    if not is_testing:
        h1 = int(np.ceil(np.random.uniform(1e-2, nx-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, ny-fine_size)))
        if nx==fine_size:
            h1=0
        if ny==fine_size:
            w1=0
        img_C = img_C[h1:h1+fine_size, w1:w1+fine_size]
    img_C = img_C/127.5 - 1.
    return img_C

# load paired images
def loadSrganTrainData(batch_filesA, batch_filesB, args):
    imgA=np.zeros((len(batch_filesA), args.fine_size,  args.fine_size, args.input_nc), dtype='uint8')
    imgB=np.zeros((len(batch_filesB), args.fine_size*4,  args.fine_size*4, args.output_nc), dtype='uint8')
    n=0
    for imgADir, imgBDir in zip(batch_filesA, batch_filesB):
        img_A = np.load(imgADir[0])
        img_B = np.load(imgBDir[0])

        lr_w = np.random.randint(img_A.shape[0] - args.fine_size+1)
        lr_h = np.random.randint(img_A.shape[1] - args.fine_size+1)
        hr_w = lr_w * 4
        hr_h = lr_h * 4
        img_A = img_A[lr_w:lr_w + args.fine_size, lr_h:lr_h + args.fine_size]
        img_B = img_B[hr_w:hr_w + args.fine_size*4, hr_h:hr_h + args.fine_size*4]
        imgA[n]=img_A
        imgB[n]=img_B
        n=n+1
    imgA = imgA/127.5 - 1.
    imgB = imgB/127.5 - 1.
    return imgA, imgB

def loadSemSegTrainData(batch_filesA, batch_filesB, args):
    if args.nDims == 2:
        imgA=np.zeros((len(batch_filesA), args.fine_size,  args.fine_size, args.input_nc), dtype='uint8')
        imgB=np.zeros((len(batch_filesB), args.fine_size,  args.fine_size, args.output_nc), dtype='uint8')
    elif args.nDims ==3:
        imgA=np.zeros((len(batch_filesA), args.fine_size,  args.fine_size, args.fine_size, args.input_nc), dtype='uint8')
        imgB=np.zeros((len(batch_filesB), args.fine_size,  args.fine_size, args.fine_size, args.output_nc), dtype='uint8')
    n=0
    for imgADir, imgBDir in zip(batch_filesA, batch_filesB):
        img_A = np.load(imgADir[0])
        img_B = np.load(imgBDir[0])


        lr_w = np.random.randint(img_A.shape[0] - args.fine_size+1)
        lr_h = np.random.randint(img_A.shape[1] - args.fine_size+1)
        hr_w = lr_w
        hr_h = lr_h
        img_A = img_A[lr_w:lr_w + args.fine_size, lr_h:lr_h + args.fine_size]
        img_B = img_B[hr_w:hr_w + args.fine_size, hr_h:hr_h + args.fine_size]
        imgA[n]=img_A
        imgB[n]=np.expand_dims(img_B[:,:,0], 2)
        n=n+1

    imgA = imgA/255.
    imgB = imgB
    return imgA, imgB
    

def loadp2pTrainData(batch_filesA, batch_filesB, args):
    if args.nDims == 2:
        imgA=np.zeros((len(batch_filesA), args.fine_size,  args.fine_size, args.input_nc), dtype='float32')
        imgB=np.zeros((len(batch_filesB), args.fine_size,  args.fine_size, args.output_nc), dtype='float32')
    elif args.nDims ==3:
        imgA=np.zeros((len(batch_filesA), args.fine_size,  args.fine_size, args.fine_size, args.input_nc), dtype='float32')
        imgB=np.zeros((len(batch_filesB), args.fine_size,  args.fine_size, args.fine_size, args.output_nc), dtype='float32')
    n=0
    for imgADir, imgBDir in zip(batch_filesA, batch_filesB):
        img_A = np.load(imgADir[0])
        img_B = np.load(imgBDir[0])
        img_A = np.expand_dims(img_A,args.nDims)
        img_B = np.transpose(img_B,[1,2,0])

        lr_w = np.random.randint(img_A.shape[0] - args.fine_size+1)
        lr_h = np.random.randint(img_A.shape[1] - args.fine_size+1)
        hr_w = lr_w
        hr_h = lr_h
        img_A = img_A[lr_w:lr_w + args.fine_size, lr_h:lr_h + args.fine_size]
        img_B = img_B[hr_w:hr_w + args.fine_size, hr_h:hr_h + args.fine_size]
        imgA[n]=img_A
        imgB[n]=img_B
        n=n+1

    imgA = imgA/127.5 - 1.
    imgB = imgB/127.5 - 1.
    return imgA, imgB
    
    
def loadDataset2Ram(dataA, dataB, args):
    # dataA and data B should be paired and of the same size/scale
    if args.acType == 'superRes':
        scale=4
    elif args.acType == 'semSeg':
        scale=1
    imgA=[]#np.zeros((len(batch_filesA), args.fine_size,  args.fine_size, args.input_nc), dtype='uint8')
    imgB=[]#np.zeros((len(batch_filesB), args.fine_size*4,  args.fine_size*4, args.output_nc), dtype='uint8')
    n=0
    for imgADir, imgBDir in zip(dataA, dataB):
        stdout.write(f'\rLoading image {imgBDir}')
        stdout.flush()
        img_A = np.load(imgADir).astype('float32')
        img_B = np.load(imgBDir).astype('float32')
        # for each image, slice it up
        nw=img_A.shape[0]//args.fine_size
        nh=img_A.shape[1]//args.fine_size

        lr_w = args.fine_size*nw
        lr_h = args.fine_size*nh
        if args.nDims == 3:
            nd=np.max((1, img_A.shape[2]//args.fine_size)) # if 2D-RGB, this will default to 1
            lr_d = args.fine_size*nd
            img_A = np.expand_dims(img_A[0:lr_w,0:lr_h,0:lr_d],0) # crop and add batch dim
            img_B = np.expand_dims(img_B[0:lr_w*scale,0:lr_h*scale,0:lr_d*scale],0)
            img_A = np.vstack(np.split(np.vstack(np.split(np.vstack(np.split(img_A,nw,1)),nh,2)),nd,3)) # split in 3D, and stack in batch dim
            img_B = np.vstack(np.split(np.vstack(np.split(np.vstack(np.split(img_B,nw,1)),nh,2)),nd,3))
        if args.nDims == 2:
            img_A = np.expand_dims(img_A[0:lr_w,0:lr_h],0) # crop and add batch dim
            img_B = np.expand_dims(img_B[0:lr_w*scale,0:lr_h*scale],0)
            img_A = np.vstack(np.split(np.vstack(np.split(img_A,nw,1)),nh,2)) # split in 2D, and stack in batch dim
            img_B = np.vstack(np.split(np.vstack(np.split(img_B,nw,1)),nh,2))

        imgA.append(img_A)
        imgB.append(img_B)
        n=n+1
    stdout.write("\n")
    
    return np.expand_dims(np.vstack(imgA)/255., 4), np.expand_dims(np.vstack(imgB), 4)
    

def summarise_model(layerVars):
    gParams=0
    for variable in layerVars:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        #print(len(shape))
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
        print(variable.name+f' numParams: {variable_parameters}')
        print(shape)
        gParams += variable_parameters
    print(f'Network Parameters: {gParams}')
    return gParams
    
