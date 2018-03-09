import skimage.io, skimage.transform, skimage.morphology
import os
import sys
import random
from tqdm import tqdm
import numpy as np
from .elastic_transform import elastic_transform
def data_aug(TRAIN_PATH, TEST_PATH, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS):
    #os.walk example
    i = 0
    for (path, dirs, files) in os.walk(TRAIN_PATH):
        print(path,dirs,files)
        i+=1
        if i == 4:
            break
    # Get train and test IDs (Next() returns the next item from the iterator)
    train_ids = next(os.walk(TRAIN_PATH))[1] #We are interested in list at index 1: List with directories ('str' format) at path 
    test_ids = next(os.walk(TEST_PATH))[1]
    print("Size of the training set: " + str(len(train_ids)))
    print("Size of the test set: " + str(len(test_ids)))
    # Get and resize train images and masks(Initialize containers for storing downsampled images)

    num_aug = 10
    X_train_aug = np.zeros((len(train_ids) * num_aug, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train_aug = np.zeros((len(train_ids) * num_aug, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8) #dtype recommended for pixels
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool) #dtype recommended for boolean matric
    #X_train, Y_train = [],[]

    #Note that the output of the NN in image segmentation (two classes) is a boolean matrix 
    #with the dimensions of the image

    print('Getting and resizing train images and masks ... ')

    sys.stdout.flush() #forces it to "flush" the buffer

    #Tip: Use tqdm() on top of the iterator to have a progress bar
    #enumerate() returns a tuple (index, item) of the iterator. Define 2 looping variables if you use enumerate() 

    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        h, w = IMG_HEIGHT, IMG_WIDTH
        img = skimage.io.imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        #Image resizing to lower resolution
        img = skimage.transform.resize(img, (h, w), mode='constant', preserve_range=True)
        #h, w = img.shape[0], img.shape[1]
        #X_train is a tensor of order 4: A "cube" of data <-> n matrices stacked together
        X_train[n] = img
        #X_train.append(img)
        
        mask = np.zeros((h, w, 1), dtype=np.bool) #Initialize container of the mask
        for mask_file in next(os.walk(path + '/masks/'))[2]:           #Use index 2 for getting name of files (.png)
            mask_ = skimage.io.imread(path + '/masks/' + mask_file)
            #print(mask_.shape) Remove the comment to see how that mask_ is an array of two dimensions! Not three!
            #Insert a new axis
            mask_ = np.expand_dims(skimage.transform.resize(mask_, (h, w), mode='constant', 
                                        preserve_range=True), axis=-1) #preserve_range=True is important!
            #axis = -1 adds an axis at the end of the tuple of dimensions of a np.array
            mask = np.maximum(mask, mask_)
            #mask gets updated at each loop and includes all the masks!
        
        Y_train[n] = mask
        #Y_train.append(mask) #Stores all the masks (true labels in a tensor of order 4: 1 tensor of order 3 per mask)
        
        #v_min, v_max = np.percentile(img, (0.2, 99.8))
        #better_img = exposure.rescale_intensity(img, in_range=(v_min, v_max))
        X_train_aug[num_aug*n + 0] = img[:, ::-1] # horiz flip
        Y_train_aug[num_aug*n + 0] = np.fliplr(mask)
        X_train_aug[num_aug*n + 1] = img[::-1, :] # horiz flip
        Y_train_aug[num_aug*n + 1] = np.flipud(mask)
        X_train_aug[num_aug*n + 2] =  elastic_transform(img)
        Y_train_aug[num_aug*n + 2] =  elastic_transform(mask)
        
        for index in range(3, 9):
            randH = random.randint(0,h)
            randW = random.randint(0,w)
            X_train_aug[num_aug*n + index] = skimage.transform.resize(img[:randH, :randW], (h, w), mode='constant', preserve_range=True)
            Y_train_aug[num_aug*n + index] = skimage.transform.resize(mask[:randH, :randW], (h, w), mode='constant', preserve_range=True)
            
        
        """skimage.io.imshow(X_train[n])
        plt.show()
        skimage.io.imshow(np.squeeze(Y_train[n]))
        plt.show()
        indexToShow = num_aug*n + 3
        skimage.io.imshow(X_train_aug[indexToShow])
        plt.show()
        skimage.io.imshow(np.squeeze(Y_train_aug[indexToShow]))
        plt.show()
        break"""
        
        """skimage.io.imshow(X_train_aug[3*n+3])
        plt.show()
        skimage.io.imshow(np.squeeze(Y_train[3*n]))
        plt.show()
        skimage.io.imshow(np.squeeze(Y_train_aug[3*n+3]))
        plt.show()
        break"""
        

    print('\n Training images succesfully downsampled!')
    return np.concatenate(X_train_aug, X_train), np.concatenate(Y_train_aug, Y_train)