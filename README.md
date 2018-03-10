# Description
...

# Usage
```
!pip install -q tqdm keras
!apt-get -qq install -y libsm6 libxext6 && pip install -q -U opencv-python
!pip install --upgrade git+https://github.com/epicfaace/laughing-memory-util.git#egg=laughing_memory_util
import laughing_memory_util
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = 'stage1_train/'
TEST_PATH = 'stage1_test/'
X_train, Y_train, train_ids = laughing_memory_util.data_aug(TRAIN_PATH, TEST_PATH, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)

#Init model.

# Keras models are trained on Numpy arrays of input data and labels. For training a model, you will typically use the  fit function.
earlystopper = keras.callbacks.EarlyStopping(patience=100, verbose=1) 
checkpointer = keras.callbacks.ModelCheckpoint(MODEL_CHECKPOINT_FILE_NAME, verbose=1, save_best_only=True)
results = model.fit((X_train), (Y_train), validation_split=len(train_ids) / len(X_train) * 0.1, batch_size=16, epochs=20, 
                    callbacks=[earlystopper, checkpointer])

# data_aug_download(X_train, Y_train, "special aug")
```