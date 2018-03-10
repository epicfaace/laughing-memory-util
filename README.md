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

# data_aug_download(X_train, Y_train, "special aug")
```