import os

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


print(tf.__version__)
print(tf.config.list_physical_devices())
