import tensorflow as tf
from tensorflow import keras


print(tf.__version__)
print(keras.__version__)

print("사용 가능한 GPU 수: ", len(tf.config.experimental.list_physical_devices('GPU')))

"""
GPU : 
Accuracy:  1.0
훈련 끝 : 13시 8분 45초
소요시간 : 0시 1분 37초

CPU :
Accuracy:  1.0
훈련 끝 : 13시 14분 23초
소요시간 : 0시 4분 23초
"""