import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

img_shape=(150,150)
data_shape=(150,150,3)

def getdataset(path_dir=''):
  if path_dir=='':
    return -1,None,None
  train_ds=tf.keras.preprocessing.image_dataset_from_directory(
    path_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=img_shape,
    batch_size=32
  )
  val_ds=tf.keras.preprocessing.image_dataset_from_directory(
    path_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=img_shape,
    batch_size=32
  )
  class_names=train_ds.class_names
  print('classes:',class_names)
  return class_names,train_ds,val_ds