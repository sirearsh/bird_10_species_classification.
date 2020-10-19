import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import sys

file_dir=sys.argv[1]
img_shape=(150,150)
data_shape=(150,150,3)

x=tf.cast(tf.image.resize(tf.io.decode_image(tf.io.read_file(file_dir),channels=3),img_shape),tf.dtypes.int64)

labels=['AFRICAN FIREFINCH', 'ALBATROSS', 'ALEXANDRINE PARAKEET', 'AMERICAN AVOCET', 'AMERICAN BITTERN', 'AMERICAN COOT', 'AMERICAN GOLDFINCH', 'AMERICAN KESTREL', 'AMERICAN PIPIT', 'AMERICAN REDSTART']

model=tf.keras.models.load_model('model/model1')

pred=model.predict(tf.reshape(x,(1,150,150,3)))

ans=[]
for i,p in enumerate(pred[0]):
    ans.append((p,labels[i]))

print('bird species',' '*18,'confidence')
for p,l in sorted(ans,reverse=True):
    print(l,' '*(30-len(l)),round(p,2))