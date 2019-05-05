#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from matplotlib.image import imread, imsave
from matplotlib import pyplot as plt 

import tensorflow as tf
import tensorflow.keras as keras


# In[19]:


class data_generator:
    dir_name=None
    fileNames=None
    count=None
    n_files=None
    def __init__(self,dir_name):
        self.dir_name=dir_name
        self.fileNames=[x for x in os.listdir(self.dir_name) if x.endswith('.jpg')]
        self.n_files=len(self.fileNames)
        self.count=np.random.randint(0,9600)
    def getRandomOne(self):
        res=[]
        img=imread(self.dir_name+self.fileNames[np.random.randint(self.n_files)])
        res.append(img/255.0)
        return np.array(res)
    
    def getEpochData(self,size):
        res=[]
        for i in range(size):
            img=imread(self.dir_name+self.fileNames[self.count])
            self.count=(self.count+1)%self.n_files
            res.append(img/255.)
        res=np.array(res)
        
        m,h,w,c=res.shape
        mask=np.zeros((m,h,w,1))
        #x=np.random.randint(127-64,127+64)
        #y=np.random.randint(127-64,127+64)
        #k=np.random.randint(64,128)
        x=127
        y=127
        k=128
        mask[:,x-k//2:x+k//2,y-k//2:y+k//2,:]=1.0
        
        generator_input=res*(1.0-mask)
        print("Read Image Bank:",res.shape)
        return res,np.concatenate((generator_input,mask),axis=-1),np.concatenate((mask,mask,mask),axis=-1)

class tsop:
    def conv2d(x,n_filters,fsize,strides=1,padding='REFLECT',activation=tf.nn.relu,
               dilation_rate=1,name=None):
        p=dilation_rate*(fsize//2)
        if padding=='REFLECT':
            x=tf.pad(x,[[0,0],[p,p],[p,p],[0,0]],mode=padding,name=name+'_padding')
            padding='VALID'
        x=keras.layers.Conv2D(n_filters,kernel_size=fsize,strides=strides,activation=activation,
                              dilation_rate=dilation_rate,name=name+'_conv',padding=padding)(x)
        return x

def generator(x):
    g_vars=[t for t in tf.global_variables() if t.name.startswith('generator')]
    with tf.variable_scope('generator',reuse=len(g_vars)>0):
        x=tsop.conv2d(x,64,fsize=5,name='step0_0')

        x=tsop.conv2d(x,128,fsize=3,strides=2,name='step1_0')
        x=tsop.conv2d(x,128,fsize=3,name='step1_1')

        x=tsop.conv2d(x,256,fsize=3,strides=2,name='step2_0')
        x=tsop.conv2d(x,256,fsize=3,name='step2_1')
        x=tsop.conv2d(x,256,fsize=3,name='step2_2')
        
        x=tsop.conv2d(x,256,fsize=3,dilation_rate=2,name='step3_0')
        x=tsop.conv2d(x,256,fsize=3,dilation_rate=4,name='step3_1')
        x=tsop.conv2d(x,256,fsize=3,dilation_rate=8,name='step3_2')
        x=tsop.conv2d(x,256,fsize=3,dilation_rate=16,name='step3_3')
        
        x=tsop.conv2d(x,256,fsize=3,name='step4_0')
        x=tsop.conv2d(x,256,fsize=3,name='step4_1')
        
        x=keras.layers.Conv2DTranspose(128,kernel_size=4,strides=2,padding='same',name='step5_0_deconv1')(x)
        x=tsop.conv2d(x,128,fsize=3,name='step5_1')
        
        x=keras.layers.Conv2DTranspose(64,kernel_size=4,strides=2,padding='same',name='step6_0_deconv2')(x)
        x=tsop.conv2d(x,32,fsize=3,name='step6_1')
        x=tsop.conv2d(x,3,fsize=3,name='step6_2',activation=tf.nn.sigmoid)
    return x


# In[20]:


data=data_generator("training_data/")
batch_size=1
np_real_images,np_generator_input,np_mask=data.getEpochData(batch_size)

tf.reset_default_graph()
real_images=tf.placeholder(name="real_images",dtype=tf.float32,shape=[batch_size,256,256,3])
G_input=tf.placeholder(name="G_input",dtype=tf.float32,shape=[batch_size,256,256,4])
mask=tf.placeholder(name="real_images",dtype=tf.float32,shape=[batch_size,256,256,3])

G_output=generator(G_input)
Final_output=G_output*mask+real_images*(1-mask)

t_vars=tf.trainable_variables()
g_vars=[var for var in t_vars if 'generator' in var.name]
saver_generator = tf.train.Saver(g_vars)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver_generator.restore(sess, "checkpoints/generator.ckpt")
    if not os.path.exists('test_out/'):
        os.makedirs('test_out/')
    img_out=sess.run(Final_output,feed_dict={real_images:np_real_images,
                                                            G_input:np_generator_input,
                                                            mask:np_mask})
    imsave('test_out/generated_image.png',img_out[0])
    imsave('test_out/original_image.png',np_real_images[0])
    imsave('test_out/input_image.png',np_generator_input[0][:,:,0:3])


# In[ ]:




