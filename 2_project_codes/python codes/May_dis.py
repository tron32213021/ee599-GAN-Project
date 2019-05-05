#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from matplotlib.image import imread, imsave
from matplotlib import pyplot as plt 

import tensorflow as tf
import tensorflow.keras as keras


# In[2]:


class data_generator:
    dir_name=None
    fileNames=None
    count=None
    n_files=None
    def __init__(self,dir_name):
        self.dir_name=dir_name
        self.fileNames=[x for x in os.listdir(self.dir_name) if x.endswith('.jpg')]
        self.count=0
        self.n_files=len(self.fileNames)

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
        x=127
        y=127
        mask[:,x-64:x+64,y-64:y+64,:]=1.0
        
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


# In[3]:


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


# In[4]:


def local_discriminator(x):
    ld_vars=[t for t in tf.global_variables() if t.name.startswith('local_discriminator')]
    with tf.variable_scope('local_discriminator',reuse=len(ld_vars)>0):
        x=tsop.conv2d(x,64,fsize=5,strides=2,name='step1')
        x=tsop.conv2d(x,128,fsize=5,strides=2,name='step2')
        x=tsop.conv2d(x,256,fsize=5,strides=2,name='step3')
        x=tsop.conv2d(x,512,fsize=5,strides=2,name='step4')
        x=tsop.conv2d(x,512,fsize=5,strides=2,name='step5')
        x=keras.layers.Flatten()(x)
        x=keras.layers.Dense(1)(x)
    return x


# In[9]:


#*************** parameters *****************#
N_each_epoch=960
total_epoches=1000
batch_size=32
data=data_generator("training_data/")

tf.reset_default_graph()
#************ define input *****************#
real_images=tf.placeholder(name="real_images",dtype=tf.float32,shape=[batch_size,256,256,3])
G_input=tf.placeholder(name="G_input",dtype=tf.float32,shape=[batch_size,256,256,4])
mask=tf.placeholder(name="real_images",dtype=tf.float32,shape=[batch_size,256,256,3])

#************ define output ***************#
G_output=generator(G_input)
Final_output=G_output*mask+real_images*(1-mask)
win_generated=tf.reshape(tf.boolean_mask(G_output,mask),shape=(batch_size,128,128,3))
win_reals=tf.reshape(tf.boolean_mask(real_images,mask),shape=(batch_size,128,128,3))

LD_generated_output=local_discriminator(win_generated)
LD_reals_output=local_discriminator(win_reals)

#************ define gradient panelty ***************#
p=tf.random.uniform([batch_size,1,1,1],minval=0.,maxval=1.)
win_mixed=p*win_generated+(1-p)*win_reals
LD_mixed_output=local_discriminator(win_mixed)
grad=tf.gradients(LD_mixed_output,win_mixed)[0]
grad_norm=tf.sqrt(tf.reduce_sum(grad**2,axis=[1,2,3]))
grad_pen=10*tf.reduce_mean(tf.nn.relu(grad_norm-1.))

#************ define loss **************#
MSE_loss=tf.reduce_mean(tf.pow(G_output-real_images,2))

LD_loss=tf.reduce_mean(LD_generated_output)-tf.reduce_mean(LD_reals_output)+grad_pen+10*MSE_loss
G_loss=-tf.reduce_mean(LD_generated_output)+10*MSE_loss

#************ define optimizers ********#
G_MSE_optimizer=tf.train.AdamOptimizer(learning_rate=1e-4).minimize(MSE_loss)
G_optimizer=tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(G_loss)
LD_optimizer=tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(LD_loss)

#************ define data, vars *********#
np_real_images,np_generator_input,np_mask=data.getEpochData(N_each_epoch)
t_vars=tf.trainable_variables()
g_vars=[var for var in t_vars if 'generator' in var.name]
ld_vars=[var for var in t_vars if 'local_discriminator' in var.name]

#training
saver_generator = tf.train.Saver(g_vars)
saver_local_dis=tf.train.Saver(ld_vars)
if not os.path.exists('checkpoints/'):
    os.makedirs('checkpoints/')


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver_generator.restore(sess, "checkpoints/model.ckpt")
    saver_local_dis.restore(sess, "checkpoints/local_dis.ckpt")
    if not os.path.exists('final_out/'):
        os.makedirs('final_out/')
    for epoch in range(total_epoches):
        total_batch=int(N_each_epoch/batch_size)
        print("#########Training the %s epoch, the total_batch size is %s###########"%(epoch,total_batch))
        
        for e in range(total_batch):
            
            batch_real_images=np_real_images[e*batch_size:(e+1)*batch_size]
            batch_generate_input=np_generator_input[e*batch_size:(e+1)*batch_size]
            _,ld_loss=sess.run([LD_optimizer,LD_loss],feed_dict={real_images:batch_real_images,
                                                                G_input:batch_generate_input,
                                                                mask:np_mask[:batch_size]})
            for i in range(1):
                _,g_loss=sess.run([G_optimizer,G_loss],feed_dict={real_images:batch_real_images,
                                                                            G_input:batch_generate_input,
                                                                            mask:np_mask[:batch_size]})
            print("LD_loss is %s, G_loss is %s"%(ld_loss,g_loss))
            
            if e % 1 ==0:
                img_out_t=sess.run(Final_output,feed_dict={real_images:batch_real_images,
                                                            G_input:batch_generate_input,
                                                            mask:np_mask[:batch_size]})
                img_=img_out_t[0]
                imsave('final_out/generated_%s.png' %(e),img_)
                imsave('final_out/original_%s.png' %(e),batch_real_images[0])
                saver_generator.save(sess, "checkpoints/generator.ckpt")
                saver_local_dis.save(sess, "checkpoints/local_dis.ckpt")
        


# In[ ]:




