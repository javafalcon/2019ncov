# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 09:11:14 2020
Modified on Sun May 03 14:13 2020
@author: lin, weizhong
"""
import numpy as np
from tensorflow.keras import layers, models, optimizers
from Capsule import CapsuleLayer, squash, Length, Mask, margin_loss
        
def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    output = layers.Conv2D(filters=dim_vector*n_channels, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          name='primaryCap_conv2d')(inputs)
    dim = output.shape[1]*output.shape[2]*output.shape[3]
    outputs = layers.Reshape(target_shape=(dim//dim_vector,dim_vector), name='primaryCap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

from tensorflow.keras.utils import to_categorical
def CapsNet(input_shape, n_class, num_routing):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid',
                         activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=9, 
                            strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=16, num_routing=num_routing,
                            name='digitcaps')(primarycaps)
    out_caps = Length(name='out_caps')(digitcaps)
    
    # Decoder network
    y = layers.Input(shape=(n_class,))
    masked = Mask()([digitcaps, y])
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(np.prod(input_shape), activation='sigmoid')(x_recon)
    x_recon = layers.Reshape(target_shape=input_shape, name='out_recon')(x_recon)
    
    return models.Model([x,y], [out_caps, x_recon])



def train(model, data, lr, lam_recon, batch_size, epochs):
    (x_train, y_train),(x_test, y_test) = data
    
    model.compile(optimizer=optimizers.Adam(lr=lr),
                 loss=[margin_loss, 'mse'],
                 loss_weights=[1., lam_recon],
                 metrics={'out_caps': 'accuracy'})
    model.fit([x_train, y_train], [y_train, x_train],
              batch_size=batch_size,
              epochs=epochs,
              validation_data=[[x_test,y_test],[y_test,x_test]])
    
def load_mnist():
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1,28,28,1).astype('float32') / 255.
    x_test = x_test.reshape(-1,28,28,1).astype('float32') / 255.
    y_train = to_categorical(y_train.astype('float32'))
    y_test = to_categorical(y_test.astype('float32'))
    
    return (x_train, y_train),(x_test, y_test)

(x_train, y_train),(x_test, y_test) = load_mnist()
model = CapsNet(input_shape=[28,28,1], n_class=10, num_routing=3)
model.summary()
train(model=model,data=((x_train,y_train),(x_test,y_test)), 
      lr=0.001, lam_recon=0.39,
      batch_size=100, epochs=10)