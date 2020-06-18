# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 20:03:57 2020

@author: lwzjc
"""


import pandas
import numpy as np
import os

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.metrics import f1_score,roc_auc_score,recall_score,precision_score

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint

from resnet import resnet_v1
from aaindexValues import aaindex1PCAValues
from Capsule import CapsuleLayer, squash, Length, Mask, margin_loss

trainFile = 'Datasets/TRB/Tran.TCRB.csv'
testFile = 'Datasets/TRB/Test.TCRB.csv'
indepFile = 'Datasets/TRB/SARS-CoV-2_TCRB.csv'

def writeMetrics(metricsFile, y_true, predicted_Probability, noteInfo=''):
    predicts = np.argmax(predicted_Probability, axis=-1)
    labels = np.argmax(y_true, axis=-1)
    
    cm=confusion_matrix(labels,predicts)
    with open(metricsFile,'a') as fw:
        if noteInfo:
            fw.write('\n\n' + noteInfo + '\n')
        for i in range(2):
            fw.write(str(cm[i,0]) + "\t" +  str(cm[i,1]) + "\n" )
        fw.write("ACC: %f "%accuracy_score(labels,predicts))
        #fw.write("\nF1: %f "%f1_score(labels,predicts))
        #fw.write("\nRecall: %f "%recall_score(labels,predicts))
        #fw.write("\nPre: %f "%precision_score(labels,predicts))
        #fw.write("\nMCC: %f "%matthews_corrcoef(labels,predicts))
        #fw.write("\nAUC: %f\n "%roc_auc_score(labels,predicted_Probability[:,0]))
        
def pca_code(seqs:list, row=30, n_features=16):
    aadict = aaindex1PCAValues(n_features)
    x = []
    col = n_features+1
    for i in range(len(seqs)):
        seq = seqs[i]
        n = len(seq)
        t = np.zeros(shape=(row, col))
        j = 0
        while j < n and j < row:
            t[j,:-1] = aadict[seq[j]]
            t[j,-1] = 0
            j += 1
        while j < row:
            t[j,-1] = 1
            j = j + 1
        x.append(t)
    return np.array(x)

def read_seqs(file):
    data = pandas.read_csv(file)
    labels = data.Class_label
    cdr3 = data.CDR3
    epitope = data.Epitope
    cdr3_seqs, epit_seqs = [], []
    for i in range(len(epitope)):
        cdr3_seqs.append(cdr3[i][2:-1])
        epit_seqs.append(epitope[i])
    
    return cdr3_seqs, epit_seqs, labels

def load_data(col=20, row=9):
    
    train_cdr3_seqs,  train_epit_seqs, train_labels = read_seqs(trainFile)
    x_train = np.ndarray(shape=(len(train_cdr3_seqs), row, col+1, 2))
    x_train[:,:,:,0] = pca_code(train_cdr3_seqs, row, col)
    x_train[:,:,:,1] = pca_code(train_epit_seqs, row, col)
     
    y_train = np.array(train_labels)
    y_train = to_categorical(y_train, 2)
    
    test_cdr3_seqs,  test_epit_seqs, test_labels = read_seqs(testFile)
    x_test = np.ndarray(shape=(len(test_cdr3_seqs), row, col+1, 2))
    x_test[:,:,:,0] = pca_code(test_cdr3_seqs, row, col)
    x_test[:,:,:,1] = pca_code(test_epit_seqs, row, col)
    
    y_test = np.array(test_labels)
    y_test = to_categorical(y_test, 2)
    
    indt_cdr3_seqs,  indt_epit_seqs, indt_labels = read_seqs(indepFile)
    x_indt = np.ndarray(shape=(len(indt_cdr3_seqs), row, col+1, 2))
    x_indt[:,:,:,0] = pca_code(indt_cdr3_seqs, row, col)
    x_indt[:,:,:,1] = pca_code(indt_epit_seqs, row, col)
    
    y_indt = np.array(indt_labels)
    y_indt = to_categorical(y_indt, 2)
    return (x_train, y_train), (x_test, y_test),(x_indt, y_indt)

def PrimaryCap(inputs, dim_vector, n_channels, kernel_size, strides, padding):
    output = layers.Conv2D(filters=dim_vector*n_channels, kernel_size=kernel_size,
                          strides=strides, padding=padding,
                          name='primaryCap_conv2d')(inputs)
    dim = output.shape[1]*output.shape[2]*output.shape[3]
    '''output = layers.Conv1D(filters=dim_vector*n_channels, kernel_size=kernel_size,
                           strides=strides, padding=padding,
                           name='primaryCap_conv1d')(inputs)
    dim = output.shape[1]*output.shape[2]'''
    outputs = layers.Reshape(target_shape=(dim//dim_vector,dim_vector), name='primaryCap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

def CapsNet(input_shape, kerl, n_class, num_routing):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=256, kernel_size=kerl, strides=1, padding='valid',
                         activation='relu', name='conv1')(x)
    '''
    conv2 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same',
                         activation='relu', name='conv2')(conv1)
    conv3 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='valid',
                         activation='relu', name='conv3')(conv2)'''
    
    primarycaps = PrimaryCap(conv1, dim_vector=8, n_channels=32, kernel_size=kerl, 
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

def lr_schedule(epoch):
    lr = 1e-3
    return lr*0.9*epoch

def resnet_train_predict():
    (x_train, y_train), (x_test, y_test) = load_data()
    model = resnet_v1(input_shape=(30,16), depth=20, num_classes=2)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=lr_schedule(0)),
                  metrics=['accuracy'])
    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5,
                                   min_lr=0.5e-6)

    save_dir = os.path.join(os.getcwd(), 'save_models')
    model_name = 'trb_resnet_model.{epoch:02d}.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)    
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_accuracy',
                                verbose=1, save_best_only=True,
                                save_weights_only=True)
    cbs = [checkpoint, lr_reducer, lr_scheduler]

    model.fit(x_train, y_train, batch_size=32, epochs=20, 
              validation_data=[x_test, y_test],
              shuffle=True,
              callbacks=cbs)
    
    y_pred = model.predict(x_test, batch_size=48)
    info = 'Training in TRB.....'
    writeMetrics('TRB_result.txt', y_test, y_pred, info)
    
def capsul_train_predict():
    (x_train, y_train), (x_test, y_test), (x_indt, y_indt) = load_data(col=9, row=11)
    model = CapsNet(input_shape=(11,10,2,),kerl=3, n_class=2, num_routing=5)
    
    model.compile(optimizer=Adam(learning_rate=lr_schedule(0)),
                 loss=[margin_loss, 'mse'],
                 loss_weights=[1., 0.325],
                 metrics={'out_caps': 'accuracy'})
    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5,
                                   min_lr=0.5e-6)

    save_dir = os.path.join(os.getcwd(), 'save_models')
    model_name = 'trb_capsul_model.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)    
    checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_out_caps_accuracy',
                                verbose=1, save_best_only=True,
                                save_weights_only=True)
    cbs = [checkpoint, lr_reducer, lr_scheduler]
    model.fit([x_train, y_train], [y_train, x_train],
                  batch_size=32,
                  epochs=10,
                  validation_data=[[x_test, y_test], [y_test, x_test]],
                  shuffle=True,
                  callbacks=cbs)
    model.load_weights(filepath)    
    y_pred, _ = model.predict([x_test, y_test])
    
    info = 'By capsul net Training in TRB, Testing in testdata.....'
    writeMetrics('TRB_result.txt', y_test, y_pred, info)
    
    indt_pred, _ = model.predict([x_indt, y_indt])
    info = 'By capsul net Tringing in TRB, Testing in nCoV-19...'
    writeMetrics('TRB_result.txt', y_indt, indt_pred, info)

def resnet_layer(inputs, num_filters, kernel_size=3, strides=1,
                 activation='relu', batch_normalization=True, conv_first=True):
    ''' 2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    '''
    conv = layers.Conv2D(num_filters, kernel_size=kernel_size, strides=strides,
                         padding='same', kernel_initializer='he_normal',
                         kernel_regularizer=l2(1e-4))
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
        x = conv(x)
        
    return x

def resnet_v1(input_shape, depth, num_classes=2):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth-2)%6 != 0:
        raise ValueError('depth should be 6n+2')
    # Start model definition.
    num_filters = 32
    num_res_blocks = int((depth-2)/6)
    
    inputs = tf.keras.Input(shape=input_shape)
    x = resnet_layer(inputs, num_filters)
    # Instantiate teh stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0: # first layer but not first stack
                strides = 2 # downsample
            y = resnet_layer(x, num_filters, strides=strides)  
            y = resnet_layer(y, num_filters, activation=None)
            
            if stack > 0 and res_block == 0: # first layer but not first stack
                # linear projection residual shortcut connection to match
                # change dims
                x = resnet_layer(x, num_filters, kernel_size=1, strides=strides,
                                 activation=None, batch_normalization=False)
            x = layers.add([x, y])
            x = layers.Activation('relu')(x)           
        num_filters *= 2
        
    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    ax = layers.GlobalAveragePooling2D()(x)
    #x = layers.AveragePooling2D()(x)
    
    ax = layers.Dense(num_filters//8, activation='relu')(ax)
    ax = layers.Dense(num_filters//2, activation='softmax')(ax)
    ax = layers.Reshape((1,1,num_filters//2))(ax)
    ax = layers.Multiply()([ax, x])
    y = layers.Flatten()(ax)
    outputs = layers.Dense(num_classes, activation='softmax',
                           kernel_initializer='he_normal')(y)
    # Instantiate model
    model = models.Model(inputs=inputs, outputs=outputs)
    return model   
def resnet_attention_train_predict(row, col, modelfile, info):
    (x_train, y_train), (x_test, y_test), (x_indt, y_indt) = load_data(col=col, row=row)
    model = resnet_v1(input_shape=(row, col+1, 2), depth=20, num_classes=2)
    
    model.compile(optimizer=Adam(learning_rate=lr_schedule(0)),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5,
                                   min_lr=0.5e-6)

    '''checkpoint = ModelCheckpoint(filepath=modelfile, monitor='val_accuracy',
                                verbose=1, save_best_only=True,
                                save_weights_only=True)
    cbs = [checkpoint, lr_reducer, lr_scheduler]
    model.fit(x_train, y_train,
                  batch_size=32,
                  epochs=10,
                  validation_data=[x_test, y_test],
                  shuffle=True,
                  callbacks=cbs)'''
    
    model.load_weights(modelfile)    
    y_pred = model.predict(x_test)
    
    info += 'By resnet with attention Training in TRB, Testing in testdata.....'
    writeMetrics('TRB_result.txt', y_test, y_pred, info)
    
    indt_pred = model.predict(x_indt)
    info += 'By resnet with attention Tringing in TRB, Testing in nCoV-19...'
    writeMetrics('TRB_result.txt', y_indt, indt_pred, info)

if  __name__ == "__main__":   
    '''for row in [9, 12, 17]:
        for col in [9, 15, 20]:
            save_dir = os.path.join(os.getcwd(), 'save_models', 'model2')
            model_name = 'trb_resnet_model_{}_{}.h5'.format(row,col)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            modelfile = os.path.join(save_dir, model_name) 
            inf = '-----------------\n'
            info = 'using {} amino acids of sequence of CDR3, features={}\n'.format(row, col)
            resnet_attention_train_predict(row, col, modelfile, info)
            #capsul_train_predict()'''
    '''    
    cdr3_seqs, epit_seqs, labels=read_seqs(indepFile)
    # static amino acids frequency
    stat = {}
    for seq in cdr3_seqs:
        stat[len(seq)] = stat.get(len(seq), 0) + 1
   
    items = list(stat.items())
    items.sort(key=lambda x: x[1], reverse = True)
    for item in items:
        print("{0:<4}{1:>5}".format(item[0], item[1]))
    '''
    resnet_attention_train_predict(20,9,'save_models//model1//trb_resnet_model_20_9.h5','Test......')
    
        