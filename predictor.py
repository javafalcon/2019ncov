#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 11:04:15 2020

@author: weizhong
"""
from ncovdata import load_data_onehot, load_data
import numpy as np
import os
from keras import callbacks
import argparse   
from keras import layers, models, optimizers
from keras import backend as K
import tensorflow as tf
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from sklearn.utils import class_weight, shuffle, resample


def CapsNet(input_shape, n_class, num_routing, kernel_size=7):
    """
    A Capsule Network on 2019ncon.
    :param input_shape: data shape, 3d, [width, height, channels],for example:[21,28,1]
    :param n_class: number of classes
    :param num_routing: number of routing iterations
    :return: A Keras Model with 2 inputs and 2 outputs
    """
    x = layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=128, kernel_size=kernel_size, strides=1, padding='valid', activation='relu', name='conv1')(x)
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
    primarycaps = PrimaryCap(conv1, dim_vector=16, n_channels=32, kernel_size=kernel_size, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_vector=32, num_routing=num_routing, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='out_caps')(digitcaps)
    
    # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked = Mask()([digitcaps, y])  # The true label is used to mask the output of capsule layer.
    x_recon = layers.Dense(512, activation='relu')(masked)
    x_recon = layers.Dense(1024, activation='relu')(x_recon)
    x_recon = layers.Dense(np.prod(input_shape), activation='sigmoid')(x_recon)
    x_recon = layers.Reshape(target_shape=input_shape, name='out_recon')(x_recon)

    # two-input-two-output keras Model
    return models.Model([x, y], [out_caps, x_recon])

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, data, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    (x_train, y_train), (x_test, y_test) = data
    
    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=args.debug)
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (0.9 ** epoch))

    # compile the model
    
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., args.lam_recon],
                  metrics={'out_caps': 'accuracy'})
    
    
    # Training without data augmentation:
    
    model.fit([x_train, y_train], [y_train, x_train], 
              batch_size=args.batch_size, 
              epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], 
              callbacks=[log, tb, checkpoint, lr_decay])
    
    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def test(model, data):
    x_test, y_test = data
    y_pred, x_recon = model.predict([x_test, y_test], batch_size=100)
    print('-'*50)
    y_p = np.argmax(y_pred, 1)
    return y_p

def writeMetrics(metricsFile, y_true, predicted_Probability, noteInfo=''):
    from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
    from sklearn.metrics import f1_score,roc_auc_score,recall_score,precision_score

    predicts = np.array(predicted_Probability> 0.5).astype(int)
    predicts = predicts[:,0]
    labels = y_true[:,0]
    cm=confusion_matrix(labels,predicts)
    with open(metricsFile,'a') as fw:
        if noteInfo:
            fw.write(noteInfo + '\n')
        for i in range(2):
            fw.write(str(cm[i,0]) + "\t" +  str(cm[i,1]) + "\n" )
        fw.write("ACC: %f "%accuracy_score(labels,predicts))
        fw.write("\nF1: %f "%f1_score(labels,predicts))
        fw.write("\nRecall: %f "%recall_score(labels,predicts))
        fw.write("\nPre: %f "%precision_score(labels,predicts))
        fw.write("\nMCC: %f "%matthews_corrcoef(labels,predicts))
        fw.write("\nAUC: %f\n "%roc_auc_score(labels,predicted_Probability[:,0]))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lam_recon', default=0.6, type=float)  # 1200 * 0.0005, paper uses sum of SE, here uses MSE
    parser.add_argument('--num_routing', default=5, type=int)  # num_routing should > 0
    parser.add_argument('--shift_fraction', default=0.1, type=float)
    parser.add_argument('--debug', default=0, type=int)  # debug>0 will save weights by TensorBoard
    parser.add_argument('--save_dir', default='./log')
    parser.add_argument('--is_training', default=1, type=int)
    parser.add_argument('--weights', default=None)
    parser.add_argument('--lr', default=0.001, type=float)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # load data
    #(x_train, y_train), (x_test, y_test) = load_data_onehot(randompair=True)
    (x_train, y_train), (x_test, y_test) = load_data('TRA')
    x_train = x_train.reshape(-1, x_train.shape[1], x_train.shape[2], 1).astype('float32')
    x_test = x_test.reshape(-1, x_test.shape[1], x_test.shape[2], 1).astype('float32')
    
            
    print("predicting x_train.shape:{}".format(x_train.shape))
    # define model
    model = CapsNet(input_shape=x_train.shape[1:],
                    n_class=len(np.unique(np.argmax(y_train, 1))),
                    num_routing=args.num_routing, kernel_size=9)
    model.summary()
    #plot_model(model, to_file=args.save_dir+'/model.png', show_shapes=True)

    train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)

    #y_pred = test(model=model, data=(x_test, y_test))
    y_pred, x_recon = model.predict([x_test, y_test], batch_size=100)
    K.clear_session()
    tf.reset_default_graph()
    writeMetrics('result.txt', y_test, y_pred, 'samples rate 1:2 Predicted Metrics:')
 
        