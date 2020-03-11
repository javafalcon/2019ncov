#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 08:41:55 2020

@author: weizhong
"""
import pandas
import numpy as np
from sklearn.utils import shuffle

def read_seq_data(randompair=False):
    if randompair:
        negdata = pandas.read_csv('Datasets/negative_random_pair_human_TRB.csv')
        datafile = 'seqdata_randompair.npz'
    else:
        negdata = pandas.read_csv('Datasets/negative_random_CDR3m_human_TRB.csv')
        datafile = 'seqdata.npz'
    neg_CDR3_s = negdata.CDR3_s
    neg_CDR3_m = negdata.CDR3_m
    neg_CDR3_e = negdata.CDR3_e
    neg_Epitope = negdata.Epitope
    
    negseq=[]
    for i in range(len(neg_CDR3_s)):
        s = "" if pandas.isnull( neg_CDR3_s[i]) else neg_CDR3_s[i]
        m = "" if pandas.isnull( neg_CDR3_m[i]) else neg_CDR3_m[i]
        e = "" if pandas.isnull( neg_CDR3_e[i]) else neg_CDR3_e[i]
        negseq.append( s + m + e + neg_Epitope[i])
    
    posdata = pandas.read_csv('Datasets/positive_human_TRB.csv')
    pos_CDR3_s = posdata.CDR3_s
    pos_CDR3_m = posdata.CDR3_m
    pos_CDR3_e = posdata.CDR3_e
    pos_Epitope = posdata.Epitope
    
    posseq=[]
    for i in range(len(pos_CDR3_s)):
        s = "" if pandas.isnull( pos_CDR3_s[i]) else pos_CDR3_s[i]
        m = "" if pandas.isnull( pos_CDR3_m[i]) else pos_CDR3_m[i]
        e = "" if pandas.isnull( pos_CDR3_e[i]) else pos_CDR3_e[i]
        posseq.append( s + m + e + pos_Epitope[i])
    
    np.savez( datafile, negseq=negseq, posseq=posseq)

def read_TRAseq_data(randompair=False):
    if randompair:
        negdata = pandas.read_csv('Datasets/data_3_9/TRA/neg_random_pair_human_TRA.csv')
        datafile = 'seqdata_randompair_TRA.npz'
    else:
        negdata = pandas.read_csv('Datasets/data_3_9/TRA/neg_random_CDR3m_human_TRA.csv')
        datafile = 'seqdata_TRA.npz'
    neg_CDR3 = negdata.CDR3
    neg_Epitope = negdata.Epitope
    negseq=[]
    for i in range(len(neg_CDR3)):
        negseq.append(  neg_CDR3[i] + neg_Epitope[i])
    
    posdata = pandas.read_csv('Datasets/data_3_9/TRA/pos_human_TRA.csv')
    pos_CDR3 = posdata.CDR3
    pos_Epitope = posdata.Epitope
    
    posseq=[]
    for i in range(len(pos_CDR3)):
        posseq.append( pos_CDR3[i] + pos_Epitope[i])
    
    np.savez( datafile, negseq=negseq, posseq=posseq)

def load_seq_data(ptype='TRB', randompair=False):
    if randompair:
        if ptype == 'TRB':
            data = np.load('seqdata_randompair.npz')
        else:
            data = np.load('seqdata_randompair_TRA.npz')
    else:
        if ptype == 'TRB':
            data = np.load('seqdata.npz')
        else:
            data = np.load('seqdata_TRA.npz')
    return data

def onehot(seq, row=60, col=20):
    if col == 20:
        amino_acids = "ACDEFHIGKLMNQPRSTVWY"
    elif col == 21:
        amino_acids = "ACDEFHIGKLMNQPRSTVWYX"
        
    x = np.zeros(shape=(row, col))
    n = len(seq) if len(seq)<=row else row
    for i in range(n):
        k = amino_acids.index(seq[i])
        x[i,k] = 1
    return x

def splitTrainAndTest(x, y, test_size=0.2, random_state=None):
    from sklearn.model_selection import ShuffleSplit
    rs = ShuffleSplit(n_splits=5, test_size=test_size, random_state=random_state)
    for train_index, test_index in rs.split(x, y):
        x_train = x[train_index]
        y_train = y[train_index]
        
        x_test = x[test_index]
        y_test = y[test_index]
    
    return (x_train, y_train), (x_test, y_test)
    
def load_data_onehot(pytpe='TRB', randompair=False, row=60, col=20):
    data = load_seq_data('TRA', randompair)
    
    x_pos = []
    for seq in data['posseq']:
        x_pos.append(onehot(seq, row, col))
    y_pos = np.zeros(shape=(len(x_pos),2))
    y_pos[:,0] = 1
    x_pos = np.array(x_pos)
    
    x_neg = []
    for seq in data['negseq']:
        x_neg.append( onehot(seq,row,col))
    y_neg = np.zeros(shape=(len(x_neg),2))
    y_neg[:,1] = 1
    x_neg = np.array(x_neg)
    
    (x_train_pos, y_train_pos), (x_test_pos, y_test_pos) = splitTrainAndTest(x_pos, y_pos)
    (x_train_neg, y_train_neg), (x_test_neg, y_test_neg) = splitTrainAndTest(x_neg, y_neg)
    
    x_train = np.concatenate((x_train_pos, x_train_neg))
    y_train = np.concatenate((y_train_pos, y_train_neg))
    
    x_test = np.concatenate((x_test_pos, x_test_neg))
    y_test = np.concatenate((y_test_pos, y_test_neg))
    
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)
    
    return( x_train, y_train), (x_test, y_test)
    
def load_data(ptype='TRB'):
    if ptype in ['TRB']:
        data1 = np.load('seqdata_randompair.npz')
        data2 = np.load('seqdata.npz')
    else:
        data1 = np.load('seqdata_randompair_TRA.npz')
        data2 = np.load('seqdata_TRA.npz')
        
    negseq = np.concatenate((data1['negseq'], data2['negseq']))
    
    x_pos = []
    for seq in data1['posseq']:
        x_pos.append(onehot(seq))
    y_pos = np.zeros(shape=(len(x_pos),2))
    y_pos[:,0] = 1
    x_pos = np.array(x_pos)
    
    x_neg = []
    for seq in negseq:
        x_neg.append( onehot(seq))
         
    y_neg = np.zeros( shape=(len(x_neg),2))
    y_neg[:,1] = 1
    x_neg = np.array( x_neg)
    
    (x_train_pos, y_train_pos), (x_test_pos, y_test_pos) = splitTrainAndTest(x_pos, y_pos)
    (x_train_neg, y_train_neg), (x_test_neg, y_test_neg) = splitTrainAndTest(x_neg, y_neg)
    
    x_train = np.concatenate((x_train_pos, x_train_neg))
    y_train = np.concatenate((y_train_pos, y_train_neg))
    
    x_test = np.concatenate((x_test_pos, x_test_neg))
    y_test = np.concatenate((y_test_pos, y_test_neg))
    
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)
    
    return (x_train, y_train), (x_test, y_test)
    
        
if __name__ == "__main__":
    """
    data = load_seq_data()   
    maxlen1, minlen1 = 0, 100
    for d in data['negseq']:
        if len(d) > maxlen1:
            maxlen1 = len(d)
        elif len(d) < minlen1:
            minlen1 = len(d)
    print("In negdata, the max len is: {}, the min len is: {}".format(maxlen1, minlen1))
    
    maxlen2, minlen2 = 0, 100
    for d in data['posseq']:
        if len(d) > maxlen2:
            maxlen2 = len(d)
        elif len(d)< minlen2:
            minlen2 =len(d)
    print("In posdata, the max len is: {}, the min len is: {}".format( maxlen2, minlen2))
    """
    #(x_train, y_train), (x_test, y_test) = load_data_onehot()     
    read_TRAseq_data(randompair=True)
    read_TRAseq_data(randompair=False)        
    #load_data('TRA')
    
    
    