'''
paper titles "Attribute-Aware Non-Linear Co-Embeddings of Graph Features" Accepted in RecSys 2019
This code was implemented using python 3.5 and TensorFlow  1.7
We would like to thank "Guocong Song" because we utilized parts of his code from "songgc/TF-recomm" in our implementation
'''


import os
import numpy as np
import pandas as pd
import time
from collections import deque
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from six import next
from sklearn import preprocessing
import sys
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import warnings 
warnings.filterwarnings(action='ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES']=''
tf.Session(config=tf.ConfigProto(device_count={"GPU": 0, "CPU": 1}))

def get_UserData():
    col_names = ["user", "r_gender", "age", "mar", "child", "edu", "job", "income", "fa_expend"]
    dummies_col = [ "age", "r_gender", "mar", "job", "child", "income", "fa_expend"]
    
    df = pd.read_csv("./kdeepfashion/user_data.csv", index_col=0)
    df = df[col_names]
    df=pd.get_dummies(df,columns=dummies_col)
    del df["user"]
    return df.values

def get_ItemData():
    col_names = ['item', 'era', 'style', 'gender']
    dummies_col = ["gender", "style", "era"]
    
    df = pd.read_csv('./kdeepfashion/item_data.csv', index_col=0)
    df = df[col_names]
    df=pd.get_dummies(df,columns=dummies_col)
    del df["item"]
    return df.values 
  
def read_process(filname, sep="\t"):
    df = pd.read_csv(filname, index_col=0)
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["rate"] = df["rate"].astype(np.float32)
    return df

def get_data(opt):
    global PERC
    df = read_process(f"{opt.TEST_PATH}")
    rows = len(df)
    df = df.iloc[np.random.RandomState(seed=329).permutation(rows)].reset_index(drop=True)
    split_index = int(rows * opt.PERC)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test

def get_data_HR(opt):
    global PERC
    df_test = read_process(f"{opt.HR_PATH}")
    return df_test

def get_data_Entire(opt):
    global PERC
    df_test = read_process(f"{opt.TEST_PATH}")
    return df_test


class ShuffleIterator(object):

    def __init__(self, inputs, batch_size=10):
        self.inputs = inputs
        self.batch_size = batch_size
        self.num_cols = len(self.inputs)
        self.len = len(self.inputs[0])
        self.inputs = np.transpose(np.vstack([np.array(self.inputs[i]) for i in range(self.num_cols)]))

    def __len__(self):
        return self.len

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        ids = np.random.randint(0, self.len, (self.batch_size,))
        out = self.inputs[ids, :]
        return [out[:, i] for i in range(self.num_cols)]
    
class OneEpochIterator(ShuffleIterator):
    def __init__(self, inputs, batch_size=10):
        super(OneEpochIterator, self).__init__(inputs, batch_size=batch_size)
        if batch_size > 0:
            self.idx_group = np.array_split(np.arange(self.len), np.ceil(self.len / batch_size))
        else:
            self.idx_group = [np.arange(self.len)]
        self.group_id = 0

    def next(self):
        if self.group_id >= len(self.idx_group):
            self.group_id = 0
            raise StopIteration
        out = self.inputs[self.idx_group[self.group_id], :]
        self.group_id += 1
        return [out[:, i] for i in range(self.num_cols)]

def inferenceDense(phase,user_batch, item_batch,idx_user,idx_item, user_num, item_num,UReg=0.05,IReg=0.1):
    with tf.device(opt.DEVICE): 
        user_batch = tf.nn.embedding_lookup(idx_user, user_batch, name="embedding_user")
        item_batch = tf.nn.embedding_lookup(idx_item, item_batch, name="embedding_item")
        
        
        ul1mf=tf.layers.dense(inputs=user_batch, units=opt.MFSIZE,activation=tf.nn.crelu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        il1mf=tf.layers.dense(inputs=item_batch, units=opt.MFSIZE,activation=tf.nn.crelu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        InferInputMF=tf.multiply(ul1mf, il1mf)


        infer=tf.reduce_sum(InferInputMF, 1, name="inference")

        regularizer = tf.add(opt.UW*tf.nn.l2_loss(ul1mf), opt.IW*tf.nn.l2_loss(il1mf), name="regularizer")

    return infer, regularizer

def clip(x):
    return np.clip(x, 1.0, 5.0)
                
                
def GraphRec_test(opt, test, ItemData=False,UserData=False,Graph=False):

    iter_test = OneEpochIterator([test["user"],test["item"],test["rate"]],batch_size=opt.BATCH_SIZE)

    phase = tf.placeholder(tf.bool, name='phase')
    init_op = tf.global_variables_initializer()    
    
    # Load model 
    
    with tf.Session() as sess:
        sess.run(init_op)
        saver = tf.train.import_meta_graph(f'./{opt.SAVE_NAME}/{opt.WEIGHT_NAME}.meta', clear_devices=True)
        saver.restore(sess, f'./{opt.SAVE_NAME}/{opt.WEIGHT_NAME}')
        
        user_batch = tf.get_collection('train_var')[0]
        item_batch = tf.get_collection('train_var')[1]
        infer = tf.get_collection('train_var')[2]
        
        test_err2 = np.array([])
        degreelist=list()
        predlist=list()
        for users, items, rates in iter_test:
            pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                    item_batch: items,
                                                    phase:False})

            pred_batch = clip(pred_batch)            
            test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
        test_err = np.sqrt(np.mean(test_err2))
        finalerror=test_err
        print('\033[31m \033[43m' + 'RMSE Score' + " : " + str(finalerror) + '\033[0m')

        
def GraphRec_hr(opt, test, ItemData=False,UserData=False,Graph=False):

    iter_test = OneEpochIterator([test["user"],test["item"],test["rate"]],batch_size=1)
    
    phase = tf.placeholder(tf.bool, name='phase')
    init_op = tf.global_variables_initializer()    
    
    # Load model
    dt_ = []
    hit_correct = 0
    user_number = 0
    with tf.Session() as sess:
        sess.run(init_op)
        saver = tf.train.import_meta_graph(f'./{opt.SAVE_NAME}/{opt.WEIGHT_NAME}.meta', clear_devices=True)
        saver.restore(sess, f'./{opt.SAVE_NAME}/{opt.WEIGHT_NAME}')
        
        user_batch = tf.get_collection('train_var')[0]
        item_batch = tf.get_collection('train_var')[1]
        infer = tf.get_collection('train_var')[2]
        
        for users, items, rates in iter_test:
            pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                    item_batch: items,
                                                    phase:False})
            pred_batch = clip(pred_batch)
            dt_.append([users[0], items[0], pred_batch[0]])
            #if pred_batch[0] > 2.7 :
            #    hit_correct += 1
            #user_number += 1
            
    #HRscore = hit_correct/user_number
    #print('\033[31m \033[42m' + 'HR@10 Score' + " : " + str(HRscore) + '\033[0m')
    
    predicted_df = pd.DataFrame(dt_, columns=["user", "item", "rate"])
    predicted_df.to_csv("./kdeepfashion/pred_df.csv")
    
        
def GraphRec_infer(opt, test, ItemData=False,UserData=False,Graph=False):

    iter_test = OneEpochIterator([test["user"],test["item"],test["rate"]],batch_size=1)
    
    phase = tf.placeholder(tf.bool, name='phase')
    init_op = tf.global_variables_initializer()    
    
    # Load model 
    dt_ = []
    with tf.Session() as sess:
        sess.run(init_op)
        saver = tf.train.import_meta_graph(f'./{opt.SAVE_NAME}/{opt.WEIGHT_NAME}.meta', clear_devices=True)
        saver.restore(sess, f'./{opt.SAVE_NAME}/{opt.WEIGHT_NAME}')
        
        user_batch = tf.get_collection('train_var')[0]
        item_batch = tf.get_collection('train_var')[1]
        infer = tf.get_collection('train_var')[2]
        
        for users, items, rates in iter_test:
            pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                    item_batch: items,
                                                    phase:False})
            pred_batch = clip(pred_batch)
            dt_.append([users[0], items[0], pred_batch[0]])
    
    predicted_df = pd.DataFrame(dt_, columns=["user", "item", "rate"])

        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TEST_PATH', default="./kdeepfashion/rate_data.csv", help="test dataset path")
    parser.add_argument('--HR_PATH', default="./kdeepfashion/HR_test.csv", help="test dataset path")
    parser.add_argument('--WEIGHT_NAME', default="trained_ckpt-6226", help="saved weighted file name")
    parser.add_argument('--EPOCH_MAX', type=int, default=196)
    parser.add_argument('--BATCH_SIZE', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--DEVICE', default="/gpu:0", help="/gpu:0")
    parser.add_argument('--SAVE_NAME', default="model_kfashion_add_externel", help="save folder name")
    parser.add_argument('--MFSIZE', type=int, default=50)
    parser.add_argument('--UW', type=float, default=0.05)
    parser.add_argument('--IW', type=float, default=0.02)
    parser.add_argument('--LR', type=float, default=0.00003, help='Learning Rate')
    parser.add_argument('--PERC', type=float, default=0.9, help='Training dataset Rate')
    parser.add_argument('--RUN', default="test", help="test+HR / infer")
    opt = parser.parse_args()


    DEVICE="/cpu"

    ################################################     
    if opt.RUN == "test":
        _, df_test = get_data(opt)
        print()
        print('\033[31m' + '----------------------------------' + '\033[0m')
        print(f'■ Size of Test Dataset : {df_test.shape}')
        tf.compat.v1.reset_default_graph()
        GraphRec_test(opt, df_test,
                      ItemData=True, UserData=True, Graph=True)
        
        df_test = get_data_HR(opt)
        df_entire = get_data_Entire(opt)
        tf.compat.v1.reset_default_graph()
        GraphRec_hr(opt, df_entire,
                    ItemData=True, UserData=True, Graph=True)
        
        predicted_df = read_process("./kdeepfashion/pred_df.csv")
        
        hr_correct = 0
        user_num = 0 
        for i in df_test["user"].unique().tolist():
            z = df_test.loc[(df_test["user"] == i), "item"].values[0]
            bool_ = z in predicted_df.loc[(predicted_df["user"] == i)].sort_values("rate", ascending=False).head(10)["item"].tolist()
            user_num += 1
            if bool_:
                hr_correct+=1
            else:
                pass
        HRscore = hr_correct / user_num
        print()
        print('\033[31m' + '----------------------------------' + '\033[0m')
        print(f'■ Number of Test Users used for HR@10 : {user_num}')
        print('\033[31m \033[42m' + 'HR@10 Score' + " : " + str(HRscore) + '\033[0m')
        
    if opt.RUN == "infer":
        df_test = get_data_HR(opt)

        tf.compat.v1.reset_default_graph()

        
        GraphRec_infer(opt, df_test,
                    ItemData=True, UserData=True, Graph=True)

        
        