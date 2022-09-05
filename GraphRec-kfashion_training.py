'''
paper titles "Attribute-Aware Non-Linear Co-Embeddings of Graph Features" Accepted in RecSys 2019
This code was implemented using python 3.5 and TensorFlow  1.7
We would like to thank "Guocong Song" because we utilized parts of his code from "songgc/TF-recomm" in our implementation
'''


import os
import sys
import numpy as np
import pandas as pd
import time
from collections import deque
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from six import next

from sklearn import preprocessing
from sklearn import preprocessing
from scipy.sparse import lil_matrix
from scipy.sparse import coo_matrix

from utils.iterator import *
from utils.load_dataset import *

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import warnings 
warnings.filterwarnings(action='ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def inferenceDense(phase,user_batch, item_batch,idx_user,idx_item, UReg=0.05, IReg=0.1):
    with tf.device(opt.DEVICE): 
        user_batch = tf.nn.embedding_lookup(idx_user, user_batch, name="embedding_user")
        item_batch = tf.nn.embedding_lookup(idx_item, item_batch, name="embedding_item")
        print(user_batch.shape)
        print(item_batch.shape)
        
        ul1mf=tf.layers.dense(inputs=user_batch, units=opt.MFSIZE,activation=tf.nn.crelu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        il1mf=tf.layers.dense(inputs=item_batch, units=opt.MFSIZE,activation=tf.nn.crelu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        InferInputMF=tf.multiply(ul1mf, il1mf)


        infer=tf.reduce_sum(InferInputMF, 1, name="inference")

        regularizer = tf.add(opt.UW*tf.nn.l2_loss(ul1mf), opt.IW*tf.nn.l2_loss(il1mf), name="regularizer")

    return infer, regularizer


def optimization(infer, regularizer, rate_batch, learning_rate=0.0005, reg=0.1):
    with tf.device(DEVICE):
        global_step = tf.train.get_global_step()
        assert global_step is not None
        cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
        cost = tf.add(cost_l2, regularizer)
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
    return cost, train_op

def clip(x):
    return np.clip(x, 1.0, 5.0)

                
def GraphRec(opt, train, test, USER_NUM, ITEM_NUM, ItemData=False,UserData=False,Graph=False):
    
    AdjacencyUsers = np.zeros((USER_NUM,ITEM_NUM), dtype=np.float32)
    DegreeUsers = np.zeros((USER_NUM,1), dtype=np.float32)
    
    AdjacencyItems = np.zeros((ITEM_NUM,USER_NUM), dtype=np.float32)
    DegreeItems =  np.zeros((ITEM_NUM,1), dtype=np.float32)
    
    for index, row in train.iterrows():
        userid=int(row['user'])
        itemid=int(row['item'])
        AdjacencyUsers[userid][itemid]=row['rate']/5.0
        AdjacencyItems[itemid][userid]=row['rate']/5.0
        DegreeUsers[userid][0]+=1
        DegreeItems[itemid][0]+=1
    
    DUserMax=np.amax(DegreeUsers) 
    DItemMax=np.amax(DegreeItems)
    DegreeUsers=np.true_divide(DegreeUsers, DUserMax)
    DegreeItems=np.true_divide(DegreeItems, DItemMax)
    
    AdjacencyUsers=np.asarray(AdjacencyUsers,dtype=np.float32)
    AdjacencyItems=np.asarray(AdjacencyItems,dtype=np.float32)
    
    if(Graph):
        UserFeatures= np.concatenate((np.identity(USER_NUM,dtype=np.bool_), AdjacencyUsers,DegreeUsers), axis=1) 
        ItemFeatures= np.concatenate((np.identity(ITEM_NUM,dtype=np.bool_), AdjacencyItems,DegreeItems), axis=1) 
    else:
        UserFeatures=np.identity(USER_NUM,dtype=np.bool_)
        ItemFeatures=np.identity(ITEM_NUM,dtype=np.bool_)

    if(UserData):
        UsrDat=get_UserData()
        UserFeatures=np.concatenate((UserFeatures,UsrDat), axis=1) 

    if(ItemData):
        ItmDat=get_ItemData()
        ItemFeatures=np.concatenate((ItemFeatures,ItmDat), axis=1) 

    UserFeaturesLength=UserFeatures.shape[1]
    ItemFeaturesLength=ItemFeatures.shape[1]
    
    print(f"UserFeatures shape : {UserFeatures.shape}")
    print(f"ItemFeatures shape : {ItemFeatures.shape}")
    
    samples_per_batch = len(train) // opt.BATCH_SIZE

    iter_train = ShuffleIterator([train["user"],train["item"],train["rate"]],batch_size=opt.BATCH_SIZE)

    iter_test = OneEpochIterator([test["user"],test["item"],test["rate"]],batch_size=10000)


    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float64, shape=[None])
    phase = tf.placeholder(tf.bool, name='phase')
    
    
    w_user = tf.constant(UserFeatures,name="userids", shape=[USER_NUM,UserFeatures.shape[1]],dtype=tf.float64)
    w_item = tf.constant(ItemFeatures,name="itemids", shape=[ITEM_NUM, ItemFeatures.shape[1]],dtype=tf.float64)
    
    
    with tf.device(opt.DEVICE): 
        user_batch2 = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
        item_batch2 = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")
        
        ul1mf=tf.layers.dense(inputs=user_batch2, units=opt.MFSIZE,activation=tf.nn.crelu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        il1mf=tf.layers.dense(inputs=item_batch2, units=opt.MFSIZE,activation=tf.nn.crelu, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        InferInputMF=tf.multiply(ul1mf, il1mf)


        infer=tf.reduce_sum(InferInputMF, 1, name="inference")

        regularizer = tf.add(opt.UW*tf.nn.l2_loss(ul1mf), opt.IW*tf.nn.l2_loss(il1mf), name="regularizer")
    
    #infer, regularizer = inferenceDense(phase, user_batch, item_batch, w_user, w_item)
        
    global_step = tf.compat.v1.train.get_or_create_global_step()
    _, train_op = optimization(infer, regularizer, rate_batch, learning_rate=opt.LR, reg=0.09)
    
    # 테스트에 쓰일 변수를 텐서플로 컬렉션에 저장
    train_var = [user_batch, item_batch, infer]
    for var in train_var:
        tf.add_to_collection('train_var', var)
    
    init_op = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    finalerror=-1
    best_err = 99999
    
    # saver 객체 생성
    save_dir = f"./{opt.SAVE_NAME}"
    if os.path.exists(save_dir) == False :
        print('Making Weights Folder')
        os.mkdir(save_dir)
        
    saver = tf.train.Saver()
    train_rmse = []
    valid_rmse = []
    with tf.Session(config=config) as sess:
        sess.run(init_op)
        print("{}  /   {}    {}    {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        errors = deque(maxlen=samples_per_batch)
        start = time.time()
        for i in range(opt.EPOCH_MAX * samples_per_batch):
            #users, items, rates,y,m,d,dw,dy,w = next(iter_train)
            users, items, rates = next(iter_train)
            _, pred_batch = sess.run([train_op, infer], feed_dict={user_batch: users,
                                                                   item_batch: items,
                                                                   rate_batch: rates,
                                                                   phase:True})
            pred_batch = clip(pred_batch)
            errors.append(np.power(pred_batch - rates, 2))
            
            if i % samples_per_batch == 0:
                train_err = np.sqrt(np.mean(errors))
                
                test_err2 = np.array([])
                degreelist=list()
                predlist=list()
                for users, items, rates in iter_test:
                    pred_batch = sess.run(infer, feed_dict={user_batch: users,
                                                            item_batch: items,
                                                            phase:False})

                    pred_batch = clip(pred_batch)            
                    test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                end = time.time()
                test_err = np.sqrt(np.mean(test_err2))
                finalerror=test_err
                print("{:3d}  /   {:f}    {:f}    {:f}(s)".format(i // samples_per_batch, train_err, test_err, end - start))
                train_rmse.append(train_err)
                valid_rmse.append(test_err)
                start = end
                # https://excelsior-cjh.tistory.com/163
                if test_err < best_err:
                    step = i
                    saver.export_meta_graph(f"./{opt.SAVE_NAME}/trained_ckpt.meta", collection_list=['train_var'])
                    saver.save(sess, f"./{opt.SAVE_NAME}/trained_ckpt", global_step=step)
                    print('Trained_model_saved')
                    best_err = test_err

    history = {}
    history["train_rmse"] = train_rmse
    history["valid_rmse"] = valid_rmse
    
    fig=plt.figure()
    plt.plot(history["train_rmse"])
    plt.plot(history["valid_rmse"])
    plt.title('Model RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(f"./{opt.SAVE_NAME}/training_result.png")
    plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--TRAIN_PATH', default="./kdeepfashion/rate_data.csv", help="train dataset path")
    parser.add_argument('--EPOCH_MAX', type=int, default=600)
    parser.add_argument('--BATCH_SIZE', type=int, default=1000, help='total batch size for all GPUs')
    parser.add_argument('--DEVICE', default="/gpu:0", help="/gpu:0")
    parser.add_argument('--SAVE_NAME', default="model_kfashion_add_externel_HR", help="save folder name")
    parser.add_argument('--MFSIZE', type=int, default=50)
    parser.add_argument('--UW', type=float, default=0.05)
    parser.add_argument('--IW', type=float, default=0.02)
    parser.add_argument('--LR', type=float, default=0.00003, help='Learning Rate')
    parser.add_argument('--PERC', type=float, default=0.9, help='Training dataset Rate')
    opt = parser.parse_args()


    DEVICE="/gpu:0"

    ############# ML 100k dataset ###########

    df_train, df_test = get_data_tr(opt)
    
    USER_NUM = df_train.append(df_test)["user"].nunique()
    ITEM_NUM = df_train.append(df_test)["item"].nunique()
    

    #With Graph Features
    tf.compat.v1.reset_default_graph()
    GraphRec(opt, df_train, df_test, USER_NUM, ITEM_NUM,
             ItemData=True, UserData=True, Graph=True)

    #############################################