import os
import numpy as np
import pandas as pd
import time
from collections import deque
import argparse
import tensorflow as tf
from six import next

def get_UserData():
    col_names = ["user", "r_gender", "age", "mar", "child", "edu", "job", "income", "fa_expend"]
    dummies_col = [ "age", "r_gender", "mar", "job", "child", "income", "fa_expend"]
    
    df = pd.read_csv("/mnt/hdd1/wearly/graphrec/GraphRec/kdeepfashion/user_data.csv", index_col=0)
    df = df[col_names]
    df=pd.get_dummies(df,columns=dummies_col)
    del df["user"]
    return df.values

def get_ItemData():
    col_names = ['item', 'era', 'style', 'gender']
    dummies_col = ["gender", "style", "era"]
    
    df = pd.read_csv('/mnt/hdd1/wearly/graphrec/GraphRec/kdeepfashion/item_data.csv', index_col=0)
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

def get_data_tr(opt):
    global PERC
    df = read_process(f"{opt.TRAIN_PATH}")
    rows = len(df)
    df = df.iloc[np.random.RandomState(seed=329).permutation(rows)].reset_index(drop=True)
    split_index = int(rows * opt.PERC)
    df_train = df[0:split_index]
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test

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
