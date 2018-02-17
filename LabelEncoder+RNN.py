#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  14 16:36:02 2018

@author: hankaei
"""

#raw_train = pd.read_csv('/Users/hankaei/Documents/mercari/data/train.tsv', sep='\t')
#raw_test = pd.read_csv('/Users/hankaei/Documents/mercari/data/test.tsv', sep='\t')
#import tools
from datetime import datetime 
start_real = datetime.now()
import numpy as np
import pandas as pd
import gc

from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten
from keras.optimizers import Adam
from keras.models import Model


# load and prepare data
raw_train = pd.read_table('../input/train.tsv')
raw_test = pd.read_table('../input/test.tsv')
print(raw_train.shape, raw_train.shape)
df = pd.concat([raw_train, raw_test], 0)
n_train = raw_train.shape[0]
y = np.log1p(raw_train["price"])  #difine y as log y 

gc.collect()


#####  full_in the none space and lower the characters
df["name"] = df["name"].str.lower()
df["category_name"] = df["category_name"].fillna("other/other/other").str.lower()
df["brand_name"] = df["brand_name"].str.lower()
all_brands = set(df['brand_name'].values)
df["brand_name"] = df["brand_name"].fillna("other")


#brand_name: fill in unassigned brand_name from name
premissing = len(df.loc[df['brand_name'] == 'other'])  #count unassigned brand_name 
def brandfinder(line):
    brand = line[0]
    name = line[1]
    namesplit = name.split(' ')
    if brand == 'other':
        for x in namesplit:
            if x in all_brands:
                return x
                break
    return brand
df['brand_name'] = df[['brand_name','name']].apply(brandfinder, axis = 1)
found = premissing-len(df.loc[df['brand_name'] == 'other'])  #count how many brand_name filled
print(found)


# description: fill in description and lower the characters
df["item_description"] = df["item_description"].fillna("no description yet").str.lower()


#category_name: split to three sub-categories
categories = df["category_name"]
cat1_list = list()
cat2_list = list()
cat3_list = list()
for category in categories:
        cats = category.split("/")
        cat1, cat2, cat3 = cats[0], cats[1], "".join(cats[2:])
        cat1_list.append(cat1)
        cat2_list.append(cat2)
        cat3_list.append(cat3)
df["cat1"] = cat1_list
df["cat2"] = cat2_list
df["cat3"] = cat3_list



#####add two new features: length of name and description: length of description & length of name
def wordCount(text):
    try:
        if text == 'no description yet':
            return 0
        else:
            words = [w for w in text.split(" ")]
            return len(words)
    except: 
        return 0
    
df['desc_len'] = df['item_description'].apply(lambda x: wordCount(x))
df['name_len'] = df['name'].apply(lambda x: wordCount(x))

# divide data to train and test dataset(not necessary, also can be splited after feature engineering)
train = df[:n_train]
test = df[n_train:]
# split train to train and development
train_df, dev_df = train_test_split(train, random_state=123, train_size=0.999)

#number of train, development, test
n_trains = train_df.shape[0]
n_devs = dev_df.shape[0]
n_tests = test.shape[0]


#feature engineering
print("categorical engineering...")
le = LabelEncoder()

le.fit(df.category_name)
df['category'] = le.transform(df.category_name)

le.fit(df.cat1)
df['cat1'] = le.transform(df.cat1)

le.fit(df.cat2)
df['cat2'] = le.transform(df.cat2)

le.fit(df.cat3)
df['cat3'] = le.transform(df.cat3)

le.fit(df.brand_name)
df['brand_name'] = le.transform(df.brand_name)

del le

#encoding description and name
print("Transforming text data to sequences...")
raw_text = np.hstack([df.item_description.str.lower(), df.name.str.lower()])

print("Fitting tokenizer...")
tok_raw = Tokenizer()
tok_raw.fit_on_texts(raw_text)

print("Transforming text to sequences...")
df['seq_item_description'] = tok_raw.texts_to_sequences(df.item_description.str.lower())
df['seq_name'] = tok_raw.texts_to_sequences(df.name.str.lower())

del tok_raw


#difine some variable to limite the length of features
MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75 
MAX_CATEGORY_SEQ = 8 
MAX_TEXT = np.max([
    np.max(df.seq_name.max()),
    np.max(df.seq_item_description.max())]) + 100
MAX_CATEGORY = np.max(df.category.max()) + 1
MAX_BRAND = np.max(df.brand_name.max()) + 1
MAX_CONDITION = np.max(df.item_condition_id.max()) + 1
MAX_DESC_LEN = np.max(df.desc_len.max()) + 1
MAX_NAME_LEN = np.max(df.name_len.max()) + 1
MAX_CAT1 = np.max(df.cat1.max()) + 1
MAX_CAT2 = np.max(df.cat2.max()) + 1
MAX_CAT3 = np.max(df.cat3.max()) + 1
                     
                     
#difine input data for RNN                     
def get_rnn_data(dataset):
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=MAX_NAME_SEQ),
        'item_desc': pad_sequences(dataset.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
        'brand_name': np.array(dataset.brand_name),
        'category': np.array(dataset.category),
        'item_condition': np.array(dataset.item_condition_id),
        'num_vars': np.array(dataset[["shipping"]]),
        'desc_len': np.array(dataset[["desc_len"]]),
        'name_len': np.array(dataset[["name_len"]]),
        'cat1': np.array(dataset.cat1),
        'cat2': np.array(dataset.cat2),
        'cat3': np.array(dataset.cat3),
    }
    return X


#define train,dev,test after preprocessing/feature engineering
train = df[:n_trains]
dev = df[n_trains:n_trains+n_devs]
test = df[n_trains+n_devs:]

X_train = get_rnn_data(train)
Y_train = y.reshape(-1, 1)[:n_trains,:]

X_dev = get_rnn_data(dev)
Y_dev = y.reshape(-1, 1)[n_trains:,:]

X_test = get_rnn_data(test)



#difine the RNN model
def new_rnn_model(lr=0.001, decay=0.0):
    #set input layer
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")

    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")
    desc_len = Input(shape=[1], name="desc_len")
    name_len = Input(shape=[1], name="name_len")
    cat1 = Input(shape=[1], name="cat1")
    cat2 = Input(shape=[1], name="cat2")
    cat3 = Input(shape=[1], name="cat3")
    #embedding
    emb_name = Embedding(MAX_TEXT, 20)(name)
    emb_item_desc = Embedding(MAX_TEXT, 60)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)
    emb_desc_len = Embedding(MAX_DESC_LEN, 5)(desc_len)
    emb_name_len = Embedding(MAX_NAME_LEN, 5)(name_len)
    emb_cat1 = Embedding(MAX_CAT1, 10)(cat1)
    emb_cat2 = Embedding(MAX_CAT2, 10)(cat2)
    emb_cat3 = Embedding(MAX_CAT3, 10)(cat3)
    
    #recurrent unit for description and name
    rnn_layer1 = GRU(16) (emb_item_desc)
    rnn_layer2 = GRU(8) (emb_name)
    #main layer, combine rnn_layer and other feature
    main_l = concatenate([
        Flatten() (emb_brand_name)
        , Flatten() (emb_item_condition)
        , Flatten() (emb_desc_len)
        , Flatten() (emb_name_len)
        , Flatten() (emb_cat1)
        , Flatten() (emb_cat2)
        , Flatten() (emb_cat3)
        , rnn_layer1
        , rnn_layer2
        , num_vars
    ])
    main_l = Dropout(0.1)(Dense(512,kernel_initializer='normal',activation='relu') (main_l))
    main_l = Dropout(0.1)(Dense(256,kernel_initializer='normal',activation='relu') (main_l))
    main_l = Dropout(0.1)(Dense(128,kernel_initializer='normal',activation='relu') (main_l))
    main_l = Dropout(0.1)(Dense(64,kernel_initializer='normal',activation='relu') (main_l))

    output = Dense(1, activation="linear") (main_l)
    
    model = Model([name, item_desc, brand_name , item_condition, 
                   num_vars, desc_len, name_len, cat1, cat2, cat3], output)

    optimizer = Adam(lr=lr, decay=decay)
    model.compile(loss = 'mse', optimizer = optimizer)

    return model



#set batch and epochs(how many times to train)
BATCH_SIZE = 512 * 3
epochs = 2
#set learning rate decay every update
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(n_trains / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.005, 0.001
lr_decay = exp_decay(lr_init, lr_fin, steps)

rnn_model = new_rnn_model(lr=lr_init, decay=lr_decay)
rnn_model.fit(
        X_train, Y_train, epochs=epochs, batch_size=BATCH_SIZE,
        validation_data=(X_dev, Y_dev), verbose=1,)

#evaluate function 
def rmsle(Y, Y_pred):
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y )))

Y_dev_preds = rnn_model.predict(X_dev, batch_size=BATCH_SIZE)
print(" RMSLE error:", rmsle(Y_dev, Y_dev_preds))

#get predict values
rnn_preds = rnn_model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)

#return y to original y, for logy before
raw_test["price"] = np.expm1(rnn_preds)
raw_test[["test_id", "price"]].to_csv("mysubmission_ridge.csv", index = False)









