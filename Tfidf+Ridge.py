#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 16:36:02 2018

@author: hankaei
"""



import pandas as pd
import numpy as np
import scipy

from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer

import gc

#load and prepare data
raw_train = pd.read_csv('../input/train.tsv', sep='\t')
raw_test = pd.read_csv('../input/test.tsv', sep='\t')

df = pd.concat([raw_train, raw_test], 0)
n_train = raw_train.shape[0]
y_train = np.log1p(raw_train["price"])

del raw_train
gc.collect()


###############
print("name feature engineering")
n_name = 10
count = CountVectorizer(min_df=n_name)
df["name"] = df["name"].str.lower()
X_name = count.fit_transform(df["name"])


##############
df["category_name"] = df["category_name"].fillna("other/other/other").astype("category").str.lower()
print("seperate category")
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
df["cat1"] = df["cat1"].astype("category")
df["cat2"] = df["cat2"].astype("category")
df["cat3"] = df["cat3"].astype("category")

print("category feature engineering")
count_category = CountVectorizer()
X_cat1 = count_category.fit_transform(df["cat1"])
X_cat2 = count_category.fit_transform(df["cat2"])
X_cat3 = count_category.fit_transform(df["cat3"])
X_cat = count_category.fit_transform(df["category_name"])
  
  
##############
print("description feature engineering")
n_description_long = 50000
count_descp = TfidfVectorizer(max_features = n_description_long, 
                              ngram_range = (1,3),
                              stop_words = "english")
df["item_description"] = df["item_description"].fillna("no description yet").str.lower()
X_description = count_descp.fit_transform(df["item_description"])



##############
print("brand feature engineering")
n_brand = 2500
df["brand_name"] = df["brand_name"].fillna("other").str.lower().astype("category")
premissing = len(df.loc[df['brand_name'] == 'other'])
def brandfinder(line):
    brand = line[0]
    name = line[1]
    namesplit = name.split(' ')
    if brand == 'other':
        for x in namesplit:
            if x in all_brands:
                return name
    if name in all_brands:
        return name
    return brand
all_brands = set(df['brand_name'].values)
df['brand_name'] = df[['brand_name','name']].apply(brandfinder, axis = 1)
found = premissing-len(df.loc[df['brand_name'] == 'other'])
print(found)
pop_brands = df["brand_name"].value_counts().index[:n_brand]
df.loc[~df["brand_name"].isin(pop_brands), "brand_name"] = "other"
vect_brand = LabelBinarizer(sparse_output=True)
X_brand = vect_brand.fit_transform(df["brand_name"])


##############
print("shipping_condition engineering")
df["item_condition_id"] = df["item_condition_id"].astype("category")
X_shipping_condition = scipy.sparse.csr_matrix(pd.get_dummies(df[["item_condition_id", "shipping"]], sparse = True).values)
  


##############
def wordCount(text):
    try:
        if text == 'no description yet':
            return 0
        else:
            words = [w for w in text.split(" ")]
            return len(words)
    except: 
        return 0
    
description_len = df['item_description'].apply(lambda x: wordCount(x))
name_len = df['name'].apply(lambda x: wordCount(x))

##############
X = scipy.sparse.hstack((X_shipping_condition, 
                         X_description,
                         X_brand,
                         X_cat1,
                         X_cat2,
                         X_cat3,
                         X_cat,
                         np.transpose(np.matrix(description_len)),
                         np.transpose(np.matrix(name_len)),
                         X_name)).tocsr()
  

##############

print([X_shipping_condition.shape, X_cat1.shape, X_cat2.shape, X_cat3.shape,
       X_name.shape, X_description.shape, X_brand.shape])


#Ridge model
X_train = X[:n_train]
model = Ridge(solver = "lsqr", fit_intercept=False)

model.fit(X_train, y_train)

X_test = X[n_train:]
preds = model.predict(X_test)

raw_test["price"] = np.expm1(preds)
raw_test[["test_id", "price"]].to_csv("mysubmission_ridge.csv", index = False)