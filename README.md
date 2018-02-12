# Mercari Price Suggestion Challenge
This is the source code of Kaggle competition: Mercari Price Suggestion Challenge.
The goal is to build an algorithm that automatically suggests the right product prices, based on  provided user-inputted text 
descriptions of their products, including details like product category name, brand name, and item condition. 
(https://www.kaggle.com/c/mercari-price-suggestion-challenge)

We submitted three solutions with respect to different solution.
1. LabelEncoder+RNN.py
 - A fixed number of most frequency categorical features (brand name, category, name, description) are encoded by one-hot 
   encoding, Recurrent neural network(RNN) is used for price prediction.
2. Tfidf+Ridge.py
 - This solution uses Tf–idf encoder for the text feature preprocessing, while Ridge regression model is used for price 
   prediction.
3. wordbatch+FM_FTRL+lightGBM.py
 - In this solution, the name and description feature are encoded by wordbatch encoder, this tool separate sentence into 
 phrase instead of words as usual, shows higher performance in this competition. The predicting model is a weighted 
 combination of Follow-the-Regularized-Leader (FTRL), FTRL+Factorization Machines(FM) and light gradient boosting(lightGBM).
 - This solution shows the highest score(0.42395) within my work.

