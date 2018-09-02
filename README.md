# Accenture_capstone

This is a project for Accenture company.

Project title:
Using sentiment analysis to identify the positive sentiment from customers in end-to-end dialogue systems

Data sourse:
yelp data, twitter data, IMDB data, Ubuntu corpus

Models:
Random Forest, Logistic regression, SVC, LSTM

Purpose: 
Use models to train yelp data and twitter data, and then apply these models on IMDB data. 
choose the best one and do sentiment analysis on Ubuntu corpus. plot the sentiment changes and do analysis about
improving the chatbot performance. 


NB：
As for some large data which cannot be submitted in the GitHub, I compress them into zips.


Combine the data: combine_yelp_twitter.py and combine_yelp_twitter_balance.py

Preprocess the data: preprocessing.py and preprocessing_CombineData_IMDB.py

Build models and save the results: RandomForest.py and Logistic_Regression.py

Apply models on the Ubuntu dataset: Apply_RF_LR_on_Ubuntu.py

Plot figures about the sentiment analysis: Plotting_ubuntu.py

Create a new feature called the "new class label" and count the number of different labels: count_new_class_labels.py



