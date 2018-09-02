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

Code file
1. combine_yelp_twitter.py and 2. combine_yelp_twitter_balance.py:     Combine the data

3. preprocessing.py and 4. preprocessing_CombineData_IMDB.py:     Preprocess the data

5. RandomForest.py and 6. Logistic_Regression.py:     Build models and save the results

7. Apply_RF_LR_on_Ubuntu.py:     Apply models on the Ubuntu dataset

8. Plotting_ubuntu.py:     Plot figures about the sentiment analysis

9. count_new_class_labels.py:     Create a new feature called the "new class label" and count the number of different labels







