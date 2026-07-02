# Customer Review Sentiment Analysis Using NLP and Machine Learning

## Project Overview

This project is an NLP sentiment analysis project where I worked with Amazon customer review data and built a machine learning model to classify reviews as positive or negative.

I wanted this project to be more than just training a model. My goal was to understand how text data is cleaned, converted into numerical features, and used in a real machine learning classification problem.

Customer reviews are useful because they show what people like or dislike about a product. If a company receives thousands of reviews, it is not realistic to read every single one manually. A sentiment analysis model can help quickly identify customer feedback patterns, especially negative reviews that may need attention.

## Business Problem

Online businesses receive a large number of product reviews. Some reviews are positive, while others show customer frustration, product issues, or poor user experience.

The main question for this project was:

**Can we use the written review text to predict whether a customer review is positive or negative?**

This type of model could help a business:

* Find unhappy customers faster
* Track customer satisfaction
* Identify product problems
* Prioritize reviews that need attention
* Understand customer feedback at scale

## Dataset

This project uses an Amazon customer review dataset with review text and star ratings.

Note: The dataset file is not included in this repository due to file size and licensing considerations. The dataset can be downloaded from Kaggle.

The main columns used were:

* `reviews.text`
* `reviews.rating`

I created sentiment labels from the star ratings:

* Ratings 4 and 5 = positive
* Ratings 1 and 2 = negative
* Rating 3 = removed as neutral

I removed neutral reviews because 3-star reviews can be mixed or unclear, and this project focuses on binary sentiment classification.

## Tools and Technologies

* Python
* pandas
* NumPy
* matplotlib
* NLTK
* scikit-learn
* TF-IDF Vectorization
* Logistic Regression
* Multinomial Naive Bayes
* Random Forest Classifier

## Project Workflow

1. Loaded and inspected the customer review dataset
2. Selected the review text and rating columns
3. Created sentiment labels from star ratings
4. Explored sentiment distribution and review length
5. Cleaned and preprocessed the review text
6. Converted text into numerical features using TF-IDF
7. Trained multiple machine learning models
8. Evaluated models using accuracy, precision, recall, F1-score, confusion matrix, and ROC curve
9. Compared model performance
10. Performed error analysis on incorrect predictions

## Text Preprocessing

The review text was cleaned before modeling. The cleaning steps included:

* Lowercasing text
* Removing punctuation
* Removing numbers
* Removing stopwords
* Lemmatizing words

This helped make the text more consistent before using TF-IDF.

## Feature Engineering

I used TF-IDF to convert cleaned review text into numerical features.

TF-IDF helps give more importance to words that are useful for classification and less importance to words that appear too often across many reviews.

I used both single words and two-word phrases because phrases like “not good” can be important in sentiment analysis.

## Models Trained

I trained and compared three models:

* Logistic Regression
* Multinomial Naive Bayes
* Random Forest Classifier

At first, Random Forest and Naive Bayes had higher overall accuracy. However, the dataset was highly imbalanced, with many more positive reviews than negative reviews.

Because of that, accuracy alone was not enough to choose the best model.

## Final Model Choice

I selected **Logistic Regression** as the best model.

The reason is that Logistic Regression performed better on negative reviews compared to the other models. This was important because negative reviews are usually more valuable from a business perspective. They can show unhappy customers, product problems, or areas where the company needs to improve.

Even though Logistic Regression did not have the highest overall accuracy, it had stronger negative-class recall, which made it more useful for this project.

## Error Analysis

I also reviewed examples where the model made incorrect predictions.

Some mistakes happened because reviews were:

* Very short
* Mixed with both positive and negative words
* Unclear
* Written in a way that did not fully match the star rating
* Difficult to understand without deeper context

This showed me that traditional TF-IDF models can work well, but they do not fully understand deeper language meaning like sarcasm or mixed opinions.

## Future Improvements

In the future, I could improve this project by:

* Trying Linear SVM
* Tuning TF-IDF parameters
* Testing different imbalance-handling techniques
* Using transformer models like BERT
* Building a simple dashboard for review monitoring
* Turning the model into a web app or API

## What I Learned

This project helped me understand how NLP projects are different from regular tabular machine learning projects.

I learned how to clean text data, create TF-IDF features, compare models, evaluate imbalanced classification problems, and think about model performance from a business perspective.
