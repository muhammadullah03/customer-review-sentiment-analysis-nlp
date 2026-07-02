# Customer Review Sentiment Analysis Using NLP and Machine Learning

## Project Overview

This project builds a Natural Language Processing machine learning pipeline to classify Amazon customer reviews as positive or negative based on review text.

Customer reviews contain valuable information about product quality, customer satisfaction, and user experience. This project shows how machine learning can help businesses automatically identify customer sentiment at scale.

## Business Problem

Online businesses receive large volumes of product reviews. Identifying negative reviews quickly is important because negative reviews often represent unhappy customers or product issues.

This project answers the question:

**Can we predict whether a customer review is positive or negative using only the written review text?**

## Dataset

The project uses an Amazon customer review dataset with review text and star ratings.

The key columns used are:

- `reviews.text`
- `reviews.rating`

Sentiment labels were created from ratings:

- Ratings 4 and 5 = Positive
- Ratings 1 and 2 = Negative
- Rating 3 = Removed as neutral

## Tools and Technologies

- Python
- pandas
- NumPy
- matplotlib
- NLTK
- scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Multinomial Naive Bayes
- Random Forest Classifier

## Project Workflow

1. Loaded and inspected the customer review dataset
2. Created sentiment labels from star ratings
3. Explored class imbalance and review length
4. Cleaned and preprocessed review text
5. Converted text into numerical features using TF-IDF
6. Trained multiple machine learning models
7. Evaluated models using accuracy, precision, recall, F1-score, confusion matrix, and ROC curve
8. Compared models based on business value
9. Performed error analysis on incorrect predictions

## Model Results

The dataset was highly imbalanced, with many more positive reviews than negative reviews.

Although Naive Bayes and Random Forest achieved higher overall accuracy, they performed poorly on negative reviews.

Logistic Regression was selected as the best model because it had stronger recall for the negative class. This means it was better at identifying unhappy customers.

## Final Model Choice

**Best Model: Logistic Regression**

Reason:

- Better performance on negative reviews
- Stronger negative-class recall
- More useful for customer support and business monitoring
- Better balance between accuracy and business value

## Error Analysis

Some incorrect predictions happened because reviews were:

- Very short
- Mixed in sentiment
- Unclear or vague
- Difficult to understand without deeper context

This shows one limitation of traditional TF-IDF models: they do not fully understand sarcasm, context, or complex language meaning.

## Future Improvements

Possible future improvements include:

- Trying Linear SVM
- Tuning TF-IDF parameters
- Applying imbalance-handling methods
- Testing transformer models such as BERT
- Building a sentiment dashboard
- Deploying the model as a web application or API

## What I Learned

This project strengthened my understanding of:

- NLP preprocessing
- Text feature engineering
- TF-IDF vectorization
- Model comparison
- Imbalanced classification
- Business-focused model evaluation
- Error analysis
