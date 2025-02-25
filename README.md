# Sentiment Analysis Project

## Introduction
This project aims to build a **Sentiment Analysis System** by scraping news details from **lite.cnn.com**, a lightweight version of CNN's news platform known for its concise and impactful reporting.

By analyzing the sentiments expressed in the news sentences, this system categorizes them into **Positive**, **Neutral**, and **Negative** sentiments, providing valuable insights into the tone of media coverage on various topics.

## Project Workflow
The project involves the following key steps:
1. **Data Collection:** Scraping 200+ news paragraphs from **lite.cnn.com** using web scraping techniques.
2. **Data Annotation:** Labeling each line with its corresponding sentiment category.
3. **Model Training:** Building and training a sentiment analysis model to accurately classify the sentiments.
4. **Deployment:** Exposing the trained model as a **REST API** using **FastAPI**, enabling real-time sentiment analysis.
5. **Optimization and A/B Testing:** Enhancing model inference speed through quantization and conducting **A/B testing** to compare performance against a baseline model.

## Data Collection
- News paragraphs were scraped from **lite.cnn.com**.
- The extracted text was stored in CSV files for further preprocessing.

## Data Cleaning and Preprocessing
- The first 3 and last 7 rows of each dataset were removed to eliminate unwanted text.
- Data was merged into a single dataset and duplicates were removed.
- Sentiments were annotated using **TextBlob**.
- The dataset was balanced by oversampling underrepresented sentiment classes.

## Model Training
Two models were trained:
1. **Baseline Model:** Multinomial Naive Bayes
2. **Optimized Model:** Logistic Regression with Hyperparameter Tuning

### Baseline Model (Naive Bayes) Report:
```
              precision    recall  f1-score   support

    Negative       0.81      1.00      0.89        25
     Neutral       0.82      0.72      0.77        25
    Positive       0.72      0.62      0.67        21

    accuracy                           0.79        71
   macro avg       0.78      0.78      0.78        71
weighted avg       0.79      0.79      0.78        71
```

### New Model (Optimized Logistic Regression) Report:
```
              precision    recall  f1-score   support

    Negative       1.00      1.00      0.93        25
     Neutral       0.83      0.76      0.79        25
    Positive       0.68      0.62      0.65        21

    accuracy                           0.82        71
   macro avg       0.79      0.79      0.79        71
weighted avg       0.80      0.80      0.80        71
```

## Deployment
- The trained models and vectorizer were saved using **joblib**.
- A **FastAPI** service was built to expose the model as a REST API.
- The API performs **A/B testing**, serving predictions from both models at random.


## Future Improvements
- Implement deep learning models for improved sentiment classification.
- Expand dataset size to improve model generalization.
- Optimize API performance for real-time analysis.

---



