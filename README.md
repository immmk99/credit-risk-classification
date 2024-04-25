# Module 20: credit-risk-classification

## Overview of the analysis
The goal of this project is to develop a predictive model capable of assessing the creditworthiness of borrowers applying for loans. Leveraging historical lending data from a peer-to-peer lending services company, the aim is to build a model that accurately predicts whether a borrower represents a healthy loan prospect or poses a high risk of default.
The dataset utilized in this project consists of comprehensive information on past lending activities, including loan status (healthy or high risk of default), loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, and total debt. This dataset serves as the foundation for training and evaluating the predictive model. 
- The dataset was split into independent (X) and dependent (y) subsets. The dependent set (y) comprised the current loan status, while the remaining elements served as independent features for model training and testing.
- Both the independent X and dependent y datasets were further divided into two additional subsets: one for training the model and one for testing it. This partitioning was performed to allocate 75% of the data for training and reserve 25% for testing the final model.
- To address any potential imbalance in feature influence within the model, the independent training data was used to create a scaling factor. This scaling factor was then applied to both the independent training and testing data subsets.
- After scaling, 75% of the independent variable data was utilized to train the model, while the remaining 25% was allocated for testing purposes.
- The model's performance was evaluated based on its ability to predict loan statuses accurately. Evaluation metrics such as confusion matrices and classification reports were generated to provide insights into the predictive accuracy, precision, recall, and F1-score of each model.

## Results
**Confusion matrix** 
[[18663,   102],
[   56,   563]]

**Classification report**
                         precision    recall  f1-score   support

     Healthy (low-risk)       1.00      0.99      1.00     18765
Non-Healthy (high-risk)       0.85      0.91      0.88       619

               accuracy                           0.99     19384
              macro avg       0.92      0.95      0.94     19384
           weighted avg       0.99      0.99      0.99     19384
           
## Summary
The machine learning model exhibits exceptional performance in identifying both healthy loans and high-risk loans. For healthy loans, the model achieves a precision score of 1.00, indicating all predicted healthy loans are indeed healthy, with a recall of 0.99, effectively capturing 99% of actual healthy loans. Similarly, for high-risk loans, the model maintains a precision and recall of 0.85, successfully identifying 85% of high-risk loans while minimizing false positives. The F1-scores for both categories are excellent, with a perfect score of 1.00 for healthy loans and 0.91 for high-risk loans. These results, coupled with substantial support values of 18765 for healthy loans and 619 for high-risk loans, demonstrate the model's robustness and reliability. Deploying this model can significantly enhance the company's lending processes by accurately identifying both low-risk and high-risk loan applicants, thereby mitigating financial risks associated with loan defaults.
