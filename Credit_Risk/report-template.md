# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any other algorithms).

## Results

Using bulleted lists, describe the accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
    * Description of Model 1 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.









# askarovamari
# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

  In the credit risk classification challenge I used a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.
  The dataset included 77,535 lines of data and was devided into training and testing sets. Logistic Regression Model 1 was built based on the training set, utilizing the LogisticRegression module of scikit-learn. The model was then used for the testing data as well. The purpose of the model was to identify whether a loan to the borrower in the test data would be of a high or a low risk.
  Dataset included the below attributes:
  - Loan size
  - Interest rate
  - Borrower income
  - Debt to income
  - Number of accounts
  - Derogatory marks
  - Amount of total debt
  - Loan status

  The goal of this challenge was to try to predict status of loan, whether it's 0 or 1, where 0 means low-risk or helathy loan and 1 stands for a high risk loan. The first set of variables include the majority of values and counts up to 75036 of healthy loan. The second set of variables counted 2500 of high risk loan.

  Major steps of the Machine Learning model are laid out below:
  1. Split the Data into Training and Testing Sets
  2. Create a Logistic Regression Model with the Original Data
  3. Evaluate the model’s performance by doing the following:
  - Generate a confusion matrix
  - Print the classification report
    

## Results
 
  Below are primary results of the Logistic Regression Model:

  Counting variable we've found out that the dataset is frankly imbalanced, as around 97% of data represents heealthy loans. The model predicted a healthy loan with 100% of precision, whereas a high-risk loan is at precision level of 84%. The balanced accuracy of the model is 99%.

  The model indicates a recall score of 99% for the low-risk loans and 94% for the high-risk loans. The scores imply that for all the instances where the loans were actually healthy, 99% of the times they were classified correctly. F1-score of healthy loans indicates absolute reliability of the data with a value of 1 = 100%, when the same metric for high-risk loans shows lower score of 89%.
  
  




















# Steph
# Module 12 Report 

## Overview

The purpose of this analysis is to develop a model utilizing existing customer data that can predict credit-risk of banking clients requesting a loan.  

Model training and analysis was all done with scikit-learn in Python ([file](Credit_Risk/credit_risk_classification.ipynb)).  [Data](Credit_Risk/Resources/lending_data.csv) from existing customers on loan status (healthy or high risk of default), loan size, interest rate, borrower income, debt to income, number of accounts, derogatory marks, and total debt was split into training data and testing data with scikit-learn train_test_split.  The training data was then used to train a logistic regression model with scikit-learn to determine if loan status would be healthy or high risk based on client characteristics.  To assess the effectiveness of the model, the test data from existing customers was used to make loan status predictions.  These predictions we compared against actual loan status, and used to create a scikit-learn confusion matrix and classification report.  

## Results

Confusion Matrix: 
- [18679, 80], [67, 558]
  - Confusion matrix indicates prescence of false negatives and false positives

Classification Report:
- Accuracy: Accuracy value of 0.99 indicates a strong ability of this model to make predictions.
- Healthy Loan: Precision = 1.00, Recall = 1.00, F1-Score = 1.00, Support = 18759
  - There is high precision, recall, and F-1 score indicating that this model is very effective at identifying "Healthy" loans.
- High-Risk Loan: Precision = 0.87, Recall = 0.89, F1-Score = 0.88, Support = 625
  - There is fairly strong precision, recall, and F-1 scores, indicating that this model is decent at identifying "High-Risk" loans.

## Summary

This model shows more difficulty in predicting high risk loans compared to predicting healthy loans. It would be important to not only minimize false-negatives (potentially not giving loans to individuals that are low risk of default) but also minimize false-positives (potentially giving loans to individuals that are high risk of default).

With a strong 99% accuracy rate I would recommend this model. It is able to capture individuals with healthy loan status, while still minimizing mislabelling of those at a high risk of default.










# John


# credit-risk-classification Module 20 Challenge Assignment

## Overview of the Analysis
The purpose of this analysis is to create a model which can correctly predict the incidence of fraudulent loan applications within a population of loan applications.  The model assesses a dataset consisting of the following applicant features or variables when determining the likelihood of a loan being fraudulent:  
- loan size - in dollars
- interest_rate 
- borrower income - annualized in dollars
- current debt to income ratio
- number of accounts 
- the number of derogatory credit bureau marks the applicant has accrued
- total debt in dollars
- current loan status - 0 = healthy loan, 1 = fraudulent loan. this element became the dependent variable of the model used to both train the model and test its accuracy.

## Approach
1. The data was split into independent (X) and dependent (y) subsets.  The dependent set (y) consisted of the current loan status.  The remaining elements were used as independent features for training and testing the model.  
2. Both the independent X and dependent y datasets were split into two additional subsets one for training the model and one for test then model. 75% of the data was used for training and 25% was reserved for testing the final model.
3. In order to remove any imbalance in feature influence in the model, the independent training data was used to create a scaling factor which was applied to both the independent training and independent testing data.
4. After scaling, 75% of the independent variable data was used to train the model, the remaining 25% was used to test the model.
5. The resulting data was then assessed with three modelling approaches:
   - Logistic Regression
   - Support Vector Machine
   - [Decision Tree](https://github.com/john-a-ellis/credit-risk-classification/blob/main/Resources/loans_tree.png)
  

## Results
            LogisticRegression      Support Vector Machine       Decision Tree
`Precision:        .84                         .84                     .84`  
`Recall:           .98                         .98                     .85`  
`f1-score:         .91                         .91                     .85`  
`accuracy:         .99                         .99                     .99`  

                    


## Summary
The three models yielded relatively consistent results, with the Logistic Regression and Support Vector Machine models having identical results, only the decision tree model had a slightly weaker recall ability. Given the results the Logistic Regression model is recommended due to the ease of implementation and interpretation.

## Chosen Model Logistic Regression Model
        	Predicted Healthy-0	Predicted Fraud-1
`Healthy-0	       18652	            113`  
`Fraudulent-1	      10	            609`      
`Accuracy Score : 0.9936545604622369`  

                        Classification Report
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.84      0.98      0.91       619

`    accuracy                           0.99     19384`  
`   macro avg       0.92      0.99      0.95     19384`  
`weighted avg       0.99      0.99      0.99     19384`  

At first glance the model appears to be very good at predicting loan health given the 99% accuracy.  However upon closer look this accuracy appears to be greatly influenced by the in-balance in the tested dataset where just over 3% of the observations were fraudulent.  Looking closer we see the model predicted 722 fraudulent loans, but only 609 of these loans were actually fraudulent, for a precision of 84.3%.  Conversely, the model correctly predicted a loan as being fraudulent 98% of the time, 609/619. As a result, if we were to use the model in a production environment using it to predict if a loan is likely to be fraudulent this is a good model.  The risk of model error lies in falsely identifying fraud when none exists which may happen in 16% of the cases.  But the financial implication of this (ie lost business) is offset by the fact it can correctly identify fraud 98% of the time, reducing the risk of writing loans which are unrecoverable due to fraud.  If attempts were to be made to improve the accuracy of the model for identifying healthy loans, care would have to be taken to ensure not to compromise the ability of the model to identify fraudulent loans thereby increasing
