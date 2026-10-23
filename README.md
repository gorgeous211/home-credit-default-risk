# Home Credit Default Risk Prediction

## 1. Project Overview

This project aims to predict the probability of loan default using applicant information and historical credit-related features.

The goal is to build a machine learning model that can distinguish between applicants with different levels of default risk and evaluate its predictive performance.

This project is based on the [Home Credit Default Risk competition](https://www.kaggle.com/competitions/home-credit-default-risk) on Kaggle.

## 2. Problem Statement

Loan default prediction is a binary classification problem.

- **Target = 0:** The applicant did not experience the target-defined default event.

- **Target = 1:** The applicant experienced the target-defined default event.

The main objective is to predict the probability of default for each applicant.

Since the target classes are imbalanced, ROC-AUC will be used as the primary evaluation metric.

## 3. Dataset

The project uses data provided by the Home Credit Default Risk competition.

The main datasets include:

- `application_train.csv`: Applicant information and target labels.

- `application_test.csv`: Applicant information for final predictions.

- `bureau.csv`: Historical credit information from other financial institutions.

- `previous_application.csv`: Previous loan application records.

- `installments_payments.csv`: Historical installment payment records.

- `credit_card_balance.csv`: Historical credit card balance records.

- `POS_CASH_balance.csv`: Historical POS and cash loan records.

The original datasets are not included in this repository. Please obtain the data from Kaggle and follow its applicable data-use rules.
