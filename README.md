# Health Insurance Premium Prediction API

## Project Overview

This project predicts health insurance premiums based on customer information such as age, sex, BMI, number of children, smoking status, and region.

A Machine Learning regression model is trained using the insurance dataset and deployed as a REST API using FastAPI.

---

## Problem Statement

Insurance companies need to estimate the medical insurance premium for customers based on their personal and demographic information.

The goal of this project is to build a Machine Learning model that predicts the insurance premium (`charges`) and expose the model through a FastAPI REST API.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Pydantic
- Uvicorn
- Joblib
- VS Code

---

## Dataset

The dataset contains the following columns:

| Column | Description |
|---|---|
| age | Age of the customer |
| sex | Gender of the customer |
| bmi | Body Mass Index |
| children | Number of children |
| smoker | Smoking status |
| region | Residential region |
| charges | Insurance premium |

Target variable:

```text
charges