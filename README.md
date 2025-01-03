# Titanic-EDA-and-Modeling

This project aims to predict the survival of passengers on the Titanic using machine learning. The dataset used for this analysis contains information about passengers on the Titanic, such as their age, sex, class, and other features. The goal is to analyze the data, perform exploratory data analysis (EDA), and build a machine learning model to predict survival.

## Table of Contents

- [Introduction](#introduction)
- [Data Description](#data-description)
- [Technologies Used](#technologies-used)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Machine Learning Model](#machine-learning-model)
- [Model Evaluation](#model-evaluation)
- [How to Run the Project](#how-to-run-the-project)
- [License](#license)

## Introduction

In this project, we explore the Titanic dataset and perform an exploratory data analysis (EDA). After cleaning and preprocessing the data, a machine learning model (Logistic Regression) is trained to predict whether a passenger survived or not. The model's accuracy is then evaluated.

## Data Description

The dataset used for this project contains the following columns:

- **PassengerId**: The ID of the passenger.
- **Pclass**: The class of the ticket the passenger bought (1 = 1st class, 2 = 2nd class, 3 = 3rd class).
- **Name**: The name of the passenger.
- **Sex**: The gender of the passenger (male or female).
- **Age**: The age of the passenger.
- **SibSp**: The number of siblings or spouses the passenger had aboard the Titanic.
- **Parch**: The number of parents or children the passenger had aboard the Titanic.
- **Fare**: The fare the passenger paid for the ticket.
- **Embarked**: The port where the passenger boarded the Titanic (C = Cherbourg; Q = Queenstown; S = Southampton).
- **Survived**: Whether the passenger survived (1 = survived, 0 = did not survive).

## Technologies Used

- **Python**: Programming language used for data analysis and modeling.
- **pandas**: Data manipulation and analysis library.
- **matplotlib**: Library for data visualization.
- **seaborn**: Data visualization library for statistical graphics.
- **scikit-learn**: Machine learning library for creating and evaluating the model.

## Exploratory Data Analysis

In the EDA phase, we perform the following:

1. **Data Overview**: Load the Titanic dataset and display basic information.
2. **Missing Data**: Identify missing values and fill them with appropriate values.
3. **Survival Analysis**: Visualize survival rates based on different features such as gender, class, age, and fare.
4. **Correlations**: Analyze correlations between numerical features.
5. **Visualizations**: Create plots to better understand the relationships between features and survival.

## Machine Learning Model

A **Logistic Regression** model is used to predict the survival of passengers. The model is trained on the dataset and evaluated based on its accuracy.

### Model Steps:

1. **Data Preprocessing**: Clean and prepare the data (e.g., handle missing values, encode categorical features).
2. **Feature Selection**: Select the relevant features for the model.
3. **Model Training**: Train the Logistic Regression model using the training data.
4. **Model Evaluation**: Evaluate the model's performance using accuracy score on the test set.

## Model Evaluation

The accuracy of the Logistic Regression model is calculated using `accuracy_score`. The final model's accuracy is printed after making predictions on the test data.

## How to Run the Project

1. Clone the repository to your local machine.
2. Make sure you have Python installed on your system.
3. Install the required libraries by running the following command:

```bash
pip install -r requirements.txt
