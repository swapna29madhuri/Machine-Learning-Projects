Diabetes Prediction Using K-Nearest Neighbors (KNN)

Project Overview

This project implements a machine learning model to predict whether a patient is diabetic or not based on medical diagnostic data. The dataset used is the "diabetes.csv" file, and the K-Nearest Neighbors (KNN) algorithm is applied to build the classification model.

Dataset

The dataset contains the following columns:

Pregnancies: Number of pregnancies.

Glucose: Plasma glucose concentration.

BloodPressure: Diastolic blood pressure (mm Hg).

SkinThickness: Triceps skinfold thickness (mm).

Insulin: 2-Hour serum insulin (µU/ml).

BMI: Body mass index (weight in kg/(height in m)^2).

DiabetesPedigreeFunction: Diabetes pedigree function (genetic factor).

Age: Age of the patient.

Outcome: Target variable (0 = non-diabetic, 1 = diabetic).

Project Workflow

1. Data Exploration and Visualization

Imported the dataset using Pandas.

Displayed basic dataset statistics using .shape and .describe().

Visualized the distribution of the target variable (Outcome) using a count plot.

Created a correlation heatmap to analyze the relationships between features.

2. Data Preprocessing

Checked for missing values and ensured data integrity.

Split the dataset into features (x) and target variable (y).

Performed a train-test split (80%-20%) to prepare for model evaluation.

Scaled the features using StandardScaler to normalize the data.

3. Model Building: K-Nearest Neighbors (KNN)

Used the KNeighborsClassifier from scikit-learn.

Configured the model with:

n_neighbors=25 (number of neighbors).

metric='minkowski' (distance metric).

Trained the KNN model on the training set and made predictions on the test set.

4. Model Evaluation

Evaluated the model using:

Confusion Matrix: Visualized using a heatmap.

Accuracy Score: Measured the prediction accuracy on the test set.

5. Model Saving

Saved the trained KNN model using Python's pickle module to ensure reusability.

Saved the StandardScaler object for consistent scaling during future predictions.


Model Performance

The KNN model achieved the following metrics:

Accuracy: 79%.

Future Enhancements

Feature Engineering: Explore additional features to improve predictions.

Hyperparameter Tuning: Optimize KNN parameters for better performance.

Deployment: Develop a web application using Flask or FastAPI for real-time predictions.


Acknowledgments

Dataset sourced from Pima Indians Diabetes Dataset.

Thanks to the scikit-learn library for providing robust machine learning tools.
