# product-purchase-prediction

## 📋 Project Overview
This project implements a **Logistic Regression**,**K-Nearest Neighbors (K-NN)** plus a **Support Vector Machine (SVM) classifier** model for a binary classification task using the `Social_Network_Ads.csv` dataset. The goal is to predict whether a user will purchase a product based on their age and estimated salary.

---

## 📊 Dataset
- **File**: `Social_Network_Ads.csv`
- **Features**:
  - `Age`: Age of the user
  - `EstimatedSalary`: Estimated salary of the user
- **Target**:
  - `Purchased`: Binary output (0 = Not Purchased, 1 = Purchased)

---

## 🛠️ Steps Performed

### 1. Importing Libraries
- `numpy`
- `matplotlib.pyplot`
- `pandas`

### 2. Loading the Dataset
- Loaded the dataset using `pandas` and displayed basic info.

### 3. Splitting the Dataset
- Split the data into training and test sets using `train_test_split` from `sklearn`.

### 4. Feature Scaling
- Applied `StandardScaler` to normalize the features.
- Only applied to `Age` and `EstimatedSalary`; the target variable `Purchased` was not scaled.

### 5. Training the Model
- Used `LogisticRegression` from `sklearn.linear_model`.
- The K-NN classifier is trained on the training set with 5 neighbors and the Minkowski distance metric.
- The SVM is trained with the `linear` kernel.
- Trained the model on the scaled training data.

### 6. Making Predictions
- Predicted on both a single sample and the entire test set.
- Compared predictions with actual values.

### 7. Evaluation
- Generated a **confusion matrix** and **accuracy score**.
- Visualized training and test results using contour plots.

---

## 📈 Visualizations
- **Confusion Matrix**: Displayed using `ConfusionMatrixDisplay`.
- **Decision Boundary**: Plotted for both training and test sets using `contourf` and scatter plots for the logistic regression model.

---

## 📁 Files
- `Jozveh_Logistic_regression(classification).ipynb`: Main notebook containing all code and outputs for logistic regression.
- `Jozveh_k_nearest_neighbors.ipynb`: Main notebook containing all code and outputs for KNN.
- `Jozveh_Support_Vector_Machine_Classification`: Codes for the SVM solution.
- `Social_Network_Ads.csv`: dataset file

---

## 🧠 Key Concepts
- **Logistic Regression**: A linear model for binary classification.
- **K-Nearest Neighbors (K-NN)**: A non-parametric, instance-based learning algorithm that classifies data points based on the majority class among their k-nearest neighbors.
- **SVM**: A powerful supervised learning algorithm that finds the optimal hyperplane to separate classes by maximizing the margin between them.
- **Feature Scaling**: Standardization of features to improve model performance.
- **Train-Test Split**: Dividing data into training and testing subsets.
- **Confusion Matrix**: A table to evaluate classification performance.
- **Decision Boundary**: Visual representation of the model's classification regions.

---

## 📌 Note
The visualization code for decision boundaries is included but may not be directly applicable to datasets with more than two features. It is provided for educational purposes.
