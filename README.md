# product-purchase-prediction

## 📋 Project Overview
This project implements a **Logistic Regression** and a **K-Nearest Neighbors (K-NN)** model for a binary classification task using the `Social_Network_Ads.csv` dataset. The goal is to predict whether a user will purchase a product based on their age and estimated salary.

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
- Used a 75–25 split (300 training, 100 test samples).

### 4. Feature Scaling
- Applied `StandardScaler` to normalize the features.
- Only applied to `Age` and `EstimatedSalary`; the target variable `Purchased` was not scaled.

### 5. Training the Model
- Used `LogisticRegression` from `sklearn.linear_model`.
- The K-NN classifier is trained on the training set with 5 neighbors and the Minkowski distance metric. 
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
- `Social_Network_Ads.csv`: dataset file

---

## 🧠 Key Concepts
- **Logistic Regression**: A linear model for binary classification.
- **K-Nearest Neighbors (K-NN)**: A non-parametric, instance-based learning algorithm that classifies data points based on the majority class among their k-nearest neighbors.
- **Feature Scaling**: Standardization of features to improve model performance.
- **Train-Test Split**: Dividing data into training and testing subsets.
- **Confusion Matrix**: A table to evaluate classification performance.
- **Decision Boundary**: Visual representation of the model's classification regions.

---

## 📌 Note
The visualization code for decision boundaries is included but may not be directly applicable to datasets with more than two features. It is provided for educational purposes.
