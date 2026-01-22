# Stacking Ensemble Learning Implementation (K-Fold Blending)

This repository contains my submission for the task **“Stacking Ensemble Learning Implementation”**.

In this project, I built a **stacking ensemble model** using multiple base classifiers and a meta-classifier.  
To make the stacking process reliable, I used **K-Fold cross-validation blending (Out-of-Fold predictions)** to generate the meta-features.

---

##  What I Implemented (Task Requirements)
1. Used **3 different types of base models** (linear, tree-based, and SVM)  
2. Implemented **K-Fold blending** to create meta-level input features  
3. Trained a **meta-classifier** to make the final prediction  
4. Compared **stacking performance vs individual models** using:
- Accuracy
- F1 Score  
5. Added visual outputs:
- Result comparison table
- Confusion Matrix

---

##  Models Used

### Base Learners
1. **Logistic Regression** – simple and fast linear model  
2. **Random Forest** – tree-based ensemble model  
3. **SVM (RBF Kernel)** – powerful non-linear classifier  

### Meta Learner
- **Logistic Regression** (used as final classifier)

---

##  Dataset Used
For this task, I used the **Breast Cancer Wisconsin dataset**, which is already available in scikit-learn.

- Dataset source: `sklearn.datasets`
- Type: Binary classification (Benign vs Malignant)

---

##  How Stacking Works in This Project (K-Fold Blending)

To implement stacking properly, I followed these steps:

1. Split the training data into **K folds** using `StratifiedKFold`
2. Train each base model using **K-1 folds**
3. Predict on the remaining fold to get **Out-of-Fold (OOF) predictions**
4. Use those OOF predictions as **meta-features**
5. Train the meta model on these meta-features
6. Make final predictions on the test data

This approach avoids data leakage and gives a realistic stacking implementation.

---

##  Results

The final output contains a performance comparison between all base models and the stacking ensemble.

 Sample results from my run:

- **Logistic Regression Accuracy:** ~0.98  
- **SVM Accuracy:** ~0.98  
- **Stacking Ensemble Accuracy:** ~0.97  
- **Random Forest Accuracy:** ~0.94  

 Note: Since stacking depends on folds and probabilities, its performance may be slightly higher/lower across runs. However, the implementation correctly demonstrates the stacking ensemble pipeline with cross-validated blending.

---

##  Screenshots

All screenshots are available inside the `screenshots/` folder, including:

- Confusion matrix output
- Accuracy & F1 comparison table
- Base learner results

---

##  How to Run This Project

###  Option 1: Run in Google Colab
1. Open the notebook: `stacking_ensemble_colab.ipynb`
2. Run the cells one by one

###  Option 2: Run Locally
First install dependencies:

```bash
pip install -r requirements.txt
