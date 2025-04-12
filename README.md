# Ensemble Learning Project

This project explores a variety of **ensemble learning methods** for binary classification on imbalanced datasets, with a special focus on performance comparison, sampling strategies, and domain adaptation.


## 📌 Objectives

- Compare ensemble methods: **Bagging**, **Boosting**, **Random Forest**, **XGBoost**, and **Stacking**
- Apply sampling techniques: **Oversampling**, **Undersampling**, and **Cost-sensitive learning**
- Evaluate models using imbalanced-aware metrics: **F1 Score**, **Precision**, **Recall**, and **Balanced Accuracy**
- Perform **domain adaptation** using **pseudo-labeling**

## 📊 Dataset Summary

| Dataset   | Train Size | Test Size | Features | Class 1 Ratio |
|-----------|------------|-----------|----------|----------------|
| Dataset 0 | 41,058     | 13,686    | 51       | 10.00%         |
| Dataset 1 | 41,058     | 13,686    | 51       | 9.95%          |
| Dataset 2 | 41,058     | 13,686    | 51       | 9.83%          |
| Dataset 3 | 41,058     | 13,686    | 51       | 9.81%          |

> The datasets are highly imbalanced and likely originate from a fraud detection problem.

## 🧠 Models & Techniques

- **Penalized Logistic Regression**
- **Simple Decision Tree**
- **Bagging**
- **Random Forest**
- **Gradient Boosting**
- **XGBoost**
- **Stacking**

Each model is evaluated with and without sampling techniques. Hyperparameters are optimized using **Grid Search** and **3-Fold Cross-Validation**.

## ⚙️ Domain Adaptation

Domain adaptation was done using **pseudo-labeling** with an XGBoost model and evaluated using the **Maximum Mean Discrepancy (MMD)** to assess source-target similarity.

## 🧪 Running the Code

1. **Clone the repo**  
   ```bash
   git clone https://github.com/MedAliAdlouni/ensemble_learning_project
   cd ENSEMBLE_LEARNING_PROJECT
    ```

2. **Create and activate a virtual environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    ```
3. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the notebooks or scripts**

    - `main.ipynb` to run baseline experiments

    - `testing_results.ipynb` to analyze baseline experiments outputs

    - `domain_adaptation.ipynb` for adaptation tasks


5. **Open the project report**
    - Open `project_report.pdf`
