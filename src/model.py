import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE  # Handling imbalance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone
import itertools
import time
from xgboost import XGBClassifier  # XGBoost
from sklearn.svm import LinearSVC  # Lighter SVM
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Set random seed for reproducibility
seed = 55


def get_models(y, sampling='none'):
    if sampling == 'cost_sensitive':
        models = {
            'Simple Decision Tree': DecisionTreeClassifier(class_weight='balanced'),
            'Penalized Logistic Regression': LogisticRegression(class_weight='balanced'),
            'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(class_weight='balanced'), random_state=seed),
            'Random Forest': RandomForestClassifier(class_weight='balanced'),
            'Boosting': GradientBoostingClassifier(max_depth=3),
            'XGBoost': XGBClassifier(eval_metric='logloss', max_depth=3, 
                                     scale_pos_weight=(sum(y == 0) / sum(y == 1))),
            'Stacking': StackingClassifier(estimators=[
                ('Simple Decision Tree', DecisionTreeClassifier(class_weight='balanced')),
                ('XGBoost', XGBClassifier(scale_pos_weight=(sum(y == 0) / sum(y == 1)))),
                ('linear_svm', LinearSVC())
            ], final_estimator=LogisticRegression(class_weight='balanced'))
        }
    else:
        models = {
            'Simple Decision Tree': DecisionTreeClassifier(),
            'Penalized Logistic Regression': LogisticRegression(),
            'Bagging': BaggingClassifier(estimator=DecisionTreeClassifier(), random_state=seed),
            'Random Forest': RandomForestClassifier(),
            'Boosting': GradientBoostingClassifier(max_depth=3),
            'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=3),
            'Stacking': StackingClassifier(estimators=[
                ('log_reg', LogisticRegression()),
                ('XGBoost', XGBClassifier()),
                ('linear_svm', LinearSVC())
            ], final_estimator=LogisticRegression())
        }
    
    return models

def get_hyperparameters():
    """Return dictionary of hyperparameters to tune for each model."""
    hyperparams = {
        'Simple Decision Tree': {'max_depth': [None, 10, 20]},  # Baseline model for comparison
        'Penalized Logistic Regression': {'C': [20, 30, 40, 50], 'max_iter' : [10000,]},
        'Bagging': {'n_estimators': [10, 50, 100]},
        'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
        'Boosting': {'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [50, 100], 'max_depth': [3]},  # Fix depth for Boosting
        'XGBoost': {'learning_rate': [0.01, 0.1], 'n_estimators': [50, 100], 'max_depth': [3]},  # Hyperparams for XGBoost
        'Stacking': {}  # No hyperparameter tuning for stacking in this example
    }
    return hyperparams

def preprocess_data(X, y, sampling='none'):

    if sampling == 'oversampling':
        # Apply SMOTE (oversampling)
        smote = SMOTE(random_state=seed)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print(f"Data shape after SMOTE (Oversampling): {X_resampled.shape}")
    
    elif sampling == 'undersampling':
        # Apply RandomUnderSampler (undersampling)
        undersampler = RandomUnderSampler(random_state=seed)
        X_resampled, y_resampled = undersampler.fit_resample(X, y)
        print(f"Data shape after Random Undersampling: {X_resampled.shape}")
    
    else:
        # No sampling, return original data
        X_resampled, y_resampled = X, y
        print(f"Data shape (No Sampling): {X_resampled.shape}")
    
    return X_resampled, y_resampled

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model with 4 performance metrics: F1, Precision, Recall, Accuracy."""
    print("Fitting the model...")
    model.fit(X_train, y_train)
    print("Model fit complete. Predicting...")
    y_pred = model.predict(X_test)
    
    # Calculate the performance metrics
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = balanced_accuracy_score(y_test, y_pred)
    
    # Print the scores
    print(f"F1 Score: {f1}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"Balanced accuracy: {accuracy}")
    
    # Return the results as a dictionary
    return {
        'F1 Score': f1,
        'Precision': precision,
        'Recall': recall,
        'Balanced accuracy': accuracy
    }

# Function to apply model and capture results
def run_model_on_dataset(model, dataset, i):
    print(f"Processing dataset {i+1} with sampling: {dataset['sampling']}, model: {model}")
    
    # Extract train and test sets for the current dataset
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    sampling_method = dataset['sampling']
    
    # Call the apply_algo function and capture the results
    best_params, test_scores, elapsed_time = apply_algo(
        model, X_train, y_train, X_test, y_test, sampling=sampling_method)
    
    # Return the results as a dictionary
    return {
        'dataset': i+1,
        'sampling': sampling_method,
        'model': model,
        'best_params': best_params,
        'F1 Score': test_scores['F1 Score'],
        'Precision': test_scores['Precision'],
        'Recall': test_scores['Recall'],
        'Balanced accuracy': test_scores['Balanced accuracy'],
        'elapsed_time': elapsed_time
    }

def apply_algo(algo_name, X, y, X_test, y_test, random_state= 1, validation_size=0.1, sampling="oversampling"):
    start_time = time.time()  # Start the timer
    models, hyperparams = get_models(y, sampling=sampling), get_hyperparameters()

    # Handle imbalance using SMOTE
    X, y = preprocess_data(X, y, sampling)
    print("Data preprocessing completed.")
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_size, random_state=random_state)
    print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}")
        
    model = models[algo_name]
    param_grid = list(itertools.product(*hyperparams.get(algo_name, {}).values())) if algo_name in hyperparams else [{}]
    best_model, best_params, best_score = None, None, 0

    print("Starting hyperparameter tuning...")
    for params in param_grid:
        param_dict = dict(zip(hyperparams[algo_name].keys(), params))
        tuned_model = clone(model).set_params(**param_dict)
        
        print(f"Evaluating with hyperparameters: {param_dict}")
        mean_cv_score = cross_validate_model(tuned_model, X_val, y_val)
        print(f"Mean CV F1 Score: {mean_cv_score}")

        if mean_cv_score > best_score:
            best_score = mean_cv_score
            best_params = param_dict
            best_model = clone(tuned_model)
            #print(f"New best model {best_model} found with hyperparameters {best_params} and score: {best_score}")

    # Final evaluation on test set
    print("Final evaluation on test set...")
    test_score = evaluate_model(best_model, X_train, y_train, X_test, y_test)
    
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time  # Calculate the elapsed time
    
    print(f"Best model: {best_model}, Test performance scores: {test_score} and elapsed time : {elapsed_time}")

    return best_params, test_score, elapsed_time

def cross_validate_model(model, X, y, splits=3):
    """Perform stratified KFold cross-validation and return F1-scores."""
    skf = StratifiedKFold(n_splits=splits)
    scores = []
    print(f"Performing {splits}-fold cross-validation...")
    for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
        
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        scaler = StandardScaler()  
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)

        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)
        fold_score = f1_score(y_test_fold, y_pred_fold)
        scores.append(fold_score)
    
    return np.mean(scores)

