import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, confusion_matrix
from models import TreatmentModel
from preprocessing import preprocess_data
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from joblib import dump, load
import numpy as np
import pandas as pd
import os

# def build_treatment_simulator(data_path, treatment_decisions, outcome_vars):
#     try:
#         # Preprocess the data
#         balanced_data = preprocess_data(data_path)
        
#         # Create a dictionary to store the SVC models for each outcome variable
#         ts_models = {}
        
#         for outcome in outcome_vars:
#             model_path = f'ts_model_{outcome}.pkl'
            
#             if os.path.exists(model_path):
#                 # Load the trained model if it exists
#                 ts_models[outcome] = load(model_path)
#                 print(f"Loaded trained Treatment Simulator model for {outcome}")
#             else:
#                 print(f"Building Treatment Simulator model for {outcome}")
                
#                 # Define the parameter grid for GridSearchCV
#                 param_grid = {
#                     'C': [0.1, 1, 10],
#                     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#                     'degree': [2, 3, 4],
#                     'gamma': ['scale', 'auto'],
#                     'class_weight': ['balanced', None]
#                 }
                
#                 # Create StratifiedKFold for cross-validation
#                 cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
#                 # Create the SVC model
#                 svc = SVC(probability=True)
                
#                 # Perform grid search with cross-validation
#                 grid_search = GridSearchCV(svc, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=2)
#                 grid_search.fit(balanced_data.drop([outcome], axis=1), balanced_data[outcome])
                
#                 # Get the best model
#                 best_model = grid_search.best_estimator_
#                 best_params = grid_search.best_params_
                
#                 # Print the best hyperparameters for each SVC model
#                 print(f"Best hyperparameters for {outcome}:")
#                 print(best_params)
                
#                 # Save the best model for the current outcome variable
#                 dump(best_model, model_path)
#                 ts_models[outcome] = best_model
        
#         return ts_models
    
#     except ValueError as e:
#         print(f"Error: {e}")
#         print("Possible reasons:")
#         print("1. Mismatch between the input data and the expected format.")
#         print("2. Invalid parameter values in the grid search.")
#     except Exception as e:
#         print(f"Error: An unexpected error occurred during Treatment Simulator building: {e}")
#         raise

# def evaluate_treatment_simulator(data_path, treatment_decisions, ts_models, num_bootstraps=2000):
#     try:
#         # Preprocess the data
#         balanced_data = preprocess_data(data_path)
        
#         # Initialize a dictionary to store the evaluation results
#         evaluation_results = {}
        
#         for outcome, model in ts_models.items():
#             print(f"Evaluating Treatment Simulator model for {outcome}")
            
#             # Perform stratified k-fold cross-validation
#             cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
#             accuracy_scores = []
#             precision_scores = []
#             recall_scores = []
#             f1_scores = []
#             roc_auc_scores = []
            
#             for train_index, test_index in cv.split(balanced_data.drop([outcome], axis=1), balanced_data[outcome]):
#                 X_train, X_test = balanced_data.drop([outcome], axis=1).iloc[train_index], balanced_data.drop([outcome], axis=1).iloc[test_index]
#                 y_train, y_test = balanced_data[outcome].iloc[train_index], balanced_data[outcome].iloc[test_index]
                
#                 # Make predictions using the Treatment Simulator model
#                 y_pred = model.predict(X_test)
#                 y_prob = model.predict_proba(X_test)[:, 1]
                
#                 # Calculate evaluation metrics
#                 accuracy = accuracy_score(y_test, y_pred)
#                 precision = precision_score(y_test, y_pred)
#                 recall = recall_score(y_test, y_pred)
#                 f1 = f1_score(y_test, y_pred)
#                 roc_auc = roc_auc_score(y_test, y_prob)
                
#                 accuracy_scores.append(accuracy)
#                 precision_scores.append(precision)
#                 recall_scores.append(recall)
#                 f1_scores.append(f1)
#                 roc_auc_scores.append(roc_auc)
            
#             # Calculate the mean and 95% confidence interval for each metric
#             mean_accuracy = np.mean(accuracy_scores)
#             mean_precision = np.mean(precision_scores)
#             mean_recall = np.mean(recall_scores)
#             mean_f1 = np.mean(f1_scores)
#             mean_roc_auc = np.mean(roc_auc_scores)
            
#             ci_lower_accuracy = np.percentile(accuracy_scores, 2.5)
#             ci_upper_accuracy = np.percentile(accuracy_scores, 97.5)
#             ci_lower_precision = np.percentile(precision_scores, 2.5)
#             ci_upper_precision = np.percentile(precision_scores, 97.5)
#             ci_lower_recall = np.percentile(recall_scores, 2.5)
#             ci_upper_recall = np.percentile(recall_scores, 97.5)
#             ci_lower_f1 = np.percentile(f1_scores, 2.5)
#             ci_upper_f1 = np.percentile(f1_scores, 97.5)
#             ci_lower_roc_auc = np.percentile(roc_auc_scores, 2.5)
#             ci_upper_roc_auc = np.percentile(roc_auc_scores, 97.5)
            
#             # Print the evaluation results for the current outcome variable
#             print(f"Outcome: {outcome}")
#             print(f"Mean Accuracy: {mean_accuracy:.4f}")
#             print(f"95% CI for Accuracy: [{ci_lower_accuracy:.4f}, {ci_upper_accuracy:.4f}]")
#             print(f"Mean Precision: {mean_precision:.4f}")
#             print(f"95% CI for Precision: [{ci_lower_precision:.4f}, {ci_upper_precision:.4f}]")
#             print(f"Mean Recall: {mean_recall:.4f}")
#             print(f"95% CI for Recall: [{ci_lower_recall:.4f}, {ci_upper_recall:.4f}]")
#             print(f"Mean F1-score: {mean_f1:.4f}")
#             print(f"95% CI for F1-score: [{ci_lower_f1:.4f}, {ci_upper_f1:.4f}]")
#             print(f"Mean ROC AUC: {mean_roc_auc:.4f}")
#             print(f"95% CI for ROC AUC: [{ci_lower_roc_auc:.4f}, {ci_upper_roc_auc:.4f}]")
            
#             # Store the evaluation results
#             evaluation_results[outcome] = {
#                 'Mean Accuracy': mean_accuracy,
#                 '95% CI for Accuracy': (ci_lower_accuracy, ci_upper_accuracy),
#                 'Mean Precision': mean_precision,
#                 '95% CI for Precision': (ci_lower_precision, ci_upper_precision),
#                 'Mean Recall': mean_recall,
#                 '95% CI for Recall': (ci_lower_recall, ci_upper_recall),
#                 'Mean F1-score': mean_f1,
#                 '95% CI for F1-score': (ci_lower_f1, ci_upper_f1),
#                 'Mean ROC AUC': mean_roc_auc,
#                 '95% CI for ROC AUC': (ci_lower_roc_auc, ci_upper_roc_auc)
#             }
        
#         return evaluation_results
    
#     except KeyError as e:
#         print(f"Error: {e}")
#         print("Possible reasons:")
#         print("1. Mismatch between the input data and the expected format.")
#         print("2. Incorrect column names or missing columns.")
#     except Exception as e:
#         print(f"Error: An unexpected error occurred during Treatment Simulator evaluation: {e}")
#         raise

# def evaluate_dql_models(data_path, treatment_decisions):
#     try:
#         # Preprocess the data
#         balanced_data = preprocess_data(data_path)
        
#         # Define input and output sizes based on the dataset
#         input_size = balanced_data.shape[1] - 1
#         output_size = 1
        
#         models = {}
#         for decision in treatment_decisions:
#             model_path = f'model_{decision}.pth'
            
#             if os.path.exists(model_path):
#                 # Load the saved model's state dictionary
#                 saved_state = torch.load(model_path)
#                 state_dict = saved_state['state_dict']
                
#                 # Extract the hidden sizes and dropout from the saved model
#                 hidden_sizes = saved_state['hidden_sizes']
#                 dropout = saved_state['dropout']
                
#                 model = TreatmentModel(input_size, hidden_sizes, output_size, dropout)
#                 model.load_state_dict(state_dict)
#                 models[decision] = model
#             else:
#                 print(f"Trained model not found for {decision}. Skipping evaluation.")
        
#         # Evaluate the models
#         evaluation_results = {}
#         for decision in treatment_decisions:
#             if decision not in models:
#                 continue
            
#             print(f"Evaluating DQL model for {decision}")
            
#             # Prepare the test data
#             X_test = balanced_data.drop([decision], axis=1).values.astype(np.float32)
#             y_test = balanced_data[decision].values
            
#             with torch.no_grad():
#                 X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#                 output = models[decision](X_test_tensor)
#                 y_pred = torch.round(output).squeeze().numpy()
            
#             # Compare DQL model's decisions with physician's decisions
#             physician_decisions = balanced_data[decision].values
#             agreement = accuracy_score(physician_decisions, y_pred)
#             kappa = cohen_kappa_score(physician_decisions, y_pred)
            
#             # Calculate evaluation metrics
#             accuracy = accuracy_score(y_test, y_pred)
#             precision = precision_score(y_test, y_pred)
#             recall = recall_score(y_test, y_pred)
#             f1 = f1_score(y_test, y_pred)
#             roc_auc = roc_auc_score(y_test, output.squeeze().numpy())
            
#             # Calculate confusion matrix
#             cm = confusion_matrix(y_test, y_pred)
            
#             # Store the evaluation results
#             evaluation_results[decision] = {
#                 'Accuracy': accuracy,
#                 'Precision': precision,
#                 'Recall': recall,
#                 'F1-score': f1,
#                 'ROC AUC': roc_auc,
#                 'Agreement': agreement,
#                 'Kappa': kappa,
#                 'Confusion Matrix': cm
#             }
            
#             print(f"Evaluation metrics for {decision}:")
#             print(f"Accuracy: {accuracy:.4f}")
#             print(f"Precision: {precision:.4f}")
#             print(f"Recall: {recall:.4f}")
#             print(f"F1-score: {f1:.4f}")
#             print(f"ROC AUC: {roc_auc:.4f}")
#             print(f"Agreement with physician's decisions: {agreement:.4f}")
#             print(f"Cohen's Kappa: {kappa:.4f}")
#             print(f"Confusion Matrix:\n{cm}")
        
#         return evaluation_results
    
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         print("Possible reasons:")
#         print("1. The trained model files are missing.")
#         print("2. Incorrect file paths or names.")
#         return None
#     except RuntimeError as e:
#         print(f"Error: {e}")
#         print("Possible reasons:")
#         print("1. Mismatch between the input data and the model's input size.")
#         print("2. Insufficient memory to allocate tensors.")
#         return None
#     except Exception as e:
#         print(f"Error: An unexpected error occurred during DQL model evaluation: {e}")
#         return None


import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, confusion_matrix
from models import TreatmentModel
from preprocessing import preprocess_data
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from joblib import dump, load
import numpy as np
import pandas as pd
import os
import warnings

def build_treatment_simulator(data_path, treatment_decisions, outcome_vars):
    try:
        # Preprocess the data
        balanced_data, feature_names = preprocess_data(data_path)
        
        # Create a dictionary to store the SVC models for each outcome variable
        ts_models = {}
        
        for outcome in outcome_vars:
            model_path = f'ts_model_{outcome}.pkl'
            
            if os.path.exists(model_path):
                # Load the trained model if it exists
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    ts_models[outcome] = load(model_path)
                print(f"Loaded trained Treatment Simulator model for {outcome}")
            else:
                print(f"Building Treatment Simulator model for {outcome}")
                
                # Define the parameter grid for GridSearchCV
                param_grid = {
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'degree': [2, 3, 4],
                    'gamma': ['scale', 'auto'],
                    'class_weight': ['balanced', None],
                    'probability': [True]  # Add probability parameter
                }
                
                # Create StratifiedKFold for cross-validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
                # Create the SVC model
                svc = SVC()
                
                # Perform grid search with cross-validation
                grid_search = GridSearchCV(svc, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
                grid_search.fit(balanced_data.drop([outcome], axis=1), balanced_data[outcome])
                
                # Get the best model
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                
                # Print the best hyperparameters for each SVC model
                print(f"Best hyperparameters for {outcome}:")
                print(best_params)
                
                # Save the best model for the current outcome variable
                dump(best_model, model_path)
                ts_models[outcome] = best_model
        
        return ts_models, feature_names
    
    except ValueError as e:
        print(f"Error: {e}")
        print("Possible reasons:")
        print("1. Mismatch between the input data and the expected format.")
        print("2. Invalid parameter values in the grid search.")
        return {}, []
    except Exception as e:
        print(f"Error: An unexpected error occurred during Treatment Simulator building: {e}")
        return {}, []

def evaluate_treatment_simulator(data_path, treatment_decisions, ts_models, feature_names, num_bootstraps=2000):
    try:
        # Preprocess the data
        balanced_data, _, _ = preprocess_data(data_path)
        
        # Initialize a dictionary to store the evaluation results
        evaluation_results = {}
        
        for outcome, model in ts_models.items():
            print(f"Evaluating Treatment Simulator model for {outcome}")
            
            # Perform stratified k-fold cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []
            roc_auc_scores = []
            
            for train_index, test_index in cv.split(balanced_data.drop([outcome], axis=1), balanced_data[outcome]):
                X_train, X_test = balanced_data.drop([outcome], axis=1).iloc[train_index], balanced_data.drop([outcome], axis=1).iloc[test_index]
                y_train, y_test = balanced_data[outcome].iloc[train_index], balanced_data[outcome].iloc[test_index]
                
                # Ensure the feature names match the training data
                X_test = X_test.reindex(columns=feature_names)
                
                # Make predictions using the Treatment Simulator model
                y_pred = model.predict(X_test)
                
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = y_pred
                
                # Calculate evaluation metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=1)
                recall = recall_score(y_test, y_pred, zero_division=1)
                f1 = f1_score(y_test, y_pred, zero_division=1)
                roc_auc = roc_auc_score(y_test, y_prob)
                
                accuracy_scores.append(accuracy)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
                roc_auc_scores.append(roc_auc)
            
            # Calculate the mean and 95% confidence interval for each metric
            mean_accuracy = np.mean(accuracy_scores)
            mean_precision = np.mean(precision_scores)
            mean_recall = np.mean(recall_scores)
            mean_f1 = np.mean(f1_scores)
            mean_roc_auc = np.mean(roc_auc_scores)
            
            ci_lower_accuracy = np.percentile(accuracy_scores, 2.5)
            ci_upper_accuracy = np.percentile(accuracy_scores, 97.5)
            ci_lower_precision = np.percentile(precision_scores, 2.5)
            ci_upper_precision = np.percentile(precision_scores, 97.5)
            ci_lower_recall = np.percentile(recall_scores, 2.5)
            ci_upper_recall = np.percentile(recall_scores, 97.5)
            ci_lower_f1 = np.percentile(f1_scores, 2.5)
            ci_upper_f1 = np.percentile(f1_scores, 97.5)
            ci_lower_roc_auc = np.percentile(roc_auc_scores, 2.5)
            ci_upper_roc_auc = np.percentile(roc_auc_scores, 97.5)
            
            # Print the evaluation results for the current outcome variable
            print(f"Outcome: {outcome}")
            print(f"Mean Accuracy: {mean_accuracy:.4f}")
            print(f"95% CI for Accuracy: [{ci_lower_accuracy:.4f}, {ci_upper_accuracy:.4f}]")
            print(f"Mean Precision: {mean_precision:.4f}")
            print(f"95% CI for Precision: [{ci_lower_precision:.4f}, {ci_upper_precision:.4f}]")
            print(f"Mean Recall: {mean_recall:.4f}")
            print(f"95% CI for Recall: [{ci_lower_recall:.4f}, {ci_upper_recall:.4f}]")
            print(f"Mean F1-score: {mean_f1:.4f}")
            print(f"95% CI for F1-score: [{ci_lower_f1:.4f}, {ci_upper_f1:.4f}]")
            print(f"Mean ROC AUC: {mean_roc_auc:.4f}")
            print(f"95% CI for ROC AUC: [{ci_lower_roc_auc:.4f}, {ci_upper_roc_auc:.4f}]")
            
            # Store the evaluation results
            evaluation_results[outcome] = {
                'Mean Accuracy': mean_accuracy,
                '95% CI for Accuracy': (ci_lower_accuracy, ci_upper_accuracy),
                'Mean Precision': mean_precision,
                '95% CI for Precision': (ci_lower_precision, ci_upper_precision),
                'Mean Recall': mean_recall,
                '95% CI for Recall': (ci_lower_recall, ci_upper_recall),
                'Mean F1-score': mean_f1,
                '95% CI for F1-score': (ci_lower_f1, ci_upper_f1),
                'Mean ROC AUC': mean_roc_auc,
                '95% CI for ROC AUC': (ci_lower_roc_auc, ci_upper_roc_auc)
            }
        
        return evaluation_results
    
    except KeyError as e:
        print(f"Error: {e}")
        print("Possible reasons:")
        print("1. Mismatch between the input data and the expected format.")
        print("2. Incorrect column names or missing columns.")
    except Exception as e:
        print(f"Error: An unexpected error occurred during Treatment Simulator evaluation: {e}")
        raise

def evaluate_dql_models(data_path, treatment_decisions):
    try:
        # Preprocess the data
        balanced_data, _ = preprocess_data(data_path)
        
        # Define input and output sizes based on the dataset
        input_size = balanced_data.shape[1] - 1
        output_size = 1
        
        models = {}
        for decision in treatment_decisions:
            model_path = f'model_{decision}.pth'
            
            if os.path.exists(model_path):
                # Load the saved model's state dictionary
                saved_state = torch.load(model_path)
                state_dict = saved_state['state_dict']
                
                # Extract the hidden sizes and dropout from the saved model
                hidden_sizes = saved_state['hidden_sizes']
                dropout = saved_state['dropout']
                
                model = TreatmentModel(input_size, hidden_sizes, output_size, dropout)
                model.load_state_dict(state_dict)
                models[decision] = model
            else:
                print(f"Trained model not found for {decision}. Skipping evaluation.")
        
        # Evaluate the models
        evaluation_results = {}
        for decision in treatment_decisions:
            if decision not in models:
                continue
            
            print(f"Evaluating DQL model for {decision}")
            
            # Prepare the test data
            X_test = balanced_data.drop([decision], axis=1).values.astype(np.float32)
            y_test = balanced_data[decision].values
            
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                output = models[decision](X_test_tensor)
                y_pred = torch.round(output).squeeze().numpy()
            
            # Compare DQL model's decisions with physician's decisions
            physician_decisions = balanced_data[decision].values
            agreement = accuracy_score(physician_decisions, y_pred)
            kappa = cohen_kappa_score(physician_decisions, y_pred)
            
            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='micro')
            recall = recall_score(y_test, y_pred, average='micro')
            f1 = f1_score(y_test, y_pred, average='micro')
            roc_auc = roc_auc_score(y_test, output.squeeze().numpy(), multi_class='ovr')
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Store the evaluation results
            evaluation_results[decision] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-score': f1,
                'ROC AUC': roc_auc,
                'Agreement': agreement,
                'Kappa': kappa,
                'Confusion Matrix': cm
            }
            
            print(f"Evaluation metrics for {decision}:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-score: {f1:.4f}")
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"Agreement with physician's decisions: {agreement:.4f}")
            print(f"Cohen's Kappa: {kappa:.4f}")
            print(f"Confusion Matrix:\n{cm}")
        
        return evaluation_results
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Possible reasons:")
        print("1. The trained model files are missing.")
        print("2. Incorrect file paths or names.")
        return None
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Possible reasons:")
        print("1. Mismatch between the input data and the model's input size.")
        print("2. Insufficient memory to allocate tensors.")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred during DQL model evaluation: {e}")
        return None

def analyze_feature_importance(X_test, y_test):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_test, y_test)
    feature_importances = model.feature_importances_
    feature_names = X_test.columns.tolist()
    importance_dict = {feature: importance for feature, importance in zip(feature_names, feature_importances) if importance > 0}
    sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_importances

# import torch
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, confusion_matrix
# from models import TreatmentModel
# from preprocessing import preprocess_data
# from sklearn.svm import SVC
# from sklearn.model_selection import StratifiedKFold, GridSearchCV
# from sklearn.ensemble import RandomForestRegressor
# from joblib import dump, load
# import numpy as np
# import pandas as pd
# import os

# import torch
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, cohen_kappa_score, confusion_matrix
# from models import TreatmentModel
# from preprocessing import preprocess_data
# from sklearn.svm import SVC
# from sklearn.model_selection import StratifiedKFold, GridSearchCV
# from joblib import dump, load
# import numpy as np
# import pandas as pd
# import os
# import warnings

# def build_treatment_simulator(data_path, treatment_decisions, outcome_vars):
#     try:
#         # Preprocess the data
#         balanced_data = preprocess_data(data_path)
        
#         # Create a dictionary to store the SVC models for each outcome variable
#         ts_models = {}
        
#         for outcome in outcome_vars:
#             model_path = f'ts_model_{outcome}.pkl'
            
#             if os.path.exists(model_path):
#                 # Load the trained model if it exists
#                 with warnings.catch_warnings():
#                     warnings.simplefilter("ignore", category=UserWarning)
#                     ts_models[outcome] = load(model_path)
#                 print(f"Loaded trained Treatment Simulator model for {outcome}")
#             else:
#                 print(f"Building Treatment Simulator model for {outcome}")
                
#                 # Define the parameter grid for GridSearchCV
#                 param_grid = {
#                     'C': [0.1, 1, 10],
#                     'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#                     'degree': [2, 3, 4],
#                     'gamma': ['scale', 'auto'],
#                     'class_weight': ['balanced', None],
#                     'probability': [True]  # Add probability parameter
#                 }
                
#                 # Create StratifiedKFold for cross-validation
#                 cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                
#                 # Create the SVC model
#                 svc = SVC()
                
#                 # Perform grid search with cross-validation
#                 grid_search = GridSearchCV(svc, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=0)
#                 grid_search.fit(balanced_data.drop([outcome], axis=1), balanced_data[outcome])
                
#                 # Get the best model
#                 best_model = grid_search.best_estimator_
#                 best_params = grid_search.best_params_
                
#                 # Print the best hyperparameters for each SVC model
#                 print(f"Best hyperparameters for {outcome}:")
#                 print(best_params)
                
#                 # Save the best model for the current outcome variable
#                 dump(best_model, model_path)
#                 ts_models[outcome] = best_model
        
#         return ts_models
    
#     except ValueError as e:
#         print(f"Error: {e}")
#         print("Possible reasons:")
#         print("1. Mismatch between the input data and the expected format.")
#         print("2. Invalid parameter values in the grid search.")
#     except Exception as e:
#         print(f"Error: An unexpected error occurred during Treatment Simulator building: {e}")
#         raise

# def evaluate_treatment_simulator(data_path, treatment_decisions, ts_models, num_bootstraps=2000):
#     try:
#         # Preprocess the data
#         balanced_data = preprocess_data(data_path)
        
#         # Initialize a dictionary to store the evaluation results
#         evaluation_results = {}
        
#         for outcome, model in ts_models.items():
#             print(f"Evaluating Treatment Simulator model for {outcome}")
            
#             # Perform stratified k-fold cross-validation
#             cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
#             accuracy_scores = []
#             precision_scores = []
#             recall_scores = []
#             f1_scores = []
#             roc_auc_scores = []
            
#             for train_index, test_index in cv.split(balanced_data.drop([outcome], axis=1), balanced_data[outcome]):
#                 X_train, X_test = balanced_data.drop([outcome], axis=1).iloc[train_index], balanced_data.drop([outcome], axis=1).iloc[test_index]
#                 y_train, y_test = balanced_data[outcome].iloc[train_index], balanced_data[outcome].iloc[test_index]
                
#                 X_test = X_test[X_train.columns]
#                 # Make predictions using the Treatment Simulator model
#                 y_pred = model.predict(X_test)
                
#                 if hasattr(model, 'predict_proba'):
#                     y_prob = model.predict_proba(X_test)[:, 1]
#                 else:
#                     y_prob = y_pred
                
#                 # Calculate evaluation metrics
#                 accuracy = accuracy_score(y_test, y_pred)
#                 precision = precision_score(y_test, y_pred, zero_division=1)
#                 recall = recall_score(y_test, y_pred, zero_division=1)
#                 f1 = f1_score(y_test, y_pred, zero_division=1)
#                 roc_auc = roc_auc_score(y_test, y_prob)
                
#                 accuracy_scores.append(accuracy)
#                 precision_scores.append(precision)
#                 recall_scores.append(recall)
#                 f1_scores.append(f1)
#                 roc_auc_scores.append(roc_auc)
            
#             # Calculate the mean and 95% confidence interval for each metric
#             mean_accuracy = np.mean(accuracy_scores)
#             mean_precision = np.mean(precision_scores)
#             mean_recall = np.mean(recall_scores)
#             mean_f1 = np.mean(f1_scores)
#             mean_roc_auc = np.mean(roc_auc_scores)
            
#             ci_lower_accuracy = np.percentile(accuracy_scores, 2.5)
#             ci_upper_accuracy = np.percentile(accuracy_scores, 97.5)
#             ci_lower_precision = np.percentile(precision_scores, 2.5)
#             ci_upper_precision = np.percentile(precision_scores, 97.5)
#             ci_lower_recall = np.percentile(recall_scores, 2.5)
#             ci_upper_recall = np.percentile(recall_scores, 97.5)
#             ci_lower_f1 = np.percentile(f1_scores, 2.5)
#             ci_upper_f1 = np.percentile(f1_scores, 97.5)
#             ci_lower_roc_auc = np.percentile(roc_auc_scores, 2.5)
#             ci_upper_roc_auc = np.percentile(roc_auc_scores, 97.5)
            
#             # Print the evaluation results for the current outcome variable
#             print(f"Outcome: {outcome}")
#             print(f"Mean Accuracy: {mean_accuracy:.4f}")
#             print(f"95% CI for Accuracy: [{ci_lower_accuracy:.4f}, {ci_upper_accuracy:.4f}]")
#             print(f"Mean Precision: {mean_precision:.4f}")
#             print(f"95% CI for Precision: [{ci_lower_precision:.4f}, {ci_upper_precision:.4f}]")
#             print(f"Mean Recall: {mean_recall:.4f}")
#             print(f"95% CI for Recall: [{ci_lower_recall:.4f}, {ci_upper_recall:.4f}]")
#             print(f"Mean F1-score: {mean_f1:.4f}")
#             print(f"95% CI for F1-score: [{ci_lower_f1:.4f}, {ci_upper_f1:.4f}]")
#             print(f"Mean ROC AUC: {mean_roc_auc:.4f}")
#             print(f"95% CI for ROC AUC: [{ci_lower_roc_auc:.4f}, {ci_upper_roc_auc:.4f}]")
            
#             # Store the evaluation results
#             evaluation_results[outcome] = {
#                 'Mean Accuracy': mean_accuracy,
#                 '95% CI for Accuracy': (ci_lower_accuracy, ci_upper_accuracy),
#                 'Mean Precision': mean_precision,
#                 '95% CI for Precision': (ci_lower_precision, ci_upper_precision),
#                 'Mean Recall': mean_recall,
#                 '95% CI for Recall': (ci_lower_recall, ci_upper_recall),
#                 'Mean F1-score': mean_f1,
#                 '95% CI for F1-score': (ci_lower_f1, ci_upper_f1),
#                 'Mean ROC AUC': mean_roc_auc,
#                 '95% CI for ROC AUC': (ci_lower_roc_auc, ci_upper_roc_auc)
#             }
        
#         return evaluation_results
    
#     except KeyError as e:
#         print(f"Error: {e}")
#         print("Possible reasons:")
#         print("1. Mismatch between the input data and the expected format.")
#         print("2. Incorrect column names or missing columns.")
#     except Exception as e:
#         print(f"Error: An unexpected error occurred during Treatment Simulator evaluation: {e}")
#         raise

# def evaluate_dql_models(data_path, treatment_decisions):
#     try:
#         # Preprocess the data
#         balanced_data = preprocess_data(data_path)
        
#         # Define input and output sizes based on the dataset
#         input_size = balanced_data.shape[1] - 1
#         output_size = 1
        
#         models = {}
#         for decision in treatment_decisions:
#             model_path = f'model_{decision}.pth'
            
#             if os.path.exists(model_path):
#                 # Load the saved model's state dictionary
#                 saved_state = torch.load(model_path)
#                 state_dict = saved_state['state_dict']
                
#                 # Extract the hidden sizes and dropout from the saved model
#                 hidden_sizes = saved_state['hidden_sizes']
#                 dropout = saved_state['dropout']
                
#                 model = TreatmentModel(input_size, hidden_sizes, output_size, dropout)
#                 model.load_state_dict(state_dict)
#                 models[decision] = model
#             else:
#                 print(f"Trained model not found for {decision}. Skipping evaluation.")
        
#         # Evaluate the models
#         evaluation_results = {}
#         for decision in treatment_decisions:
#             if decision not in models:
#                 continue
            
#             print(f"Evaluating DQL model for {decision}")
            
#             # Prepare the test data
#             X_test = balanced_data.drop([decision], axis=1).values.astype(np.float32)
#             y_test = balanced_data[decision].values
            
#             with torch.no_grad():
#                 X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#                 output = models[decision](X_test_tensor)
#                 y_pred = torch.round(output).squeeze().numpy()
            
#             # Compare DQL model's decisions with physician's decisions
#             physician_decisions = balanced_data[decision].values
#             agreement = accuracy_score(physician_decisions, y_pred)
#             kappa = cohen_kappa_score(physician_decisions, y_pred)
            
#             # Calculate evaluation metrics
#             accuracy = accuracy_score(y_test, y_pred)
#             precision = precision_score(y_test, y_pred, average='micro')
#             recall = recall_score(y_test, y_pred, average='micro')
#             f1 = f1_score(y_test, y_pred, average='micro')
#             roc_auc = roc_auc_score(y_test, output.squeeze().numpy(), multi_class='ovr')
            
#             # Calculate confusion matrix
#             cm = confusion_matrix(y_test, y_pred)
            
#             # Store the evaluation results
#             evaluation_results[decision] = {
#                 'Accuracy': accuracy,
#                 'Precision': precision,
#                 'Recall': recall,
#                 'F1-score': f1,
#                 'ROC AUC': roc_auc,
#                 'Agreement': agreement,
#                 'Kappa': kappa,
#                 'Confusion Matrix': cm
#             }
            
#             print(f"Evaluation metrics for {decision}:")
#             print(f"Accuracy: {accuracy:.4f}")
#             print(f"Precision: {precision:.4f}")
#             print(f"Recall: {recall:.4f}")
#             print(f"F1-score: {f1:.4f}")
#             print(f"ROC AUC: {roc_auc:.4f}")
#             print(f"Agreement with physician's decisions: {agreement:.4f}")
#             print(f"Cohen's Kappa: {kappa:.4f}")
#             print(f"Confusion Matrix:\n{cm}")
        
#         return evaluation_results
    
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         print("Possible reasons:")
#         print("1. The trained model files are missing.")
#         print("2. Incorrect file paths or names.")
#         return None
#     except RuntimeError as e:
#         print(f"Error: {e}")
#         print("Possible reasons:")
#         print("1. Mismatch between the input data and the model's input size.")
#         print("2. Insufficient memory to allocate tensors.")
#         return None
#     except Exception as e:
#         print(f"Error: An unexpected error occurred during DQL model evaluation: {e}")
#         return None

# def analyze_feature_importance(X_test, y_test):
#     model = RandomForestRegressor(random_state=42)
#     model.fit(X_test, y_test)
#     feature_importances = model.feature_importances_
#     feature_names = X_test.columns.tolist()
#     importance_dict = {feature: importance for feature, importance in zip(feature_names, feature_importances) if importance > 0}
#     sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
#     return sorted_importances

# if __name__ == "__main__":
#     data_path = "synthetic_dataset.csv"
#     treatment_decisions = ['surgery', 'chemotherapy', 'hormone_therapy', 'radiotherapy']
#     outcome_vars = ['OS', 'DFS', 'RFS']
#     num_epochs = 300
#     sample_size = 1200
#     param_space = {
#         'hidden_sizes': [(64, 32), (128, 64), (256, 128), (512, 256)],
#         'learning_rate': [0.0001, 0.001, 0.01, 0.1],
#         'batch_size': [16, 32, 64, 128],
#         'dropout': [0.1, 0.2, 0.3, 0.4],
#         'l2_reg': [0.0001, 0.001, 0.01, 0.1]
#     }
