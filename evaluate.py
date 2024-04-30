# import torch
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from models import TreatmentModel
# from preprocessing import preprocess_data

# def evaluate_models(data_path, treatment_decisions, hidden_size):
#     # Preprocess the data
#     balanced_data = preprocess_data(data_path)

#     # Load the trained models
#     models = {decision: TreatmentModel(input_size, hidden_size, output_size) for decision in treatment_decisions}
#     for decision in treatment_decisions:
#         models[decision].load_state_dict(torch.load(f'model_{decision}.pth'))

#     # Evaluate the models
#     for decision in treatment_decisions:
#         X_test = torch.tensor(balanced_data[decision].drop([decision], axis=1).values, dtype=torch.float32)
#         y_test = balanced_data[decision][decision].values

#         with torch.no_grad():
#             y_pred = models[decision](X_test).numpy()

#         accuracy = accuracy_score(y_test, y_pred.round())
#         precision = precision_score(y_test, y_pred.round())
#         recall = recall_score(y_test, y_pred.round())
#         f1 = f1_score(y_test, y_pred.round())

#         print(f"Evaluation metrics for {decision}:")
#         print(f"Accuracy: {accuracy:.4f}")
#         print(f"Precision: {precision:.4f}")
#         print(f"Recall: {recall:.4f}")
#         print(f"F1-score: {f1:.4f}")
    
    
# if __name__ == "__main__":
#     data_path = "dataset.csv"
#     treatment_decisions = ['surgery', 'chemotherapy', 'hormone_therapy', 'radiotherapy']
#     hidden_size = 50 
#     evaluate_models(data_path, treatment_decisions, hidden_size)


# evaluate.py version 1.0

# import torch
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from models import TreatmentModel
# from preprocessing import preprocess_data
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
# from joblib import dump, load
# import numpy as np
# import pandas as pd
# import os
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
# from sklearn.ensemble import RandomForestClassifier

# # import eli5
# # from eli5.sklearn import PermutationImportance


# def build_treatment_simulator(data_path, treatment_decisions, outcome_vars):
#     try:
#         # Preprocess the data
#         balanced_data = preprocess_data(data_path)

#         # Create a dictionary to store the SVC models for each outcome variable
#         ts_models = {}

#         for outcome in outcome_vars:
#             model_path = f'ts_model_{outcome}.pkl'
#             if os.path.exists(model_path):
#             # Load the trained model if it exists
#                 ts_models[outcome] = load(model_path)
#                 print(f"Loaded trained Treatment Simulator model for {outcome}")
#         else:
#             print(f"Building Treatment Simulator model for {outcome}")

#             # Define the parameter grid for GridSearchCV
#             param_grid = {
#                 'C': [0.1, 1, 10],
#                 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#                 'degree': [2, 3, 4],
#                 'gamma': ['scale', 'auto'],
#                 'class_weight': ['balanced', None]
#             }

#             # Create StratifiedShuffleSplit for cross-validation
#             cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

#             # Create the SVC model
#             svc = SVC()

#             # Perform grid search with cross-validation
#             grid_search = GridSearchCV(svc, param_grid, cv=cv, n_jobs=-1, verbose=2)

#             # Fit the grid search model
#             grid_search.fit(balanced_data.drop(outcome, axis=1), balanced_data[outcome])

#             # Get the best model
#             best_model = grid_search.best_estimator_

#             # Save the best model for the current outcome variable
#             dump(best_model, model_path)
#             ts_models[outcome] = best_model

#         return ts_models

#     except ValueError as e:
#         print(f"Error: {e}")
#         print("Possible reasons:")
#         print("1. Mismatch between the input data and the expected format.")
#         print("2. Invalid parameter values in the grid search.")
#     except Exception as e:
#         print(f"Error: An unexpected error occurred during Treatment Simulator building: {e}")
#         raise

# def evaluate_treatment_simulator(data_path, treatment_decisions, ts_models, num_bootstraps=1000):
#     try:
#         # Preprocess the data
#         balanced_data = preprocess_data(data_path)

#         # Initialize a dictionary to store the evaluation results
#         evaluation_results = {}

#         for outcome, model in ts_models.items():
#             print(f"Evaluating Treatment Simulator model for {outcome}")

#             # Initialize lists to store accuracy scores
#             accuracy_scores = []

#             # Perform bootstrapping
#             for _ in range(num_bootstraps):
#                 # Perform stratified bootstrapping
#                 bootstrap_indices = np.random.choice(len(balanced_data), size=len(balanced_data), replace=True)
#                 bootstrap_data = balanced_data.iloc[bootstrap_indices]

#                 # Split the bootstrap data into features and target
#                 X_bootstrap = bootstrap_data.drop(outcome, axis=1)
#                 y_bootstrap = bootstrap_data[outcome]

#                 # Make predictions using the Treatment Simulator model
#                 y_pred = model.predict(X_bootstrap)

#                 # Calculate the accuracy score
#                 accuracy = accuracy_score(y_bootstrap, y_pred)
#                 accuracy_scores.append(accuracy)

#             # Calculate the mean and 95% confidence interval for accuracy
#             mean_accuracy = np.mean(accuracy_scores)
#             ci_lower = np.percentile(accuracy_scores, 2.5)
#             ci_upper = np.percentile(accuracy_scores, 97.5)

#             # Print the evaluation results for the current outcome variable
#             print(f"Outcome: {outcome}")
#             print(f"Mean Accuracy: {mean_accuracy:.4f}")
#             print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

#             # Store the evaluation results
#             evaluation_results[outcome] = {
#                 'Mean Accuracy': mean_accuracy,
#                 '95% CI Lower': ci_lower,
#                 '95% CI Upper': ci_upper
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

# # def evaluate_dql_models(data_path, treatment_decisions, hidden_sizes):
# def evaluate_dql_models(data_path, treatment_decisions):

#     try:
#         # Preprocess the data
#         balanced_data = preprocess_data(data_path)

#         # Define input and output sizes based on the dataset
#         input_size = balanced_data.shape[1] - 1  # Exclude the target variable
#         output_size = 1  # Binary survival outcome

#         # Load the trained DQL models
#         # models = {decision: TreatmentModel(input_size, hidden_sizes, output_size) for decision in treatment_decisions}
#         # for decision in treatment_decisions:
#         #     models[decision].load_state_dict(torch.load(f'model_{decision}.pth'))

#         models = {}
#         for decision in treatment_decisions:
#             model_path = f'model_{decision}.pth'
#             if os.path.exists(model_path):
#                 # Load the saved model's state dictionary
#                 state_dict = torch.load(model_path)

#                 # Extract the hidden sizes from the saved model
#                 hidden_sizes = [state_dict['hidden_layers.0.weight'].shape[0], state_dict['hidden_layers.3.weight'].shape[0]]

#                 # Create a new model with the same architecture as the saved model
#                 model = TreatmentModel(input_size, hidden_sizes, output_size)
#                 model.load_state_dict(state_dict)
#                 models[decision] = model
#             else:
#                 print(f"Trained model not found for {decision}. Skipping evaluation.")


#         # Evaluate the models
#         evaluation_results = {}

#         for decision in treatment_decisions:
#             print(f"Evaluating DQL model for {decision}")

#             # Prepare the test data
#             X_test = torch.tensor(balanced_data.drop([decision], axis=1).values, dtype=torch.float32)
#             y_test = balanced_data[decision].values

#             with torch.no_grad():
#                 y_pred = models[decision](X_test).squeeze().numpy()

#             # Compare DQL model's decisions with physician's decisions
#             physician_decisions = balanced_data[decision].values
#             agreement = accuracy_score(physician_decisions, y_pred.round())
#             kappa = cohen_kappa_score(physician_decisions, y_pred.round())

#             # Calculate evaluation metrics
#             accuracy = accuracy_score(y_test, y_pred.round())
#             precision = precision_score(y_test, y_pred.round(), average='micro')
#             recall = recall_score(y_test, y_pred.round(), average='micro')
#             f1 = f1_score(y_test, y_pred.round(), average='micro')

#             # Store the evaluation results
#             evaluation_results[decision] = {
#                 'Accuracy': accuracy,
#                 'Precision': precision,
#                 'Recall': recall,
#                 'F1-score': f1,
#                 'Agreement': agreement,
#                 'Kappa': kappa
#             }

#             print(f"Evaluation metrics for {decision}:")
#             print(f"Accuracy: {accuracy:.4f}")
#             print(f"Precision: {precision:.4f}")
#             print(f"Recall: {recall:.4f}")
#             print(f"F1-score: {f1:.4f}")

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


# # def analyze_feature_importance(model, X_test, y_test):
# #     perm_importance = PermutationImportance(model, random_state=42).fit(X_test, y_test)
# #     return eli5.explain_weights(perm_importance, feature_names=X_test.columns.tolist())

# def analyze_feature_importance(X_test, y_test):
#     model = RandomForestClassifier(random_state=42)
#     model.fit(X_test, y_test)
#     feature_importances = model.feature_importances_
#     feature_names = X_test.columns.tolist()
#     importance_dict = {feature: importance for feature, importance in zip(feature_names, feature_importances)}
#     sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
#     return sorted_importances

# if __name__ == "__main__":
#     data_path = "dataset.csv"
#     treatment_decisions = ['surgery', 'chemotherapy', 'hormone_therapy', 'radiotherapy']
#     outcome_vars = ['OS', 'RFS', 'DFS']
#     hidden_sizes = [128, 64]

#     # Build and evaluate the Treatment Simulator models
#     ts_models = build_treatment_simulator(data_path, treatment_decisions, outcome_vars)
#     ts_evaluation_results = evaluate_treatment_simulator(data_path, treatment_decisions, ts_models)
#     print("Treatment Simulator evaluation completed.")

#     # Evaluate the DQL models
#     dql_evaluation_results = evaluate_dql_models(data_path, treatment_decisions, hidden_sizes)
#     print("DQL model evaluation completed.")



import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models import TreatmentModel
from preprocessing import preprocess_data
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from joblib import dump, load
import numpy as np
import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.ensemble import RandomForestClassifier

# import eli5
# from eli5.sklearn import PermutationImportance


def build_treatment_simulator(data_path, treatment_decisions, outcome_vars):
    try:
        # Preprocess the data
        balanced_data = preprocess_data(data_path)

        # Create a dictionary to store the SVC models for each outcome variable
        ts_models = {}

        for outcome in outcome_vars:
            model_path = f'ts_model_{outcome}.pkl'
            if os.path.exists(model_path):
            # Load the trained model if it exists
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
                'class_weight': ['balanced', None]
            }

            # Create StratifiedShuffleSplit for cross-validation
            cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

            # Create the SVC model
            svc = SVC()

            # Perform grid search with cross-validation
            grid_search = GridSearchCV(svc, param_grid, cv=cv, n_jobs=-1, verbose=2)

            # Fit the grid search model
            grid_search.fit(balanced_data.drop(outcome, axis=1), balanced_data[outcome])

            # Get the best model
            best_model = grid_search.best_estimator_

            # Save the best model for the current outcome variable
            dump(best_model, model_path)
            ts_models[outcome] = best_model

        return ts_models

    except ValueError as e:
        print(f"Error: {e}")
        print("Possible reasons:")
        print("1. Mismatch between the input data and the expected format.")
        print("2. Invalid parameter values in the grid search.")
    except Exception as e:
        print(f"Error: An unexpected error occurred during Treatment Simulator building: {e}")
        raise

def evaluate_treatment_simulator(data_path, treatment_decisions, ts_models, num_bootstraps=1000):
    try:
        # Preprocess the data
        balanced_data = preprocess_data(data_path)

        # Initialize a dictionary to store the evaluation results
        evaluation_results = {}

        for outcome, model in ts_models.items():
            print(f"Evaluating Treatment Simulator model for {outcome}")

            # Initialize lists to store accuracy scores
            accuracy_scores = []

            # Perform bootstrapping
            for _ in range(num_bootstraps):
                # Perform stratified bootstrapping
                bootstrap_indices = np.random.choice(len(balanced_data), size=len(balanced_data), replace=True)
                bootstrap_data = balanced_data.iloc[bootstrap_indices]

                # Split the bootstrap data into features and target
                X_bootstrap = bootstrap_data.drop(outcome, axis=1)
                y_bootstrap = bootstrap_data[outcome]

                # Make predictions using the Treatment Simulator model
                y_pred = model.predict(X_bootstrap)

                # Calculate the accuracy score
                accuracy = accuracy_score(y_bootstrap, y_pred)
                accuracy_scores.append(accuracy)

            # Calculate the mean and 95% confidence interval for accuracy
            mean_accuracy = np.mean(accuracy_scores)
            ci_lower = np.percentile(accuracy_scores, 2.5)
            ci_upper = np.percentile(accuracy_scores, 97.5)

            # Print the evaluation results for the current outcome variable
            print(f"Outcome: {outcome}")
            print(f"Mean Accuracy: {mean_accuracy:.4f}")
            print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

            # Store the evaluation results
            evaluation_results[outcome] = {
                'Mean Accuracy': mean_accuracy,
                '95% CI Lower': ci_lower,
                '95% CI Upper': ci_upper
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

# def evaluate_dql_models(data_path, treatment_decisions):
#     try:
#         # Preprocess the data
#         balanced_data = preprocess_data(data_path)

#         # Define input and output sizes based on the dataset
#         input_size = balanced_data.shape[1] - 1  # Exclude the target variable
#         output_size = 1  # Binary survival outcome

#         models = {}
#         for decision in treatment_decisions:
#             model_path = f'model_{decision}.pth'
#             if os.path.exists(model_path):
#                 # Load the saved model's state dictionary
#                 state_dict = torch.load(model_path)

#                 # Extract the hidden sizes and dropout from the saved model
#                 # hidden_sizes = [state_dict['hidden_layers.0.weight'].shape[0], state_dict['hidden_layers.4.weight'].shape[0]]
#                 # dropout = state_dict['hidden_layers.3.p']

#                 hidden_sizes = [v.shape[0] for k, v in state_dict.items() if 'weight' in k]
#                 dropout = next(v for k, v in state_dict.items() if 'p' in k)

#                 # Create a new model with the same architecture as the saved model
#                 model = TreatmentModel(input_size, hidden_sizes, output_size, dropout)
#                 model.load_state_dict(state_dict)
#                 models[decision] = model
#             else:
#                 print(f"Trained model not found for {decision}. Skipping evaluation.")

#         # Evaluate the models
#         evaluation_results = {}

#         for decision in treatment_decisions:
#             print(f"Evaluating DQL model for {decision}")

#             # Prepare the test data
#             X_test = torch.tensor(balanced_data.drop([decision], axis=1).values, dtype=torch.float32)
#             y_test = balanced_data[decision].values

#             with torch.no_grad():
#                 y_pred = models[decision](X_test).squeeze().numpy()

#             # Compare DQL model's decisions with physician's decisions
#             physician_decisions = balanced_data[decision].values
#             agreement = accuracy_score(physician_decisions, y_pred.round())
#             kappa = cohen_kappa_score(physician_decisions, y_pred.round())

#             # Calculate evaluation metrics
#             accuracy = accuracy_score(y_test, y_pred.round())
#             precision = precision_score(y_test, y_pred.round(), average='micro')
#             recall = recall_score(y_test, y_pred.round(), average='micro')
#             f1 = f1_score(y_test, y_pred.round(), average='micro')

#             # Store the evaluation results
#             evaluation_results[decision] = {
#                 'Accuracy': accuracy,
#                 'Precision': precision,
#                 'Recall': recall,
#                 'F1-score': f1,
#                 'Agreement': agreement,
#                 'Kappa': kappa
#             }

#             print(f"Evaluation metrics for {decision}:")
#             print(f"Accuracy: {accuracy:.4f}")
#             print(f"Precision: {precision:.4f}")
#             print(f"Recall: {recall:.4f}")
#             print(f"F1-score: {f1:.4f}")

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

# def evaluate_dql_models(data_path, treatment_decisions):
#     try:
#         # Preprocess the data
#         balanced_data = preprocess_data(data_path)

#         # Define input and output sizes based on the dataset
#         input_size = balanced_data.shape[1] - 1  # Exclude the target variable
#         output_size = 1  # Binary survival outcome

#         models = {}
#         for decision in treatment_decisions:
#             model_path = f'model_{decision}.pth'
#             if os.path.exists(model_path):
#                 # Load the saved model's state dictionary
#                 state_dict = torch.load(model_path)

#                 # Extract the hidden sizes and dropout from the saved model
#                 hidden_sizes = [v.shape[0] for k, v in state_dict.items() if 'weight' in k and 'hidden_layers' in k]
#                 dropout = next((v for k, v in state_dict.items() if 'dropout' in k), 0.0)

#                 # Create a new model with the same architecture as the saved model
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
#             X_test = torch.tensor(balanced_data.drop([decision], axis=1).values, dtype=torch.float32)
#             y_test = balanced_data[decision].values

#             with torch.no_grad():
#                 y_pred = models[decision](X_test).squeeze().numpy()

#             # Compare DQL model's decisions with physician's decisions
#             physician_decisions = balanced_data[decision].values
#             agreement = accuracy_score(physician_decisions, y_pred.round())
#             kappa = cohen_kappa_score(physician_decisions, y_pred.round())

#             # Calculate evaluation metrics
#             accuracy = accuracy_score(y_test, y_pred.round())
#             precision = precision_score(y_test, y_pred.round(), average='micro')
#             recall = recall_score(y_test, y_pred.round(), average='micro')
#             f1 = f1_score(y_test, y_pred.round(), average='micro')

#             # Store the evaluation results
#             evaluation_results[decision] = {
#                 'Accuracy': accuracy,
#                 'Precision': precision,
#                 'Recall': recall,
#                 'F1-score': f1,
#                 'Agreement': agreement,
#                 'Kappa': kappa
#             }

#             print(f"Evaluation metrics for {decision}:")
#             print(f"Accuracy: {accuracy:.4f}")
#             print(f"Precision: {precision:.4f}")
#             print(f"Recall: {recall:.4f}")
#             print(f"F1-score: {f1:.4f}")

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

# def evaluate_dql_models(data_path, treatment_decisions, hidden_sizes, dropout):
#     try:
#         # Preprocess the data
#         balanced_data = preprocess_data(data_path)

#         # Define input and output sizes based on the dataset
#         input_size = balanced_data.shape[1] - 1  # Exclude the target variable
#         output_size = 1  # Binary survival outcome

#         models = {}
#         for decision in treatment_decisions:
#             model_path = f'model_{decision}.pth'
#             if os.path.exists(model_path):
#                 # Load the saved model's state dictionary
#                 state_dict = torch.load(model_path)

#                 # Create a new model with fixed architecture
#                 # model = TreatmentModel(input_size, output_size)
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
#             # X_test = torch.tensor(balanced_data.drop([decision], axis=1).values, dtype=torch.float32)
#             X_test = torch.tensor(balanced_data.drop([decision], axis=1).values.astype(np.float32), dtype=torch.float32, device=torch.device('cpu'))
#             y_test = balanced_data[decision].values

#             with torch.no_grad():
#                 y_pred = models[decision](X_test).squeeze().numpy()

#             # Compare DQL model's decisions with physician's decisions
#             physician_decisions = balanced_data[decision].values
#             agreement = accuracy_score(physician_decisions, y_pred.round())
#             kappa = cohen_kappa_score(physician_decisions, y_pred.round())

#             # Calculate evaluation metrics
#             accuracy = accuracy_score(y_test, y_pred.round())
#             precision = precision_score(y_test, y_pred.round(), average='micro')
#             recall = recall_score(y_test, y_pred.round(), average='micro')
#             f1 = f1_score(y_test, y_pred.round(), average='micro')

#             # Store the evaluation results
#             evaluation_results[decision] = {
#                 'Accuracy': accuracy,
#                 'Precision': precision,
#                 'Recall': recall,
#                 'F1-score': f1,
#                 'Agreement': agreement,
#                 'Kappa': kappa
#             }

#             print(f"Evaluation metrics for {decision}:")
#             print(f"Accuracy: {accuracy:.4f}")
#             print(f"Precision: {precision:.4f}")
#             print(f"Recall: {recall:.4f}")
#             print(f"F1-score: {f1:.4f}")

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

def evaluate_dql_models(data_path, treatment_decisions, hidden_sizes, dropout):
    try:
        # Preprocess the data
        balanced_data = preprocess_data(data_path)

        # Define input and output sizes based on the dataset
        input_size = balanced_data.shape[1] - 1  # Exclude the target variable
        output_size = 1  # Binary survival outcome

        models = {}
        for decision in treatment_decisions:
            model_path = f'model_{decision}.pth'
            if os.path.exists(model_path):
                # Load the saved model's state dictionary
                state_dict = torch.load(model_path)

                # Create a new model with fixed architecture
                model = TreatmentModel(input_size, hidden_sizes, output_size, dropout)
                model.load_state_dict(state_dict)
                models[decision] = model
            else:
                print(f"Trained model not found for {decision}. Skipping evaluation.")

        # Evaluate the models
        evaluation_results = {}

        for decision in treatment_decisions:
            print("Processing decision:", decision)
            if decision not in models:
                continue

            print(f"Evaluating DQL model for {decision}")

            # Prepare the test data
            # X_test = balanced_data.drop([decision], axis=1).values.astype(np.float32)
            # y_test = balanced_data[decision].values

            # with torch.no_grad():
            #     X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            #     y_pred = models[decision](X_test_tensor).squeeze().numpy()

            # Prepare the test data
            X_test = np.array(balanced_data.drop([decision], axis=1).values.astype(np.float32))
            y_test = balanced_data[decision].values

            print(type(X_test))
            print(X_test.shape)

            with torch.no_grad():


                # X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                X_test_tensor = torch.Tensor(X_test)
                y_pred = models[decision](X_test_tensor).squeeze().numpy()

            # Compare DQL model's decisions with physician's decisions
            physician_decisions = balanced_data[decision].values
            agreement = accuracy_score(physician_decisions, y_pred.round())
            kappa = cohen_kappa_score(physician_decisions, y_pred.round())

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_test, y_pred.round())
            precision = precision_score(y_test, y_pred.round(), average='micro')
            recall = recall_score(y_test, y_pred.round(), average='micro')
            f1 = f1_score(y_test, y_pred.round(), average='micro')

            # Store the evaluation results
            evaluation_results[decision] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-score': f1,
                'Agreement': agreement,
                'Kappa': kappa
            }

            print(f"Evaluation metrics for {decision}:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-score: {f1:.4f}")

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
    model = RandomForestClassifier(random_state=42)
    model.fit(X_test, y_test)
    feature_importances = model.feature_importances_
    feature_names = X_test.columns.tolist()
    importance_dict = {feature: importance for feature, importance in zip(feature_names, feature_importances)}
    sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_importances

if __name__ == "__main__":
    data_path = "dataset.csv"
    treatment_decisions = ['surgery', 'chemotherapy', 'hormone_therapy', 'radiotherapy']
    outcome_vars = ['OS']
    hidden_sizes =[(128, 64), (256, 128), (512, 256), (256, 128, 64)],
    dropout = [0.1, 0.2, 0.3]


    # Build and evaluate the Treatment Simulator models
    ts_models = build_treatment_simulator(data_path, treatment_decisions, outcome_vars)
    ts_evaluation_results = evaluate_treatment_simulator(data_path, treatment_decisions, ts_models)
    print("Treatment Simulator evaluation completed.")

    # Evaluate the DQL models
    dql_evaluation_results = evaluate_dql_models(data_path, treatment_decisions, hidden_sizes)
    print("DQL model evaluation completed.")