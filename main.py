# main.py version 1.0

# from preprocessing import preprocess_data, select_features
# from train import train_models
# import os
# import torch
# from joblib import load
# from evaluate import build_treatment_simulator, evaluate_treatment_simulator, evaluate_dql_models, analyze_feature_importance


# def main():
#     data_path = "synthetic_dataset.csv"
#     treatment_decisions = ['surgery', 'chemotherapy', 'hormone_therapy', 'radiotherapy']
#     # outcome_vars = ['OS', 'RFS', 'DFS']
#     outcome_vars = ['OS']
#     # hidden_sizes = [128, 64]
#     num_epochs = 100
#     # learning_rate = 0.001
#     sample_size = 500

#     param_space = {
#         'hidden_sizes': [(128, 64), (256, 128), (512, 256)],
#         'learning_rate': [0.001, 0.01, 0.1],
#     }


#     # Preprocess the data
#     # processed_data = preprocess_data(data_path)
#     processed_data = preprocess_data(data_path, sample_size=sample_size)
#     print("Data preprocessing completed.")

#     # Perform feature selection
#     selected_features = select_features(processed_data, outcome_vars)
#     print("Feature selection completed.")
#     print("Selected features for each outcome variable:")
#     for outcome, features in selected_features.items():
#         print(f"{outcome}: {features}")

#     # Train the DQL models
#     # trained_models = train_models(data_path, treatment_decisions, hidden_sizes, num_epochs, learning_rate)
#     trained_models = train_models(data_path, param_space, treatment_decisions, num_trials=10, num_epochs=num_epochs)
#     print("Model training completed.")

#     # Analyze feature importance
#     # for decision, model in trained_models.items():
#     #     X_test = torch.tensor(processed_data.drop([decision], axis=1).values, dtype=torch.float32)
#     #     y_test = processed_data[decision].values

#     #     importance_results = analyze_feature_importance(model, X_test, y_test)
#     #     print(f"Feature importance for {decision}:")
#     #     print(importance_results)
#     for decision in treatment_decisions:
#         X_test = processed_data.drop([decision], axis=1)
#         y_test = processed_data[decision]

#         importance_results = analyze_feature_importance(X_test, y_test)
#         print(f"Feature importance for {decision}:")
#         print(importance_results)


#     # Build and evaluate the Treatment Simulator models
#     ts_models = build_treatment_simulator(data_path, treatment_decisions, outcome_vars)
#     ts_evaluation_results = evaluate_treatment_simulator(data_path, treatment_decisions, ts_models)
#     print("Treatment Simulator evaluation completed.")
#     ts_models = {}
#     ts_evaluation_results = {}
#     for outcome in outcome_vars:
#         model_path = f'ts_model_{outcome}.pkl'
#         if os.path.exists(model_path):
#             print(f"Treatment Simulator model for {outcome} already exists. Loading from file.")
#             ts_models[outcome] = load(model_path)
#         else:
#             print(f"Building and evaluating Treatment Simulator model for {outcome}.")
#             ts_model = build_treatment_simulator(data_path, treatment_decisions, [outcome])
#             ts_models.update(ts_model)
#             ts_evaluation_result = evaluate_treatment_simulator(data_path, treatment_decisions, ts_model)
#             ts_evaluation_results.update(ts_evaluation_result)
#             print(f"Treatment Simulator evaluation completed for {outcome}.")
    

#     # Evaluate the DQL models
#     # dql_evaluation_results = evaluate_dql_models(data_path, treatment_decisions, hidden_sizes)
#     dql_evaluation_results = evaluate_dql_models(data_path, treatment_decisions)
#     if dql_evaluation_results is not None:
#         print("DQL model evaluation completed.")
#     else:
#         print("DQL model evaluation failed.")

#     # Print the evaluation results
#     print("\nTreatment Simulator Evaluation Results:")
#     for outcome, results in ts_evaluation_results.items():
#         print(f"Outcome: {outcome}")
#         print(f"Mean Accuracy: {results['Mean Accuracy']:.4f}")
#         print(f"95% Confidence Interval: [{results['95% CI Lower']:.4f}, {results['95% CI Upper']:.4f}]")

#     if dql_evaluation_results is not None:
#         print("\nDQL Model Evaluation Results:")
#         for decision, results in dql_evaluation_results.items():
#             print(f"Decision: {decision}")
#             print(f"Accuracy: {results['Accuracy']:.4f}")
#             print(f"Precision: {results['Precision']:.4f}")
#             print(f"Recall: {results['Recall']:.4f}")
#             print(f"F1-score: {results['F1-score']:.4f}")
#     else:
#         print("DQL Model Evaluation failed.")

# if __name__ == "__main__":
#     main()





# remove this
#     # Check if trained models exist
#     trained_models_exist = all(os.path.exists(f'model_{decision}.pth') for decision in treatment_decisions)

#     if not trained_models_exist:
#         print("Trained DQL models do not exist. Please train the models first.")
#         return

#     # Evaluate the DQL models
#     try:
#         dql_evaluation_results = evaluate_dql_models(data_path, treatment_decisions)
#         print("DQL model evaluation completed.")
#     except Exception as e:
#         print(f"Error: An unexpected error occurred during DQL model evaluation: {e}")
#         return

#     # Print the evaluation results
#     print("\nDQL Model Evaluation Results:")
#     for decision, results in dql_evaluation_results.items():
#         print(f"Decision: {decision}")
#         print(f"Accuracy: {results['Accuracy']:.4f}")
#         print(f"Precision: {results['Precision']:.4f}")
#         print(f"Recall: {results['Recall']:.4f}")
#         print(f"F1-score: {results['F1-score']:.4f}")

# if __name__ == "__main__":
#     main()



# def main():
#     data_path = "dataset.csv"
#     treatment_decisions = ['surgery', 'chemotherapy', 'hormone_therapy', 'radiotherapy']
#     outcome_vars = ['OS', 'RFS', 'DFS']
#     num_epochs = 100
#     sample_size = 10000

#     param_space = {
#         'hidden_sizes': [(128, 64), (256, 128), (512, 256)],
#         'learning_rate': [0.001, 0.01, 0.1],
#     }

#     # Preprocess the data
#     # processed_data = preprocess_data(data_path, sample_size=sample_size)
#     # print("Data preprocessing completed.")

#     # Perform feature selection
#     # selected_features = select_features(processed_data, outcome_vars)
#     # print("Feature selection completed.")
#     # print("Selected features for each outcome variable:")
#     # for outcome, features in selected_features.items():
#     #     print(f"{outcome}: {features}")

#     # Train the DQL models
#     # trained_models = train_models(data_path, param_space, treatment_decisions, num_trials=10, num_epochs=num_epochs)
#     # print("Model training completed.")

#     # Analyze feature importance
#     # for decision in treatment_decisions:
#     #     X_test = processed_data.drop([decision], axis=1)
#     #     y_test = processed_data[decision]

#     #     importance_results = analyze_feature_importance(X_test, y_test)
#     #     print(f"Feature importance for {decision}:")
#     #     print(importance_results)

#     # Check if Treatment Simulator for DFS exists
#     dfs_model_path = 'ts_model_DFS.pkl'
#     if os.path.exists(dfs_model_path):
#         print("Treatment Simulator model for DFS already exists. Loading from file.")
#         ts_models = {'DFS': load(dfs_model_path)}
#         ts_evaluation_results = {'DFS': evaluate_treatment_simulator(data_path, treatment_decisions, ts_models)}
#         print("Treatment Simulator evaluation completed for DFS.")

#         # Evaluate the DQL model for DFS
#         dfs_dql_evaluation_results = evaluate_dql_models(data_path, treatment_decisions)
#         if dfs_dql_evaluation_results is not None:
#             print("DQL model evaluation completed for DFS.")
#             dql_evaluation_results = {'DFS': dfs_dql_evaluation_results}
#         else:
#             print("DQL model evaluation failed for DFS.")
#     else:
#         print("Treatment Simulator model for DFS does not exist. Building from scratch for all outcomes.")
        
#         # Build and evaluate the Treatment Simulator models for all outcomes
#         ts_models = {}
#         ts_evaluation_results = {}
#         for outcome in outcome_vars:
#             print(f"Building and evaluating Treatment Simulator model for {outcome}.")
#             ts_model = build_treatment_simulator(data_path, treatment_decisions, [outcome])
#             ts_models.update(ts_model)
#             ts_evaluation_result = evaluate_treatment_simulator(data_path, treatment_decisions, ts_model)
#             ts_evaluation_results.update(ts_evaluation_result)
#             print(f"Treatment Simulator evaluation completed for {outcome}.")
        
#         # Evaluate the DQL models for all outcomes
#         dql_evaluation_results = {}
#         for outcome in outcome_vars:
#             outcome_dql_evaluation_results = evaluate_dql_models(data_path, treatment_decisions)
#             if outcome_dql_evaluation_results is not None:
#                 print(f"DQL model evaluation completed for {outcome}.")
#                 dql_evaluation_results[outcome] = outcome_dql_evaluation_results
#             else:
#                 print(f"DQL model evaluation failed for {outcome}.")
    
#     # Print the evaluation results
#     print("\nTreatment Simulator Evaluation Results:")
#     for outcome, results in ts_evaluation_results.items():
#         print(f"Outcome: {outcome}")
#         print(f"Mean Accuracy: {results['Mean Accuracy']:.4f}")
#         print(f"95% Confidence Interval: [{results['95% CI Lower']:.4f}, {results['95% CI Upper']:.4f}]")

#     print("\nDQL Model Evaluation Results:")
#     for outcome, results in dql_evaluation_results.items():
#         print(f"Outcome: {outcome}")
#         for decision, metrics in results.items():
#             print(f"Decision: {decision}")
#             print(f"Accuracy: {metrics['Accuracy']:.4f}")
#             print(f"Precision: {metrics['Precision']:.4f}")
#             print(f"Recall: {metrics['Recall']:.4f}")
#             print(f"F1-score: {metrics['F1-score']:.4f}")

# if __name__ == "__main__":
#     main()


from preprocessing import preprocess_data, select_features
from train import train_models
import os
import torch
from joblib import load
from evaluate import build_treatment_simulator, evaluate_treatment_simulator, evaluate_dql_models, analyze_feature_importance


def main():
    data_path = "synthetic_dataset.csv"
    treatment_decisions = ['surgery', 'chemotherapy', 'hormone_therapy', 'radiotherapy']
    outcome_vars = ['OS']
    num_epochs = 100
    sample_size = 500

    param_space = {
        'hidden_sizes': [(128, 64), (256, 128), (512, 256), (256, 128, 64)],
        'learning_rate': [0.001, 0.01, 0.1],
        'batch_size': [32, 64, 128],
        'dropout': [0.1, 0.2, 0.3],
        'l2_reg': [0.001, 0.01, 0.1]
    }

    # Preprocess the data
    processed_data = preprocess_data(data_path, sample_size=sample_size)
    print("Data preprocessing completed.")

    # Perform feature selection
    selected_features = select_features(processed_data, outcome_vars)
    print("Feature selection completed.")
    print("Selected features for each outcome variable:")
    for outcome, features in selected_features.items():
        print(f"{outcome}: {features}")

    # Train the DQL models
    trained_models, train_losses, val_losses = train_models(data_path, param_space, treatment_decisions, num_trials=10, num_epochs=num_epochs)
    print("Model training completed.")

    # Analyze feature importance
    for decision in treatment_decisions:
        X_test = processed_data.drop([decision], axis=1)
        y_test = processed_data[decision]

        importance_results = analyze_feature_importance(X_test, y_test)
        print(f"Feature importance for {decision}:")
        print(importance_results)

    # Build and evaluate the Treatment Simulator models
    ts_models = {}
    ts_evaluation_results = {}
    # for outcome in outcome_vars:
    #     model_path = f'ts_model_{outcome}.pkl'
    #     if os.path.exists(model_path):
    #         print(f"Treatment Simulator model for {outcome} already exists. Loading from file.")
    #         ts_models[outcome] = load(model_path)
    #     else:
    #         print(f"Building and evaluating Treatment Simulator model for {outcome}.")
    #         ts_model = build_treatment_simulator(data_path, treatment_decisions, [outcome])
    #         ts_models.update(ts_model)
    #         ts_evaluation_result = evaluate_treatment_simulator(data_path, treatment_decisions, ts_model)
    #         ts_evaluation_results.update(ts_evaluation_result)
    #         print(f"Treatment Simulator evaluation completed for {outcome}.")


    for outcome in outcome_vars:
        print(f"Building and evaluating Treatment Simulator model for {outcome}.")
        ts_model = build_treatment_simulator(data_path, treatment_decisions, [outcome])
        ts_models.update(ts_model)
        ts_evaluation_result = evaluate_treatment_simulator(data_path, treatment_decisions, ts_model)
        ts_evaluation_results.update(ts_evaluation_result)
        print(f"Treatment Simulator evaluation completed for {outcome}.")

    # Evaluate the DQL models
    dql_evaluation_results = evaluate_dql_models(data_path, treatment_decisions, param_space['hidden_sizes'], param_space['dropout'][0])
    if dql_evaluation_results is not None:
        print("DQL model evaluation completed.")
    else:
        print("DQL model evaluation failed.")

    # Print the evaluation results
    print("\nTreatment Simulator Evaluation Results:")
    for outcome, results in ts_evaluation_results.items():
        print(f"Outcome: {outcome}")
        print(f"Mean Accuracy: {results['Mean Accuracy']:.4f}")
        print(f"95% Confidence Interval: [{results['95% CI Lower']:.4f}, {results['95% CI Upper']:.4f}]")

    if dql_evaluation_results is not None:
        print("\nDQL Model Evaluation Results:")
        for decision, results in dql_evaluation_results.items():
            print(f"Decision: {decision}")
            print(f"Accuracy: {results['Accuracy']:.4f}")
            print(f"Precision: {results['Precision']:.4f}")
            print(f"Recall: {results['Recall']:.4f}")
            print(f"F1-score: {results['F1-score']:.4f}")
    else:
        print("DQL Model Evaluation failed.")

if __name__ == "__main__":
    main()