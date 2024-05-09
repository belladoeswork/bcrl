from preprocessing_try import preprocess_data, select_features
from train_try import train_models
import os
import torch
from joblib import load
# from evaluate_try import build_treatment_simulator, evaluate_treatment_simulator, evaluate_dql_models
from evaluate_try import build_treatment_simulator, evaluate_treatment_simulator, evaluate_dql_models, analyze_feature_importance

def main():
    data_path = "synthetic_dataset.csv"
    treatment_decisions = ['surgery', 'chemotherapy', 'hormone_therapy', 'radiotherapy']
    outcome_vars = ['OS', 'DFS', 'RFS']
    num_epochs = 300
    sample_size = 1200
    param_space = {
        # 'hidden_sizes': [(128, 64), (256, 128), (512, 256), (256, 128, 64)],
        'hidden_sizes': [(64, 32), (128, 64), (256, 128), (512, 256)],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1],
        # 'batch_size': [32, 64, 128],
        'batch_size': [16, 32, 64, 128],
        'dropout': [0.1, 0.2, 0.3, 0.4],
        'l2_reg': [0.0001, 0.001, 0.01, 0.1]
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
    # trained_models, train_losses, val_losses = train_models(data_path, param_space, treatment_decisions, num_trials=10, num_epochs=num_epochs)
    trained_models = train_models(data_path, param_space, treatment_decisions, num_trials=10, num_epochs=num_epochs)
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
    for outcome in outcome_vars:
        print(f"Building and evaluating Treatment Simulator model for {outcome}.")
        ts_model, feature_names = build_treatment_simulator(data_path, treatment_decisions, [outcome])
        ts_models.update(ts_model)
        ts_evaluation_result = evaluate_treatment_simulator(data_path, treatment_decisions, ts_model, feature_names)
        ts_evaluation_results.update(ts_evaluation_result)
        print(f"Treatment Simulator evaluation completed for {outcome}.")       
    # Evaluate the DQL models
    dql_evaluation_results = evaluate_dql_models(data_path, treatment_decisions)
    if dql_evaluation_results is not None:
        print("DQL model evaluation completed.")
    else:
        print("DQL model evaluation failed.")
    # Print the evaluation results
    print("\nTreatment Simulator Evaluation Results:")
    for outcome, results in ts_evaluation_results.items():
        print(f"Outcome: {outcome}")
        print(f"Mean Accuracy: {results['Mean Accuracy']:.4f}")
        # print(f"95% Confidence Interval: [{results['95% CI Lower']:.4f}, {results['95% CI Upper']:.4f}]")
        print(f"95% Confidence Interval: [{results['95% CI for Accuracy'][0]:.4f}, {results['95% CI for Accuracy'][1]:.4f}]")
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