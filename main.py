# main.py version 1.0

from preprocessing import preprocess_data
from train import train_models
from evaluate import build_treatment_simulator, evaluate_treatment_simulator, evaluate_dql_models

def main():
    data_path = "dataset.csv"
    treatment_decisions = ['surgery', 'chemotherapy', 'hormone_therapy', 'radiotherapy']
    outcome_vars = ['OS', 'RFS', 'DFS']
    hidden_sizes = [128, 64]
    num_epochs = 100
    learning_rate = 0.001

    # Preprocess the data
    processed_data = preprocess_data(data_path)
    print("Data preprocessing completed.")

    # Train the DQL models
    trained_models = train_models(data_path, treatment_decisions, hidden_sizes, num_epochs, learning_rate)
    print("Model training completed.")

    # Build and evaluate the Treatment Simulator models
    ts_models = build_treatment_simulator(data_path, treatment_decisions, outcome_vars)
    ts_evaluation_results = evaluate_treatment_simulator(data_path, treatment_decisions, ts_models)
    print("Treatment Simulator evaluation completed.")

    # Evaluate the DQL models
    dql_evaluation_results = evaluate_dql_models(data_path, treatment_decisions, hidden_sizes)
    print("DQL model evaluation completed.")

    # Print the evaluation results
    print("\nTreatment Simulator Evaluation Results:")
    for outcome, results in ts_evaluation_results.items():
        print(f"Outcome: {outcome}")
        print(f"Mean Accuracy: {results['Mean Accuracy']:.4f}")
        print(f"95% Confidence Interval: [{results['95% CI Lower']:.4f}, {results['95% CI Upper']:.4f}]")

    print("\nDQL Model Evaluation Results:")
    for decision, results in dql_evaluation_results.items():
        print(f"Decision: {decision}")
        print(f"Accuracy: {results['Accuracy']:.4f}")
        print(f"Precision: {results['Precision']:.4f}")
        print(f"Recall: {results['Recall']:.4f}")
        print(f"F1-score: {results['F1-score']:.4f}")

if __name__ == "__main__":
    main()
