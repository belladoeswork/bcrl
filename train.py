# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from models import TreatmentModel
# from preprocessing import preprocess_data

# def train_models(data_path, treatment_decisions, hidden_size, num_epochs, learning_rate):
#     # Preprocess the data
#     balanced_data = preprocess_data(data_path)

#     # Create a dictionary to store the models for each treatment decision
#     models = {decision: TreatmentModel(input_size, hidden_size, output_size) for decision in treatment_decisions}

#     # Train the models recursively
#     for i in range(len(treatment_decisions) - 1, -1, -1):
#         decision = treatment_decisions[i]
#         model = models[decision]

#         # Split the balanced data into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(balanced_data[decision].drop([decision], axis=1),
#                                                             balanced_data[decision][decision],
#                                                             test_size=0.2, random_state=42)

#         # Convert data to PyTorch tensors
#         X_train = torch.tensor(X_train.values, dtype=torch.float32)
#         y_train = torch.tensor(y_train.values, dtype=torch.float32)

#         # Train the model
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#         for epoch in range(num_epochs):
#             optimizer.zero_grad()
#             outputs = model(X_train)
#             loss = criterion(outputs, y_train)
#             loss.backward()
#             optimizer.step()

#         # Save the trained model
#         torch.save(model.state_dict(), f'model_{decision}.pth')

#     return models

# train.py version 1.0

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from models import TreatmentModel
from preprocessing import preprocess_data

def train_models(data_path, treatment_decisions, hidden_sizes, num_epochs, learning_rate):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # Preprocess the data
        balanced_data = preprocess_data(data_path)

        # Define input and output sizes based on the dataset
        input_size = balanced_data.shape[1] - 1  # Exclude the target variable
        output_size = 1  # Binary survival outcome

        # Create a dictionary to store the models for each treatment decision
        models = {decision: TreatmentModel(input_size, hidden_sizes, output_size).to(device) for decision in treatment_decisions}

        # Train the models recursively
        for i in range(len(treatment_decisions) - 1, -1, -1):
            decision = treatment_decisions[i]
            model = models[decision]

            # Split the balanced data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(balanced_data.drop([decision], axis=1),
                                                                balanced_data[decision],
                                                                test_size=0.2, random_state=42)

            # Convert data to PyTorch tensors
            X_train = torch.tensor(X_train.values, dtype=torch.float32)
            y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

            # Train the model
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

            # Save the trained model
            torch.save(model.state_dict(), f'model_{decision}.pth')

        return models

    except RuntimeError as e:
        print(f"Error: {e}")
        print("Possible reasons:")
        print("1. Mismatch between the input data and the model's input size.")
        print("2. Insufficient memory to allocate tensors.")
    except Exception as e:
        print(f"Error: An unexpected error occurred during training: {e}")
        raise
    

if __name__ == "__main__":
    data_path = "dataset.csv"
    treatment_decisions = ['surgery', 'chemotherapy', 'hormone_therapy', 'radiotherapy']
    hidden_sizes = [128, 64]
    num_epochs = 100
    learning_rate = 0.001

    trained_models = train_models(data_path, treatment_decisions, hidden_sizes, num_epochs, learning_rate)
    print("Model training completed.")