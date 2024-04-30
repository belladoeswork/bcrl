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

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from models import TreatmentModel
# from preprocessing import preprocess_data
# from sklearn.model_selection import ParameterSampler


# # def train_models(data_path, treatment_decisions, hidden_sizes, num_epochs, learning_rate):

# def train_models(data_path, param_space, treatment_decisions, num_trials=10, num_epochs=100):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     try:
#         # Preprocess the data
#         balanced_data = preprocess_data(data_path)

#         # Define the hyperparameter search space
#         param_space = {
#             'hidden_sizes': [(128, 64), (256, 128), (512, 256)],
#             'learning_rate': [0.001, 0.01, 0.1],
#         }



#         # Perform random search
#         best_models = {}
#         for decision in treatment_decisions:
#             best_loss = float('inf')
#             best_model = None

#             # Define input and output sizes based on the dataset
#             input_size = balanced_data.drop([decision], axis=1).shape[1] 
#             output_size = 1  # Binary survival outcome

#             for _ in range(num_trials):
#                 # Sample hyperparameters
#                 params = list(ParameterSampler(param_space, n_iter=1))[0]

#                 # Train the model with the sampled hyperparameters
#                 model = TreatmentModel(input_size, params['hidden_sizes'], output_size).to(device)

                

#                 # Split the balanced data into training and testing sets
#                 X_train, X_test, y_train, y_test = train_test_split(balanced_data.drop([decision], axis=1),
#                                                                     balanced_data[decision],
#                                                                     test_size=0.2, random_state=42)

#                 # Convert data to PyTorch tensors
#                 X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
#                 y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)

#                 # Train the model
#                 criterion = nn.BCEWithLogitsLoss()
#                 optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

#                 for epoch in range(num_epochs):
#                     optimizer.zero_grad()
#                     outputs = model(X_train)
#                     loss = criterion(outputs, y_train)
#                     loss.backward()
#                     optimizer.step()

#                 # Save the trained model
#                 torch.save(model.state_dict(), f'model_{decision}.pth')

#                 # Update the best model if the current model has a lower loss
#                 if loss < best_loss:
#                     best_loss = loss
#                     best_model = model

#             best_models[decision] = best_model

#         return best_models

#     except RuntimeError as e:
#         print(f"Error: {e}")
#         print("Possible reasons:")
#         print("1. Mismatch between the input data and the model's input size.")
#         print("2. Insufficient memory to allocate tensors.")
#     except Exception as e:
#         print(f"Error: An unexpected error occurred during training: {e}")
#         raise






import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from models import TreatmentModel
from preprocessing import preprocess_data
from sklearn.model_selection import ParameterSampler

def train_models(data_path, param_space, treatment_decisions, num_trials=10, num_epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # Preprocess the data
        balanced_data = preprocess_data(data_path)

        # Define the hyperparameter search space
        param_space = {
            'hidden_sizes': [(128, 64), (256, 128), (512, 256), (256, 128, 64)],
            'learning_rate': [0.001, 0.01, 0.1],
            'batch_size': [32, 64, 128],
            'dropout': [0.1, 0.2, 0.3],
            'l2_reg': [0.001, 0.01, 0.1]
        }

        # Perform random search
        best_models = {}
        for decision in treatment_decisions:
            best_loss = float('inf')
            best_model = None

            # Define input and output sizes based on the dataset
            input_size = balanced_data.drop([decision], axis=1).shape[1]
            output_size = 1  # Binary survival outcome

            for _ in range(num_trials):
                # Sample hyperparameters
                params = list(ParameterSampler(param_space, n_iter=1))[0]

                # Train the model with the sampled hyperparameters
                model = TreatmentModel(input_size, params['hidden_sizes'], output_size, params['dropout']).to(device)

                # Split the balanced data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(balanced_data.drop([decision], axis=1),
                                                                    balanced_data[decision],
                                                                    test_size=0.2, random_state=42)

                # Convert data to PyTorch tensors
                X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
                y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
                X_test = torch.tensor(X_test.values, dtype=torch.float32).to(device)
                y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)

                # Train the model
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_reg'])

                batch_size = params['batch_size']
                train_losses = []
                val_losses = []

                for epoch in range(num_epochs):
                    # Mini-batch training
                    for i in range(0, len(X_train), batch_size):
                        batch_X = X_train[i:i+batch_size]
                        batch_y = y_train[i:i+batch_size]

                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                    # Evaluate on training set
                    with torch.no_grad():
                        train_outputs = model(X_train)
                        train_loss = criterion(train_outputs, y_train)
                        train_losses.append(train_loss.item())

                    # Evaluate on validation set
                    with torch.no_grad():
                        val_outputs = model(X_test)
                        val_loss = criterion(val_outputs, y_test)
                        val_losses.append(val_loss.item())

                # Save the trained model
                torch.save(model.state_dict(), f'model_{decision}.pth')

                # Update the best model if the current model has a lower validation loss
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model = model

            best_models[decision] = best_model

        return best_models, train_losses, val_losses

    except RuntimeError as e:
        print(f"Error: {e}")
        print("Possible reasons:")
        print("1. Mismatch between the input data and the model's input size.")
        print("2. Insufficient memory to allocate tensors.")
    except Exception as e:
        print(f"Error: An unexpected error occurred during training: {e}")
        raise