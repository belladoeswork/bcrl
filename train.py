import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
            'hidden_sizes': [(64, 32), (128, 64), (256, 128), (512, 256)],
            'learning_rate': [0.0001, 0.001, 0.01, 0.1],
            'batch_size': [16, 32, 64, 128],
            'dropout': [0.1, 0.2, 0.3, 0.4],
            'l2_reg': [0.0001, 0.001, 0.01, 0.1]
        }
        
        # Perform random search
        best_models = {}
        for decision in treatment_decisions:
            best_loss = float('inf')
            best_model = None
            
            # Define input and output sizes based on the dataset
            input_size = balanced_data.drop([decision], axis=1).shape[1]
            output_size = 1
            
            for _ in range(num_trials):
                # Sample hyperparameters
                params = list(ParameterSampler(param_space, n_iter=1))[0]
                
                # Train the model with the sampled hyperparameters using the MDP approach
                model = TreatmentModel(input_size, params['hidden_sizes'], output_size, params['dropout']).to(device)
                
                # Split the balanced data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(balanced_data.drop([decision], axis=1), balanced_data[decision], test_size=0.2, random_state=42)
                
                # Convert data to PyTorch tensors
                X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
                y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
                X_test = torch.tensor(X_test.values, dtype=torch.float32).to(device)
                y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)
                
                # Define the loss function and optimizer
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_reg'])
                
                # Train the model using the MDP approach
                for epoch in range(num_epochs):
                    # Forward pass
                    outputs = model(X_train)
                    loss = criterion(outputs, y_train)
                    
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Compute the reward for the current decision
                    with torch.no_grad():
                        reward = torch.mean(((outputs > 0.5).float() == y_train).float())
                    
                    # Compute the value estimate for the current decision
                    with torch.no_grad():
                        value_estimate = torch.mean(outputs)
                    
                    # Update the best model if the current model has a lower loss
                    if loss < best_loss:
                        best_loss = loss
                        best_model = model
            
            # Save the best model for the current decision
            best_models[decision] = best_model
        
        # Train the models recursively using the MDP approach
        for i in range(len(treatment_decisions) - 1, -1, -1):
            decision = treatment_decisions[i]
            model = best_models[decision]
            
            # Compute the value estimate for the current decision using the trained model
            with torch.no_grad():
                value_estimate = torch.mean(model(X_train))
            
            # Use the value estimate as the reward for the previous decision
            if i > 0:
                prev_decision = treatment_decisions[i - 1]
                prev_model = best_models[prev_decision]
                
                # Update the previous model's reward with the current value estimate
                with torch.no_grad():
                    prev_model_outputs = prev_model(X_train)
                    # prev_model_reward = value_estimate
                    prev_model_reward = torch.full_like(prev_outputs, prev_model_reward)
                
                # Fine-tune the previous model with the updated reward
                prev_optimizer = optim.Adam(prev_model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_reg'])
                for _ in range(num_epochs // 2):
                    prev_optimizer.zero_grad()
                    prev_outputs = prev_model(X_train)
                    prev_loss = criterion(prev_outputs, prev_model_reward)
                    prev_loss.backward()
                    prev_optimizer.step()
        
        print("Training for all decisions completed.")
        return best_models
    
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Possible reasons:")
        print("1. Mismatch between the input data and the model's input size.")
        print("2. Insufficient memory to allocate tensors.")
        return {}
    except Exception as e:
        print(f"Error: An unexpected error occurred during training: {e}")
        return {}