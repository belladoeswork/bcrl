import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from models_try import TreatmentModel
from preprocessing import preprocess_data
from sklearn.model_selection import ParameterSampler
from torch.utils.data import DataLoader, TensorDataset

def train_models(data_path, param_space, treatment_decisions, num_trials=20, num_epochs=300):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        # Preprocess the data
        balanced_data = preprocess_data(data_path)
        
        # Perform stratified k-fold cross-validation
        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        
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
                
                fold_losses = []
                for train_index, val_index in skf.split(balanced_data.drop([decision], axis=1), balanced_data[decision]):
                    # Split the data into training and validation sets
                    X_train, X_val = balanced_data.drop([decision], axis=1).iloc[train_index], balanced_data.drop([decision], axis=1).iloc[val_index]
                    y_train, y_val = balanced_data[decision].iloc[train_index], balanced_data[decision].iloc[val_index]
                    
                    # Convert data to PyTorch tensors
                    X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
                    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
                    X_val = torch.tensor(X_val.values, dtype=torch.float32).to(device)
                    y_val = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1).to(device)
                    
                    # Create data loaders
                    train_dataset = TensorDataset(X_train, y_train)
                    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
                    
                    # Initialize the model
                    model = TreatmentModel(input_size, params['hidden_sizes'], output_size, params['dropout']).to(device)
                    
                    # Define the loss function, optimizer, and learning rate scheduler
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_reg'])
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
                    
                    # Train the model
                    for epoch in range(num_epochs):
                        model.train()
                        for batch in train_loader:
                            X_batch, y_batch = batch
                            
                            # Forward pass
                            outputs = model(X_batch)
                            loss = criterion(outputs, y_batch)
                            
                            # Backward pass and optimization
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        
                        # Evaluate the model on the validation set
                        model.eval()
                        with torch.no_grad():
                            outputs = model(X_val)
                            val_loss = criterion(outputs, y_val)
                        
                        # Update the learning rate
                        scheduler.step(val_loss)
                        
                        # Check for early stopping
                        if optimizer.param_groups[0]['lr'] < 1e-6:
                            print(f"Early stopping at epoch {epoch}")
                            break
                    
                    fold_losses.append(val_loss.item())
                
                avg_loss = np.mean(fold_losses)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
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
                    prev_model_reward = torch.full_like(prev_model_outputs, value_estimate.item())
                
                # Fine-tune the previous model with the updated reward
                prev_optimizer = optim.AdamW(prev_model.parameters(), lr=params['learning_rate'], weight_decay=params['l2_reg'])
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