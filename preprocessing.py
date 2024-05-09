import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif

def preprocess_data(data_path, sample_size=None, random_state=42):
    try:
        # Load the data
        data = pd.read_csv(data_path)

        # print("Unique values of 'OS' variable before preprocessing:")
        # print(data['OS'].unique())

        if sample_size is not None:
            # Perform stratified sampling based on all outcome variables
            outcome_vars = ['OS', 'DFS', 'RFS']
            stratify_data = data[outcome_vars].apply(lambda x: ''.join(x.astype(str)), axis=1)
            split = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=random_state)
            for _, sample_indices in split.split(data, stratify_data):
                data = data.loc[sample_indices]
        # Remove constant features
        constant_columns = data.columns[data.nunique() <= 1]
        data = data.drop(columns=constant_columns)
        # Handle missing data
        data.fillna(data.mode().iloc[0], inplace=True) 
        # Encode categorical variables
        label_encoder = LabelEncoder()
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            data[col] = label_encoder.fit_transform(data[col])
        # Scale the features
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data[data.columns] = scaler.fit_transform(data[data.columns])
        # Handle class imbalance
        # outcome_vars = ['OS']
        # original_columns = data.columns
        # for outcome in outcome_vars:
        #     data_copy = data.copy()
        #     X = data_copy.drop(outcome, axis=1)  
        #     y = data_copy[outcome]
        #     oversampler = RandomOverSampler(random_state=random_state)
        #     X_resampled, y_resampled = oversampler.fit_resample(X, y)
        #     data_copy = pd.concat([X_resampled, y_resampled], axis=1)
        outcome_vars = ['OS', 'DFS', 'RFS']
        original_columns = data.columns
        for outcome in outcome_vars:
            data_copy = data.copy()
            X = data_copy.drop(outcome, axis=1)
            y = data_copy[outcome]
            smote = SMOTE(random_state=random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            data_copy = pd.concat([X_resampled, y_resampled], axis=1)
            # Reorder columns to match the original order
            data_copy = data_copy[original_columns]

    except FileNotFoundError:
        print(f"Error: The file '{data_path}' could not be found.")
        raise
    except KeyError as e:
        print(f"Error: The column '{e.args[0]}' does not exist in the dataset.")
        raise
    except Exception as e:
        print(f"Error: An unexpected error occurred during preprocessing: {e}")
        raise
    # return data_copy

    # print("Unique values of 'OS' variable after preprocessing:")
    # print(data['OS'].unique())

    return data

def select_features(data, outcome_vars, k=10):
    X = data.drop(outcome_vars, axis=1)
    selected_features = {}
        
    for outcome in outcome_vars:
        if outcome not in data.columns:
            raise KeyError(f"The column '{outcome}' does not exist in the dataset.")
            
        y = data[outcome]
            
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)
            
        selected_columns = X.columns[selector.get_support()].tolist()
        selected_features[outcome] = selected_columns
        
    return selected_features

if __name__ == "__main__":
    data_path = "synthetic_dataset.csv"
    sample_size = 1200
    processed_data = preprocess_data(data_path, sample_size=sample_size)
    print("Data preprocessing completed.")




from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd

# def preprocess_data(data_path, sample_size=None, random_state=42):
#     try:
#         # Load the data
#         data = pd.read_csv(data_path)
        
#         if sample_size is not None:
#             # Perform stratified sampling based on all outcome variables
#             outcome_vars = ['OS']
#             stratify_data = data[outcome_vars].apply(lambda x: ''.join(x.astype(str)), axis=1)
            
#             # Check if stratified sampling is possible
#             if stratify_data.nunique() > 1 and all(stratify_data.value_counts() > 1):
#                 split = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=random_state)
#                 for _, sample_indices in split.split(data, stratify_data):
#                     data = data.loc[sample_indices]
#             else:
#                 # If stratified sampling is not possible, perform random sampling
#                 data = data.sample(n=sample_size, random_state=random_state)
        
#         # Remove constant features
#         constant_columns = data.columns[data.nunique() <= 1]
#         data = data.drop(columns=constant_columns)
        
#         # Handle missing data
#         data.fillna(data.mode().iloc[0], inplace=True)
        
#         # Encode categorical variables
#         label_encoder = LabelEncoder()
#         categorical_columns = data.select_dtypes(include=['object']).columns
#         for col in categorical_columns:
#             data[col] = label_encoder.fit_transform(data[col])
        
#         # Scale the features
#         scaler = MinMaxScaler(feature_range=(-1, 1))
#         data[data.columns] = scaler.fit_transform(data[data.columns])
        
#         outcome_vars = ['OS']
#         original_columns = data.columns
        
#         for outcome in outcome_vars:
#             data_copy = data.copy()
#             X = data_copy.drop(outcome, axis=1)
#             y = data_copy[outcome].astype(int)
            
#             smote = SMOTE(random_state=random_state)
#             X_resampled, y_resampled = smote.fit_resample(X, y)
            
#             data_copy = pd.concat([X_resampled, y_resampled], axis=1)
            
#             # Reorder columns to match the original order
#             data_copy = data_copy[original_columns]


#             unique_values = data['OS'].unique()
#             if len(unique_values) != 2 or not all(value in [0, 1] for value in unique_values):
#                 raise ValueError("The 'OS' variable should contain only binary values (0 or 1).")
            
        
#         return data_copy
    
#     except FileNotFoundError:
#         print(f"Error: The file '{data_path}' could not be found.")
#         raise
#     except KeyError as e:
#         print(f"Error: The column '{e.args[0]}' does not exist in the dataset.")
#         raise
#     except Exception as e:
#         print(f"Error: An unexpected error occurred during preprocessing: {e}")
#         raise

# def select_features(data, outcome_vars, k=10):
#     X = data.drop(outcome_vars, axis=1)
#     selected_features = {}
    
#     for outcome in outcome_vars:
#         if outcome not in data.columns:
#             raise KeyError(f"The column '{outcome}' does not exist in the dataset.")
        
#         y = data[outcome]
        
#         selector = SelectKBest(score_func=f_classif, k=k)
#         selector.fit(X, y)
        
#         selected_columns = X.columns[selector.get_support()].tolist()
#         selected_features[outcome] = selected_columns
    
#     return selected_features

# if __name__ == "__main__":
#     data_path = "synthetic_dataset.csv"
#     sample_size = 1500
#     processed_data = preprocess_data(data_path, sample_size=sample_size)
#     print("Data preprocessing completed.")


from sklearn.exceptions import DataConversionWarning
import warnings

# def preprocess_data(data_path, sample_size=None, random_state=42):
#     try:
#         # Load the data
#         data = pd.read_csv(data_path)
        
#         # Debug: Print unique values of the 'OS' variable before any modifications
#         print("Unique values of 'OS' variable before preprocessing:")
#         print(data['OS'].unique())
        
#         if sample_size is not None:
#             # Perform stratified sampling based on all outcome variables
#             outcome_vars = ['OS']
#             stratify_data = data[outcome_vars].apply(lambda x: ''.join(x.astype(str)), axis=1)
            
#             # Check if stratified sampling is possible
#             if stratify_data.nunique() > 1 and all(stratify_data.value_counts() > 1):
#                 split = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=random_state)
#                 for _, sample_indices in split.split(data, stratify_data):
#                     data = data.loc[sample_indices]
#             else:
#                 # If stratified sampling is not possible, perform random sampling
#                 data = data.sample(n=sample_size, random_state=random_state)
        
#         # Remove constant features
#         constant_columns = data.columns[data.nunique() <= 1]
#         data = data.drop(columns=constant_columns)
        
#         # Handle missing data
#         data.fillna(data.mode().iloc[0], inplace=True)
        
#         # Encode categorical variables
#         label_encoder = LabelEncoder()
#         categorical_columns = data.select_dtypes(include=['object']).columns
#         for col in categorical_columns:
#             data[col] = label_encoder.fit_transform(data[col])
        
#         # Scale the features
#         scaler = MinMaxScaler(feature_range=(-1, 1))
#         data[data.columns] = scaler.fit_transform(data[data.columns])
        
#         outcome_vars = ['OS']
#         original_columns = data.columns
        
#         for outcome in outcome_vars:
#             data_copy = data.copy()
#             X = data_copy.drop(outcome, axis=1)
#             y = data_copy[outcome].astype(int)
            
#             smote = SMOTE(random_state=random_state)
#             X_resampled, y_resampled = smote.fit_resample(X, y)
            
#             data_copy = pd.concat([X_resampled, y_resampled], axis=1)
            
#             # Reorder columns to match the original order
#             data_copy = data_copy[original_columns]
        
#         # Debug: Print unique values of the 'OS' variable after preprocessing
#         print("Unique values of 'OS' variable after preprocessing:")
#         print(data_copy['OS'].unique())
        
#         return data_copy
    
#     except FileNotFoundError:
#         print(f"Error: The file '{data_path}' could not be found.")
#         raise
#     except KeyError as e:
#         print(f"Error: The column '{e.args[0]}' does not exist in the dataset.")
#         raise
#     except Exception as e:
#         print(f"Error: An unexpected error occurred during preprocessing: {e}")
#         raise

# def select_features(data, outcome_vars, k=10):
#     X = data.drop(outcome_vars, axis=1)
#     selected_features = {}
    
#     for outcome in outcome_vars:
#         if outcome not in data.columns:
#             raise KeyError(f"The column '{outcome}' does not exist in the dataset.")
        
#         y = data[outcome]
        
#         selector = SelectKBest(score_func=f_classif, k=k)
#         selector.fit(X, y)
        
#         selected_columns = X.columns[selector.get_support()].tolist()
#         selected_features[outcome] = selected_columns
    
#     return selected_features

# if __name__ == "__main__":
#     data_path = "synthetic_dataset.csv"
#     sample_size = 1500
#     processed_data = preprocess_data(data_path, sample_size=sample_size)
#     print("Data preprocessing completed.")