# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler

# def preprocess_data(data_path):
#     # Load the data
#     data = pd.read_csv(data_path)

#     # Handle missing data
#     # data.fillna(data.median(), inplace=True)  # Numerical variables
#     # data.fillna(data.mode().iloc[0], inplace=True)  # Categorical variables

#     # Encode categorical variables
#     label_encoder = LabelEncoder()
#     categorical_columns = data.select_dtypes(include=['object']).columns
#     for col in categorical_columns:
#         data[col] = label_encoder.fit_transform(data[col])

#     # Scale the features
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     data[data.columns] = scaler.fit_transform(data[data.columns])

#     # Address class imbalance for each target variable
#     balanced_data = {}
#     for target in ['surgery', 'chemotherapy', 'hormone_therapy', 'radiotherapy']:
#         X = data.drop([target], axis=1)  # Features
#         y = data[target]  # Target variable

#         # Oversample minority classes
#         oversampler = RandomOverSampler(sampling_strategy='minority')
#         X_over, y_over = oversampler.fit_resample(X, y)

#         # Undersample majority classes
#         undersampler = RandomUnderSampler(sampling_strategy='majority')
#         X_balanced, y_balanced = undersampler.fit_resample(X_over, y_over)

#         balanced_data[target] = pd.concat([X_balanced, y_balanced], axis=1)

#     return balanced_data


# preprocessing.py version 1.0

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler
# from sklearn.model_selection import StratifiedShuffleSplit


# def preprocess_data(data_path, sample_size=None, random_state=42):
#     try:
#         # Load the data
#         data = pd.read_csv(data_path)
        
#         # Handle missing data
#         # data.fillna(data.median(), inplace=True)  # Numerical variables
#         data.fillna(data.mode().iloc[0], inplace=True)  # Categorical variables

#         # Encode categorical variables
#         label_encoder = LabelEncoder()
#         categorical_columns = data.select_dtypes(include=['object']).columns
#         for col in categorical_columns:
#             data[col] = label_encoder.fit_transform(data[col])

#         # Scale the features
#         scaler = MinMaxScaler(feature_range=(-1, 1))
#         data[data.columns] = scaler.fit_transform(data[data.columns])

#         # Address class imbalance for the survival outcome
#         X = data.drop(['OS'], axis=1)  # Features
#         y = data['OS']  # Target variable (survival)

#         # Oversample minority class
#         oversampler = RandomOverSampler(sampling_strategy='minority')
#         X_over, y_over = oversampler.fit_resample(X, y)

#         # Undersample majority class
#         undersampler = RandomUnderSampler(sampling_strategy='majority')
#         X_balanced, y_balanced = undersampler.fit_resample(X_over, y_over)

#         balanced_data = pd.concat([X_balanced, y_balanced], axis=1)

#         return balanced_data

#     except FileNotFoundError:
#         print(f"Error: The file '{data_path}' could not be found.")
#     except KeyError as e:
#         print(f"Error: The column '{e.args[0]}' does not exist in the dataset.")
#     except Exception as e:
#         print(f"Error: An unexpected error occurred during preprocessing: {e}")
#         raise
    
    
# if __name__ == "__main__":
#     data_path = "dataset.csv"
#     processed_data = preprocess_data(data_path)
#     print("Data preprocessing completed.")



# preprocessing.py

# import pandas as pd
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.feature_selection import SelectKBest, f_classif



# # def preprocess_data(data_path, sample_size=None, random_state=42):
# #     try:
# #         # Load the data
# #         data = pd.read_csv(data_path)

# #         if sample_size is not None:
# #             # Perform stratified sampling
# #             outcome_vars = ['OS', 'RFS', 'DFS']
# #             stratify_data = data[outcome_vars].apply(lambda x: ''.join(x.astype(str)), axis=1)
# #             split = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=random_state)
# #             for _, sample_indices in split.split(data, stratify_data):
# #                 data = data.loc[sample_indices]

# #         # Handle missing data
# #         data.fillna(data.mode().iloc[0], inplace=True)  # Categorical variables

# #         # Encode categorical variables
# #         label_encoder = LabelEncoder()
# #         categorical_columns = data.select_dtypes(include=['object']).columns
# #         for col in categorical_columns:
# #             data[col] = label_encoder.fit_transform(data[col])

# #         # Scale the features
# #         scaler = MinMaxScaler(feature_range=(-1, 1))
# #         data[data.columns] = scaler.fit_transform(data[data.columns])

# #         # Handle class imbalance
# #         outcome_vars = ['OS', 'RFS', 'DFS']
# #         for outcome in outcome_vars:
# #             X = data.drop(outcome_vars, axis=1)  # Features
# #             y = data[outcome]
# #             oversampler = RandomOverSampler(random_state=random_state)
# #             X_resampled, y_resampled = oversampler.fit_resample(X, y)
# #             data = pd.concat([X_resampled, y_resampled], axis=1)
# #             data = data[list(X.columns) + [outcome]]


# #     except FileNotFoundError:
# #         print(f"Error: The file '{data_path}' could not be found.")
# #         raise
# #     except KeyError as e:
# #         print(f"Error: The column '{e.args[0]}' does not exist in the dataset.")
# #         raise
# #     except Exception as e:
# #         print(f"Error: An unexpected error occurred during preprocessing: {e}")
# #         raise

# #     return data

# # def select_features(data, outcome_vars, k=10):
# #     X = data.drop(outcome_vars, axis=1)  # Features
# #     selected_features = {}

# #     for outcome in outcome_vars:
# #         y = data[outcome]  # Target variable

# #         selector = SelectKBest(score_func=f_classif, k=k)
# #         selector.fit(X, y)

# #         selected_columns = X.columns[selector.get_support()].tolist()
# #         selected_features[outcome] = selected_columns

# #     return selected_features


# def preprocess_data(data_path, sample_size=None, random_state=42):
#     try:
#         # Load the data
#         data = pd.read_csv(data_path)

#         if sample_size is not None:
#             # Perform stratified sampling based on all outcome variables
#             outcome_vars = ['OS', 'RFS', 'DFS']
#             stratify_data = data[outcome_vars].apply(lambda x: ''.join(x.astype(str)), axis=1)
#             split = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=random_state)
#             for _, sample_indices in split.split(data, stratify_data):
#                 data = data.loc[sample_indices]

        
#         # Remove constant features
#         constant_columns = data.columns[data.nunique() <= 1]
#         data = data.drop(columns=constant_columns)
                
#         # Handle missing data
#         data.fillna(data.mode().iloc[0], inplace=True)  # Categorical variables

#         # Encode categorical variables
#         label_encoder = LabelEncoder()
#         categorical_columns = data.select_dtypes(include=['object']).columns
#         for col in categorical_columns:
#             data[col] = label_encoder.fit_transform(data[col])

#         # Scale the features
#         scaler = MinMaxScaler(feature_range=(-1, 1))
#         data[data.columns] = scaler.fit_transform(data[data.columns])

#         # Handle class imbalance
#         outcome_vars = ['OS', 'RFS', 'DFS']
#         original_columns = data.columns
#         for outcome in outcome_vars:
#             # print(f"Current columns in data: {data.columns}")
#             data_copy = data.copy() 
#             X = data_copy.drop(outcome, axis=1)  # Features
#             y = data_copy[outcome]
#             oversampler = RandomOverSampler(random_state=random_state)
#             X_resampled, y_resampled = oversampler.fit_resample(X, y)
#             data_copy = pd.concat([X_resampled, y_resampled], axis=1)

#         # Reorder columns to match the original order
#         data_copy = data_copy[original_columns]

#     except FileNotFoundError:
#         print(f"Error: The file '{data_path}' could not be found.")
#         raise
#     except KeyError as e:
#         print(f"Error: The column '{e.args[0]}' does not exist in the dataset.")
#         raise
#     except Exception as e:
#         print(f"Error: An unexpected error occurred during preprocessing: {e}")
#         raise

#     return data_copy

# def select_features(data, outcome_vars, k=10):
#     X = data.drop(outcome_vars, axis=1)  # Features
#     selected_features = {}

#     for outcome in outcome_vars:
#         if outcome not in data.columns:
#             raise KeyError(f"The column '{outcome}' does not exist in the dataset.")

#         y = data[outcome]  # Target variable

#         selector = SelectKBest(score_func=f_classif, k=k)
#         selector.fit(X, y)

#         selected_columns = X.columns[selector.get_support()].tolist()
#         selected_features[outcome] = selected_columns

#     return selected_features


# if __name__ == "__main__":
#     data_path = "dataset.csv"
#     sample_size = 10000
#     processed_data = preprocess_data(data_path, sample_size=sample_size)
#     print("Data preprocessing completed.")










import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif

def preprocess_data(data_path, sample_size=None, random_state=42):
    try:
        # Load the data
        data = pd.read_csv(data_path)
        if sample_size is not None:
            # Perform stratified sampling based on all outcome variables
            outcome_vars = ['OS', 'RFS', 'DFS']
            stratify_data = data[outcome_vars].apply(lambda x: ''.join(x.astype(str)), axis=1)
            split = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=random_state)
            for _, sample_indices in split.split(data, stratify_data):
                data = data.loc[sample_indices]
        # Remove constant features
        constant_columns = data.columns[data.nunique() <= 1]
        data = data.drop(columns=constant_columns)
        # Handle missing data
        data.fillna(data.mode().iloc[0], inplace=True)  # Categorical variables
        # Encode categorical variables
        label_encoder = LabelEncoder()
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            data[col] = label_encoder.fit_transform(data[col])
        # Scale the features
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data[data.columns] = scaler.fit_transform(data[data.columns])
        # Handle class imbalance
        outcome_vars = ['OS', 'RFS', 'DFS']
        original_columns = data.columns
        for outcome in outcome_vars:
            data_copy = data.copy()
            X = data_copy.drop(outcome, axis=1)  # Features
            y = data_copy[outcome]
            oversampler = RandomOverSampler(random_state=random_state)
            X_resampled, y_resampled = oversampler.fit_resample(X, y)
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

    return data_copy

def select_features(data, outcome_vars, k=10):
    X = data.drop(outcome_vars, axis=1)  # Features
    selected_features = {}
    for outcome in outcome_vars:
        if outcome not in data.columns:
            raise KeyError(f"The column '{outcome}' does not exist in the dataset.")
        y = data[outcome]  # Target variable
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit(X, y)
        selected_columns = X.columns[selector.get_support()].tolist()
        selected_features[outcome] = selected_columns
    return selected_features

if __name__ == "__main__":
    data_path = "synthetic_dataset.csv"
    sample_size = 500
    processed_data = preprocess_data(data_path, sample_size=sample_size)
    print("Data preprocessing completed.")