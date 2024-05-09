# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_selection import SelectKBest, mutual_info_classif

# def preprocess_data(data_path, sample_size=None, random_state=42):
#     try:
#         # Load the data
#         data = pd.read_csv(data_path)
        
#         if sample_size is not None:
#             # Perform stratified sampling based on all outcome variables
#             outcome_vars = ['OS', 'DFS', 'RFS']
#             stratify_data = data[outcome_vars].apply(lambda x: ''.join(x.astype(str)), axis=1)
            
#             skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)
#             train_index, _ = next(skf.split(data, stratify_data))
#             data = data.iloc[train_index]
        
#         # Remove constant features
#         constant_columns = data.columns[data.nunique() <= 1]
#         data = data.drop(columns=constant_columns)
        
        
#         # Handle missing data using KNNImputer
#         from sklearn.impute import KNNImputer
#         imputer = KNNImputer(n_neighbors=5)
#         data[data.columns] = imputer.fit_transform(data[data.columns])
        
#         # Encode categorical variables
#         from sklearn.preprocessing import LabelEncoder
#         label_encoder = LabelEncoder()
#         categorical_columns = data.select_dtypes(include=['object']).columns
#         for col in categorical_columns:
#             data[col] = label_encoder.fit_transform(data[col])
#         # Scale the features using standardization
#         scaler = StandardScaler()
#         data[data.columns] = scaler.fit_transform(data[data.columns])
        
#         outcome_vars = ['OS', 'DFS', 'RFS']
#         original_columns = data.columns
        
#         # Convert continuous outcome variables to categorical
#         # [outcome_vars] = pd.cut(data[outcome_vars], bins=3, labels=False)
        
#         for outcome in outcome_vars:
#             data_copy = data.copy()
#             X = data_copy.drop(outcome, axis=1)
#             data_copy[outcome] = pd.cut(data_copy[outcome], bins=3, labels=False)
#             # y = data_copy[outcome]
#             y = label_encoder.fit_transform(data_copy[outcome])
            
#             smote = SMOTE(random_state=random_state)
#             X_resampled, y_resampled = smote.fit_resample(X, y)
            
#             # data_copy = pd.concat([X_resampled, y_resampled], axis=1)
#             X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
#             y_resampled_series = pd.Series(y_resampled, name=outcome)

#             data_copy = pd.concat([X_resampled_df, y_resampled_series], axis=1)
            
#             # Reorder columns to match the original order
#             data_copy = data_copy[original_columns]
        
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
        
#         selector = SelectKBest(score_func=mutual_info_classif, k=k)
#         selector.fit(X, y)
        
#         selected_columns = X.columns[selector.get_support()].tolist()
#         selected_features[outcome] = selected_columns
    
#     return selected_features




import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data_path, sample_size=None, random_state=42):
    try:
        # Load the data
        data = pd.read_csv(data_path)
        
        if sample_size is not None:
            # Perform stratified sampling based on all outcome variables
            outcome_vars = ['OS', 'DFS', 'RFS']
            stratify_data = data[outcome_vars].apply(lambda x: ''.join(x.astype(str)), axis=1)
            
            skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)
            train_index, _ = next(skf.split(data, stratify_data))
            data = data.iloc[train_index]
        
        # Remove constant features
        constant_columns = data.columns[data.nunique() <= 1]
        data = data.drop(columns=constant_columns)

        # Identify categorical columns
        # categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
        # Encode categorical variables using one-hot encoding
        # data = pd.get_dummies(data, columns=categorical_columns)

        label_encoder = LabelEncoder()
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            data[col] = label_encoder.fit_transform(data[col])

        # Handle missing data using KNNImputer
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        data[data.columns] = imputer.fit_transform(data[data.columns])


        
        # Scale the features using standardization
        scaler = StandardScaler()
        data[data.columns] = scaler.fit_transform(data[data.columns])
        
        outcome_vars = ['OS', 'DFS', 'RFS']
        original_columns = data.columns
        
        # Convert continuous outcome variables to categorical using KBinsDiscretizer
        kbins_discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform', subsample=None)
        data[outcome_vars] = kbins_discretizer.fit_transform(data[outcome_vars]).astype(int)
        
        for outcome in outcome_vars:
            data_copy = data.copy()
            X = data_copy.drop(outcome, axis=1)
            y = data_copy[outcome]
            
            smote = SMOTE(random_state=random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
            y_resampled_series = pd.Series(y_resampled, name=outcome)
            data_copy = pd.concat([X_resampled_df, y_resampled_series], axis=1)
            
            # Reorder columns to match the original order
            data_copy = data_copy[original_columns]
        
        return data_copy
    
    except FileNotFoundError:
        print(f"Error: The file '{data_path}' could not be found.")
        raise
    except KeyError as e:
        print(f"Error: The column '{e.args[0]}' does not exist in the dataset.")
        raise
    except Exception as e:
        print(f"Error: An unexpected error occurred during preprocessing: {e}")
        raise

def select_features(data, outcome_vars, k=10):
    X = data.drop(outcome_vars, axis=1)
    selected_features = {}
    
    for outcome in outcome_vars:
        if outcome not in data.columns:
            raise KeyError(f"The column '{outcome}' does not exist in the dataset.")
        
        y = data[outcome]
        
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        selector.fit(X, y)
        
        selected_columns = X.columns[selector.get_support()].tolist()
        selected_features[outcome] = selected_columns
    
    return selected_features




# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
# from sklearn.model_selection import StratifiedKFold
# from sklearn.feature_selection import SelectKBest, mutual_info_classif
# from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder

# def preprocess_data(data_path, sample_size=None, random_state=42):
#     try:
#         # Load the data
#         data = pd.read_csv(data_path)
        
#         if sample_size is not None:
#             # Perform stratified sampling based on all outcome variables
#             outcome_vars = ['OS', 'DFS', 'RFS']
#             stratify_data = data[outcome_vars].apply(lambda x: ''.join(x.astype(str)), axis=1)
            
#             skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=random_state)
#             train_index, _ = next(skf.split(data, stratify_data))
#             data = data.iloc[train_index]
        
#         # Remove constant features
#         constant_columns = data.columns[data.nunique() <= 1]
#         data = data.drop(columns=constant_columns)

#         # Encode categorical variables
#         categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
        
#         # Handle missing data using KNNImputer
#         from sklearn.impute import KNNImputer
#         imputer = KNNImputer(n_neighbors=5)
#         data[data.columns] = imputer.fit_transform(data[data.columns])


#         data = pd.get_dummies(data, columns=categorical_columns)

        
#         # Encode categorical variables
#         # label_encoder = LabelEncoder()
#         # categorical_columns = data.select_dtypes(include=['object']).columns
#         # for col in categorical_columns:
#         #     data[col] = label_encoder.fit_transform(data[col])
        
#         # Scale the features using standardization
#         scaler = StandardScaler()
#         data[data.columns] = scaler.fit_transform(data[data.columns])
        
#         outcome_vars = ['OS', 'DFS', 'RFS']
#         original_columns = data.columns
        
#         # Convert continuous outcome variables to categorical using KBinsDiscretizer
#         kbins_discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform', subsample=None)
#         data[outcome_vars] = kbins_discretizer.fit_transform(data[outcome_vars])
        
#         # Encode the discretized outcome variables using LabelEncoder
#         # for outcome in outcome_vars:
#         #     data[outcome] = label_encoder.fit_transform(data[outcome].astype(str))
        
#         for outcome in outcome_vars:
#             data_copy = data.copy()
#             X = data_copy.drop(outcome, axis=1)
#             y = data_copy[outcome]
            
#             smote = SMOTE(random_state=random_state)
#             X_resampled, y_resampled = smote.fit_resample(X, y)
            
#             X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
#             y_resampled_series = pd.Series(y_resampled, name=outcome)
#             data_copy = pd.concat([X_resampled_df, y_resampled_series], axis=1)
            
#             # Reorder columns to match the original order
#             data_copy = data_copy[original_columns]
        
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
        
#         selector = SelectKBest(score_func=mutual_info_classif, k=k)
#         selector.fit(X, y)
        
#         selected_columns = X.columns[selector.get_support()].tolist()
#         selected_features[outcome] = selected_columns
    
#     return selected_features