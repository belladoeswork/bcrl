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

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def preprocess_data(data_path):
    try:
        # Load the data
        data = pd.read_csv(data_path)
        
        # Handle missing data
        # data.fillna(data.median(), inplace=True)  # Numerical variables
        data.fillna(data.mode().iloc[0], inplace=True)  # Categorical variables

        # Encode categorical variables
        label_encoder = LabelEncoder()
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            data[col] = label_encoder.fit_transform(data[col])

        # Scale the features
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data[data.columns] = scaler.fit_transform(data[data.columns])

        # Address class imbalance for the survival outcome
        X = data.drop(['OS'], axis=1)  # Features
        y = data['OS']  # Target variable (survival)

        # Oversample minority class
        oversampler = RandomOverSampler(sampling_strategy='minority')
        X_over, y_over = oversampler.fit_resample(X, y)

        # Undersample majority class
        undersampler = RandomUnderSampler(sampling_strategy='majority')
        X_balanced, y_balanced = undersampler.fit_resample(X_over, y_over)

        balanced_data = pd.concat([X_balanced, y_balanced], axis=1)

        return balanced_data

    except FileNotFoundError:
        print(f"Error: The file '{data_path}' could not be found.")
    except KeyError as e:
        print(f"Error: The column '{e.args[0]}' does not exist in the dataset.")
    except Exception as e:
        print(f"Error: An unexpected error occurred during preprocessing: {e}")
        raise
    
    
if __name__ == "__main__":
    data_path = "dataset.csv"
    processed_data = preprocess_data(data_path)
    print("Data preprocessing completed.")