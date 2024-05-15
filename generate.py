import numpy as np
import pandas as pd

def generate_synthetic_data(num_samples):
    feature_names = ['age', 'menopausal_status', 'tumor_size', 'lymph_nodes', 'stage', 'histology', 'grade',
                     'er_status', 'pr_status', 'her2_status', 'ki67_expression', 'oncotype_score',
                     'surgery', 'tumor_size_post_surgery', 'lymph_nodes_post_surgery',
                     'chemotherapy', 'tumor_size_post_chemo', 'lymph_nodes_post_chemo', 'recurrence_post_chemo',
                     'hormone_therapy', 'tumor_size_post_hormone', 'lymph_nodes_post_hormone', 'recurrence_post_hormone',
                     'radiotherapy', 'tumor_size_post_radio', 'lymph_nodes_post_radio', 'recurrence_post_radio',
                     'OS', 'DFS', 'RFS']

    data = {}

    # Pre-treatment features
    data['age'] = np.random.normal(loc=55, scale=10, size=num_samples).astype(int)
    data['menopausal_status'] = np.where(data['age'] > 50, 'Post', 'Pre')
    data['stage'] = np.random.choice([1, 2, 3, 4], size=num_samples, p=[0.4, 0.3, 0.2, 0.1])
    data['tumor_size'] = np.where(data['stage'] < 3, np.random.normal(loc=1.5, scale=0.5, size=num_samples), np.random.normal(loc=4.5, scale=1.0, size=num_samples)).clip(0.1, None)
    data['lymph_nodes'] = np.where(data['stage'] < 3, np.random.binomial(n=3, p=0.2, size=num_samples), np.random.binomial(n=10, p=0.6, size=num_samples))
    data['histology'] = np.random.choice(['Ductal', 'Lobular', 'Mixed', 'Other'], size=num_samples, p=[0.8, 0.1, 0.05, 0.05])
    data['grade'] = np.where(data['stage'] < 3, np.random.choice([1, 2, 3], size=num_samples, p=[0.3, 0.5, 0.2]), np.random.choice([1, 2, 3], size=num_samples, p=[0.1, 0.3, 0.6]))
    data['er_status'] = np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.8, 0.2])
    data['pr_status'] = np.where(data['er_status'] == 'Positive', np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.9, 0.1]), np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.1, 0.9]))
    data['her2_status'] = np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.2, 0.8])
    data['ki67_expression'] = np.where(data['grade'] < 3, np.random.normal(loc=15, scale=5, size=num_samples).clip(0, 100), np.random.normal(loc=40, scale=10, size=num_samples).clip(0, 100))
    data['oncotype_score'] = np.where((data['er_status'] == 'Positive') & (data['grade'] < 3), np.random.normal(loc=10, scale=5, size=num_samples).clip(0, 100), np.random.normal(loc=30, scale=10, size=num_samples).clip(0, 100))

    data['surgery'] = np.where((data['stage'] <= 2) & (data['tumor_size'] <= 2) & (data['lymph_nodes'] <= 1) & (data['er_status'] == 'Positive') & (data['her2_status'] == 'Negative') & (data['ki67_expression'] <= 20), 'Lumpectomy', 'Mastectomy')
    data['surgery'] = np.where((data['age'] >= 40) & (data['age'] <= 70) & (data['stage'] == 1) & (data['tumor_size'] <= 2) & (data['lymph_nodes'] == 0) & (data['er_status'] == 'Positive') & (data['her2_status'] == 'Negative'), 'Lumpectomy', data['surgery'])
    data['surgery'] = np.where((data['er_status'] == 'Negative') | (data['her2_status'] == 'Positive') | (data['age'] < 40) | (data['age'] > 70) | (data['stage'] > 2) | (data['grade'] > 2) | (data['ki67_expression'] > 20), 'Mastectomy', data['surgery'])
    
    df = pd.DataFrame(data)
    
    # Adjust the distribution of surgery decisions to meet the criteria
    surgery_dist = pd.Series(df['surgery']).value_counts(normalize=True)
    if surgery_dist['Lumpectomy'] < 0.85:
        lumpectomy_indices = np.where(df['surgery'] == 'Lumpectomy')[0]
        mastectomy_indices = np.where(df['surgery'] == 'Mastectomy')[0]
        num_convert = int((0.85 - surgery_dist['Lumpectomy']) * num_samples)
        convert_indices = np.random.choice(mastectomy_indices, size=num_convert, replace=False)
        df.loc[convert_indices, 'surgery'] = 'Lumpectomy'

    # Post-surgery features
    data['tumor_size_post_surgery'] = np.where(data['surgery'] == 'Lumpectomy', data['tumor_size'] * np.random.normal(loc=0.1, scale=0.05, size=num_samples), 0.0)
    data['lymph_nodes_post_surgery'] = np.where(data['surgery'] == 'Lumpectomy', data['lymph_nodes'] * np.random.binomial(n=1, p=0.1, size=num_samples), 0)

    # Treatment decision 2: Chemotherapy
    data['chemotherapy'] = np.where((data['stage'] > 1) | (data['grade'] > 2) | (data['ki67_expression'] > 20) | (data['er_status'] == 'Negative') | (data['her2_status'] == 'Positive'), 'Yes', 'No')

    # Post-chemotherapy features
    data['tumor_size_post_chemo'] = np.where(data['chemotherapy'] == 'Yes', data['tumor_size_post_surgery'] * np.random.normal(loc=0.3, scale=0.1, size=num_samples), data['tumor_size_post_surgery'])
    data['lymph_nodes_post_chemo'] = np.where(data['chemotherapy'] == 'Yes', data['lymph_nodes_post_surgery'] * np.random.binomial(n=1, p=0.3, size=num_samples), data['lymph_nodes_post_surgery'])
    data['recurrence_post_chemo'] = np.where(data['chemotherapy'] == 'Yes', np.random.binomial(n=1, p=0.05, size=num_samples), np.random.binomial(n=1, p=0.2, size=num_samples))

    # Treatment decision 3: Hormone therapy
    data['hormone_therapy'] = np.where((data['er_status'] == 'Positive') | (data['pr_status'] == 'Positive'), 'Yes', 'No')

    # Post-hormone therapy features
    data['tumor_size_post_hormone'] = np.where(data['hormone_therapy'] == 'Yes', data['tumor_size_post_chemo'] * np.random.normal(loc=0.8, scale=0.1, size=num_samples), data['tumor_size_post_chemo'])
    data['lymph_nodes_post_hormone'] = np.where(data['hormone_therapy'] == 'Yes', data['lymph_nodes_post_chemo'] * np.random.binomial(n=1, p=0.9, size=num_samples), data['lymph_nodes_post_chemo'])
    data['recurrence_post_hormone'] = np.where(data['hormone_therapy'] == 'Yes', np.random.binomial(n=1, p=0.02, size=num_samples), np.random.binomial(n=1, p=0.1, size=num_samples))

    data['radiotherapy'] = np.where((data['surgery'] == 'Lumpectomy') | ((data['stage'] > 1) & ((data['lymph_nodes_post_hormone'] > 0) | (data['tumor_size_post_hormone'] > 1))), 'Yes', 'No')
    data['radiotherapy'] = np.where((data['lymph_nodes_post_hormone'] > 0) | (data['tumor_size_post_hormone'] > 1) | (data['lymph_nodes'] > 0), 'Yes', data['radiotherapy'])
    data['radiotherapy'] = np.where((data['stage'] > 2) & (data['surgery'] == 'Mastectomy') & (data['tumor_size'] > 3) & (data['lymph_nodes'] > 0), 'Yes', data['radiotherapy'])

    # Adjust the distribution of radiotherapy decisions to meet the criteria
    radiotherapy_dist = pd.Series(data['radiotherapy']).value_counts(normalize=True)
    if radiotherapy_dist['Yes'] < 0.95:
        no_indices = np.where(data['radiotherapy'] == 'No')[0]
        yes_indices = np.where(data['radiotherapy'] == 'Yes')[0]
        num_convert = int((0.95 - radiotherapy_dist['Yes']) * num_samples)
        convert_indices = np.random.choice(no_indices, size=num_convert, replace=False)
        data['radiotherapy'][convert_indices] = 'Yes'

    # Post-radiotherapy features
    data['tumor_size_post_radio'] = np.where(data['radiotherapy'] == 'Yes', data['tumor_size_post_hormone'] * np.random.normal(loc=0.9, scale=0.05, size=num_samples), data['tumor_size_post_hormone'])
    data['lymph_nodes_post_radio'] = np.where(data['radiotherapy'] == 'Yes', data['lymph_nodes_post_hormone'] * np.random.binomial(n=1, p=0.95, size=num_samples), data['lymph_nodes_post_hormone'])
    data['recurrence_post_radio'] = np.where(data['radiotherapy'] == 'Yes', np.random.binomial(n=1, p=0.01, size=num_samples), np.random.binomial(n=1, p=0.05, size=num_samples))

    # Outcomes
    data['OS'] = np.where((data['stage'] < 3) & (data['recurrence_post_radio'] == 0), np.random.binomial(n=1, p=0.98, size=num_samples), np.random.binomial(n=1, p=0.7, size=num_samples))
    data['DFS'] = np.where((data['stage'] < 3) & (data['recurrence_post_radio'] == 0), np.random.binomial(n=1, p=0.95, size=num_samples), np.random.binomial(n=1, p=0.6, size=num_samples))
    data['RFS'] = np.where((data['stage'] < 3) & (data['recurrence_post_radio'] == 0) & (data['recurrence_post_chemo'] == 0) & (data['recurrence_post_hormone'] == 0), np.random.binomial(n=1, p=0.95, size=num_samples), np.random.binomial(n=1, p=0.6, size=num_samples))

    # Create  DataFrame 
    df = pd.DataFrame(data)

    return df

if __name__ == "__main__":
    synthetic_data = generate_synthetic_data(num_samples=2000)
    synthetic_data.to_csv('synthetic_dataset.csv', index=False)
    print("Synthetic dataset generated and saved as 'synthetic_dataset.csv'.")