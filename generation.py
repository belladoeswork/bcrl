import pandas as pd
import numpy as np

def generate_synthetic_data(num_samples):
    feature_names = ['biopsy_preTreat', 'pCR_postTrt_days', 'tumor_size_cm_preTrt_preSurgery',
                     'tumor_size_cm_secondAxis_preTrt_preSurgery', 'tumor_size_cm_postTrt',
                     'preTrt_totalLymphNodes', 'preTrt_numPosLymphNodes', 'postTrt_numPosLymphNodes',
                     'preTrt_posDichLymphNodes', 'hist_grade', 'nuclear_grade_preTrt', 'pCR', 'near_pCR',
                     'RFS', 'DFS', 'OS', 'metastasis', 'metastasis_months', 'died_from_cancer_if_dead', 'age',
                     'ER_preTrt', 'ER_fmolmg_preTrt', 'ESR1_preTrt', 'ERbb2_preTrt', 'Erbeta_preTrt',
                     'ERBB2_CPN_amplified', 'PR_preTrt', 'PR_percentage_preTrt', 'PR_fmolmg_preTrt',
                     'HER2_preTrt', 'HER2_fish_cont_score_preTrt', 'cytokeratin5_pos', 'top2atri_preTrt',
                     'topoihc_preTrt', 'S_phase', 'radiotherapy', 'postmenopausal_only', 'anthracycline',
                     'taxane', 'anti_estrogen', 'aromatase_inhibitor', 'anti_HER2', 'tamoxifen', 'doxorubicin',
                     'epirubicin', 'docetaxel', 'capecitabine', 'fluorouracil', 'paclitaxel', 'cyclophosphamide',
                     'anastrozole', 'fulvestrant', 'gefitinib', 'trastuzumab', 'letrozole', 'chemotherapy',
                     'hormone_therapy', 'methotrexate', 'cetuximab', 'carboplatin', 'other_treatment',
                     'tumor_size_cm_postTrt_1', 'postTrt_numPosLymphNodes_1', 'metastasis_1',
                     'metastasis_months_1', 'tumor_size_cm_postTrt_2', 'postTrt_numPosLymphNodes_2',
                     'metastasis_2', 'metastasis_months_2', 'clinical_AJCC_stage', 'treatment_admin',
                     'Race', 'preTrt_lymph_node_status', 'postTrt_lymph_node_status', 'tumor_stage_postTrt',
                     'tumor_stage_preTrt', 'pam50', 'pCR_spectrum', 'RCB', 'menopausal_status',
                     'HER2_IHC_score_preTrt', 'ploidy', 'estrogen_receptor', 'surgery', 'therapy']
    
    # Generate synthetic data
    data = {}
    data['OS'] = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])

    # Generate treatment decisions based on OS and other features
    data['surgery'] = np.where(data['OS'] == 1, np.random.choice([0, 1], size=num_samples, p=[0.1, 0.9]),
                                                 np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1]))
    data['chemotherapy'] = np.where((data['OS'] == 1) & (data['surgery'] == 1), np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8]),
                                                                                 np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2]))
    data['hormone_therapy'] = np.where((data['OS'] == 1) & (data['surgery'] == 1) & (data['chemotherapy'] == 1), np.random.choice([0, 1], size=num_samples, p=[0.1, 0.9]),
                                                                                                                 np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1]))
    data['radiotherapy'] = np.where((data['OS'] == 1) & (data['surgery'] == 1) & (data['chemotherapy'] == 1) & (data['hormone_therapy'] == 1), np.random.choice([0, 1], size=num_samples, p=[0.1, 0.9]),
                                                                                                                                              np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1]))

    # for feature in feature_names:
    #     # if feature in ['OS', 'RFS', 'DFS']:
    #     #     data[feature] = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
    #     if feature in ['surgery', 'chemotherapy', 'hormone_therapy', 'radiotherapy']:
    #         # data[feature] = np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])
    #         data[feature] = np.where(data['OS'] == 1, np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8]),
    #                                                    np.random.choice([0, 1], size=num_samples, p=[0.6, 0.4]))
            
    #     elif feature in ['preTrt_numPosLymphNodes', 'postTrt_numPosLymphNodes', 'metastasis_months', 'metastasis_months_1', 'metastasis_months_2']:
    #         data[feature] = np.where(data['OS'] == 1, np.random.randint(0, 5, num_samples),
    #                                                    np.random.randint(5, 15, num_samples))
    #     elif feature in ['hist_grade', 'nuclear_grade_preTrt']:
    #         data[feature] = np.where(data['OS'] == 1, np.random.choice([1, 2], size=num_samples),
    #                                                    np.random.choice([2, 3], size=num_samples))
    #     elif feature in ['pCR', 'near_pCR', 'metastasis', 'died_from_cancer_if_dead']:
    #         data[feature] = np.where(data['OS'] == 1, np.random.choice(['Y', 'N'], size=num_samples, p=[0.8, 0.2]),
    #                                                    np.random.choice(['Y', 'N'], size=num_samples, p=[0.2, 0.8]))
    #     elif feature == 'age':
    #         data[feature] = np.where(data['OS'] == 1, np.random.randint(40, 70, num_samples),
    #                                                    np.random.randint(60, 90, num_samples))
    #     elif feature == 'tumor_size_cm_preTrt_preSurgery':
    #         data[feature] = np.where(data['OS'] == 1, np.random.randint(0, 5, num_samples),
    #                                                    np.random.randint(5, 10, num_samples))
    #     elif feature == 'tumor_size_cm_postTrt':
    #         data[feature] = np.where(data['OS'] == 1, np.random.randint(0, 2, num_samples),
    #                                                    np.random.randint(2, 5, num_samples))
    #     elif feature == 'topoihc_preTrt':
    #         data[feature] = np.where(data['OS'] == 1, np.random.randint(0, 50, num_samples),
    #                                                    np.random.randint(50, 100, num_samples))
    #     elif feature == 'S_phase':
    #         data[feature] = np.where(data['OS'] == 1, np.random.randint(1, 20, num_samples),
    #                                                    np.random.randint(20, 40, num_samples))
    for feature in feature_names:
        if feature in ['preTrt_numPosLymphNodes', 'postTrt_numPosLymphNodes', 'metastasis_months', 'metastasis_months_1', 'metastasis_months_2']:
            data[feature] = np.where((data['OS'] == 1) & (data['surgery'] == 1) & (data['chemotherapy'] == 1) & (data['hormone_therapy'] == 1) & (data['radiotherapy'] == 1),
                                     np.random.randint(0, 3, num_samples),
                                     np.random.randint(10, 20, num_samples))
        elif feature in ['hist_grade', 'nuclear_grade_preTrt']:
            data[feature] = np.where((data['OS'] == 1) & (data['surgery'] == 1) & (data['chemotherapy'] == 1) & (data['hormone_therapy'] == 1) & (data['radiotherapy'] == 1),
                                     np.random.choice([1, 2], size=num_samples, p=[0.9, 0.1]),
                                     np.random.choice([2, 3], size=num_samples, p=[0.1, 0.9]))
        elif feature in ['pCR', 'near_pCR', 'metastasis', 'died_from_cancer_if_dead']:
            data[feature] = np.where((data['OS'] == 1) & (data['surgery'] == 1) & (data['chemotherapy'] == 1) & (data['hormone_therapy'] == 1) & (data['radiotherapy'] == 1),
                                     np.random.choice(['Y', 'N'], size=num_samples, p=[0.9, 0.1]),
                                     np.random.choice(['Y', 'N'], size=num_samples, p=[0.1, 0.9]))
        elif feature == 'age':
            data[feature] = np.where((data['OS'] == 1) & (data['surgery'] == 1) & (data['chemotherapy'] == 1) & (data['hormone_therapy'] == 1) & (data['radiotherapy'] == 1),
                                     np.random.randint(30, 60, num_samples),
                                     np.random.randint(70, 90, num_samples))
        elif feature == 'tumor_size_cm_preTrt_preSurgery':
            data[feature] = np.where((data['OS'] == 1) & (data['surgery'] == 1) & (data['chemotherapy'] == 1) & (data['hormone_therapy'] == 1) & (data['radiotherapy'] == 1),
                                     np.random.randint(0, 3, num_samples),
                                     np.random.randint(7, 12, num_samples))
        elif feature == 'tumor_size_cm_postTrt':
            data[feature] = np.where((data['OS'] == 1) & (data['surgery'] == 1) & (data['chemotherapy'] == 1) & (data['hormone_therapy'] == 1) & (data['radiotherapy'] == 1),
                                     np.random.randint(0, 1, num_samples),
                                     np.random.randint(3, 6, num_samples))
        elif feature == 'topoihc_preTrt':
            data[feature] = np.where((data['OS'] == 1) & (data['surgery'] == 1) & (data['chemotherapy'] == 1) & (data['hormone_therapy'] == 1) & (data['radiotherapy'] == 1),
                                     np.random.randint(0, 30, num_samples),
                                     np.random.randint(70, 100, num_samples))
        elif feature == 'S_phase':
            data[feature] = np.where((data['OS'] == 1) & (data['surgery'] == 1) & (data['chemotherapy'] == 1) & (data['hormone_therapy'] == 1) & (data['radiotherapy'] == 1),
                                     np.random.randint(1, 15, num_samples),
                                     np.random.randint(25, 40, num_samples))
#         elif feature in ['biopsy_preTreat', 'pCR', 'near_pCR', 'metastasis',  'died_from_cancer_if_dead', 'ER_preTrt', 'ESR1_preTrt',  'ERbb2_preTrt', 'Erbeta_preTrt', 'ERBB2_CPN_amplified', 'PR_preTrt', 'HER2_preTrt', 'cytokeratin5_pos',  
# 'anthracycline','taxane', 'anti_estrogen', 'aromatase_inhibitor', 'anti_HER2',  'doxorubicin', 'epirubicin', 'docetaxel', 'capecitabine', 'fluorouracil', 'paclitaxel', 'cyclophosphamide',
# 'anastrozole', 'fulvestrant', 'gefitinib', 'trastuzumab', 'letrozole', 'methotrexate', 'cetuximab', 'carboplatin', 'other_treatment', 'metastasis_1', 'metastasis_2']:
#             data[feature] = np.random.choice(['Y', 'N'], size=num_samples)
        elif feature in ['biopsy_preTreat', 'ER_preTrt', 'ESR1_preTrt',  'ERbb2_preTrt', 'Erbeta_preTrt', 'ERBB2_CPN_amplified', 'PR_preTrt', 'HER2_preTrt', 'cytokeratin5_pos',  
        'anthracycline','taxane', 'anti_estrogen', 'aromatase_inhibitor', 'anti_HER2',  'doxorubicin', 'epirubicin', 'docetaxel', 'capecitabine', 'fluorouracil', 'paclitaxel', 'cyclophosphamide',
        'anastrozole', 'fulvestrant', 'gefitinib', 'trastuzumab', 'letrozole', 'methotrexate', 'cetuximab', 'carboplatin', 'other_treatment', 'metastasis_1', 'metastasis_2']:
            data[feature] = np.random.choice(['Y', 'N'], size=num_samples)
        elif feature in ['top2atri_preTrt']: 
            data[feature] = np.random.choice([-1, 0, 1], size=num_samples)
        elif feature in ['tumor_size_cm_postTrt_2', 'tumor_size_cm_postTrt_1']:
            data[feature] = np.random.choice([0, 1, 2], size=num_samples)
        elif feature in ['preTrt_posDichLymphNodes', 'tamoxifen']:
            data[feature] = np.random.choice([0, 1], size=num_samples)
        # elif feature in ['hist_grade', 'nuclear_grade_preTrt']:
        #     data[feature] = np.random.choice([1, 2, 3], size=num_samples)
        # elif feature in ['age']:
        #     data[feature] = np.random.randint(20, 90, num_samples)
        elif feature in ['postmenopausal_only']:
            data[feature] = np.where(data['age'] >= 45, 1, 0)
        elif feature in ['menopausal_status']:
            data[feature] = np.where(data['postmenopausal_only'] == 1, 'post', 'pre')
        elif feature in ['ER_fmolmg_preTrt']:
            data[feature] = np.random.randint(0, 500, num_samples)
        # elif feature in ['S_phase']:
        #     data[feature] = np.random.randint(1, 40, num_samples)
        # elif feature in ['topoihc_preTrt']:
        #     data[feature] = np.random.randint(0, 100, num_samples)
        elif feature in ['HER2_fish_cont_score_preTrt']:
            data[feature] = np.random.randint(0, 20, num_samples)
        elif feature in ['postTrt_numPosLymphNodes_1']:
            data[feature] = np.random.randint(1, 15, num_samples)
        elif feature in ['postTrt_numPosLymphNodes_2']:
            data[feature] = np.random.randint(0, 15, num_samples)
        elif feature in ['pCR_postTrt_days', 'PR_percentage_preTrt']:
            data[feature] = np.random.randint(0, 100, num_samples)
        # elif feature in ['metastasis_months', 'metastasis_months_1', 'metastasis_months_2']:
        #     data[feature] = np.random.randint(1, 100, num_samples)
        # elif feature in ['tumor_size_cm_preTrt_preSurgery', 'tumor_size_cm_secondAxis_preTrt_preSurgery','tumor_size_cm_postTrt']:
        #     data[feature] = np.random.randint(0, 25, num_samples)
        elif feature in [ 'tumor_size_cm_secondAxis_preTrt_preSurgery']:
            data[feature] = np.random.randint(0, 25, num_samples)
        # elif feature in ['preTrt_totalLymphNodes', 'preTrt_numPosLymphNodes', 'postTrt_numPosLymphNodes', 'PR_fmolmg_preTrt']:
        #     data[feature] = np.random.randint(0, 500, num_samples) //hi
        elif feature in ['preTrt_totalLymphNodes', 'preTrt_numPosLymphNodes', 'PR_fmolmg_preTrt']:
            data[feature] = np.random.randint(0, 500, num_samples)
        elif feature in ['clinical_AJCC_stage']:
            data[feature] = np.random.choice(['I', 'II', 'III',' IV', 'IIB', 'IIA',  'IIIA', 'IIIB', 'IIIC '], size=num_samples)
        elif feature in ['Race']:
            data[feature] = np.random.choice(['White', 'Hispanic', 'Black ', 'Asian', 'Mixed'], size=num_samples)
        elif feature in ['treatment_admin']:
            data[feature] = np.random.choice(['oral','NOS','intarvenous'], size=num_samples)
        elif feature in ['preTrt_lymph_node_status']:
            data[feature] = np.random.choice(['N0', 'N1', 'positive',' N2', 'N3',' ND'], size=num_samples)
        elif feature in ['postTrt_lymph_node_status']:
            data[feature] = np.random.choice(['N0', 'N1', 'positive', 'N2', 'N3', 'ND', 'N1a', 'N2a'], size=num_samples)
        elif feature in ['tumor_stage_postTrt']:
            data[feature] = np.random.choice(['T1c', 'T0', 'T1b', 'T1a', 'T2', 'Tis', 'T1', 'T1mic'], size=num_samples)
        elif feature in ['tumor_stage_preTrt']:
            data[feature] = np.random.choice(['T2', 'T1', 'T3','T4', 'T0'], size=num_samples)    
        elif feature in ['pam50']:
            data[feature] = np.random.choice(['Basal', 'Luminal A', 'Luminal B', 'Her2', 'Normal' 'Claudin'], size=num_samples)
        elif feature in ['pCR_spectrum']:
            data[feature] = np.random.choice(['PR', 'CR', 'NCR', 'PD', 'SD', 'EPD', 'NE'], size=num_samples)
        elif feature in ['RCB']:
            data[feature] = np.random.choice(['II', '0/I', '2', 'III', '3', '0', '1'], size=num_samples)
        elif feature in ['HER2_IHC_score_preTrt']:
            data[feature] = np.random.choice(['NEG' ,0 ,1 ,2, 3 ,'ND'], size=num_samples)
        elif feature in ['ploidy']:
            data[feature] = np.random.choice(['aneuploid', 'diploid', 'multiploid'], size=num_samples)
        elif feature in ['estrogen_receptor']:
            data[feature] = np.random.choice(['NOS','block_and_stop' ,'block_and_eliminate'], size=num_samples)
        elif feature in ['therapy']:
            data[feature] = np.random.choice(['neo' ,'adj' ,'mixed' ,'unspecified'], size=num_samples)
        elif feature in ['surgery', 'chemotherapy', 'hormone_therapy', 'radiotherapy']:
            data[feature] = np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])
        elif feature in ['surgery', 'chemotherapy', 'hormone_therapy', 'radiotherapy']:
            data[feature] = np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])
        elif feature in ['surgery', 'chemotherapy', 'hormone_therapy', 'radiotherapy']:
            data[feature] = np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])
        elif feature in ['surgery', 'chemotherapy', 'hormone_therapy', 'radiotherapy']:
            data[feature] = np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])
        else:
            # data[feature] = np.random.rand(num_samples)
            data[feature] = np.random.choice(2, size=num_samples)
    
    # Create a DataFrame from the generated data
    df = pd.DataFrame(data)
    
    return df

# Generate the synthetic dataset
synthetic_data = generate_synthetic_data(num_samples=1000)
synthetic_data.to_csv('synthetic_dataset.csv', index=False)
print("Synthetic dataset generated and saved as 'synthetic_dataset.csv'.")
