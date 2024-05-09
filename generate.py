# import pandas as pd
# import numpy as np

# def generate_synthetic_data(num_samples):
#     feature_names = ['biopsy_preTreat', 'pCR_postTrt_days', 'tumor_size_cm_preTrt_preSurgery',
#                      'tumor_size_cm_secondAxis_preTrt_preSurgery', 'tumor_size_cm_postTrt',
#                      'preTrt_totalLymphNodes', 'preTrt_numPosLymphNodes', 'postTrt_numPosLymphNodes',
#                      'preTrt_posDichLymphNodes', 'hist_grade', 'nuclear_grade_preTrt', 'pCR', 'near_pCR',
#                      'RFS', 'DFS', 'OS', 'metastasis', 'metastasis_months', 'died_from_cancer_if_dead', 'age',
#                      'ER_preTrt', 'ER_fmolmg_preTrt', 'ESR1_preTrt', 'ERbb2_preTrt', 'Erbeta_preTrt',
#                      'ERBB2_CPN_amplified', 'PR_preTrt', 'PR_percentage_preTrt', 'PR_fmolmg_preTrt',
#                      'HER2_preTrt', 'HER2_fish_cont_score_preTrt', 'cytokeratin5_pos', 'top2atri_preTrt',
#                      'topoihc_preTrt', 'S_phase', 'radiotherapy', 'postmenopausal_only', 'anthracycline',
#                      'taxane', 'anti_estrogen', 'aromatase_inhibitor', 'anti_HER2', 'tamoxifen', 'doxorubicin',
#                      'epirubicin', 'docetaxel', 'capecitabine', 'fluorouracil', 'paclitaxel', 'cyclophosphamide',
#                      'anastrozole', 'fulvestrant', 'gefitinib', 'trastuzumab', 'letrozole', 'chemotherapy',
#                      'hormone_therapy', 'methotrexate', 'cetuximab', 'carboplatin', 'other_treatment',
#                      'tumor_size_cm_postTrt_1', 'postTrt_numPosLymphNodes_1', 'metastasis_1',
#                      'metastasis_months_1', 'tumor_size_cm_postTrt_2', 'postTrt_numPosLymphNodes_2',
#                      'metastasis_2', 'metastasis_months_2', 'clinical_AJCC_stage', 'treatment_admin',
#                      'Race', 'preTrt_lymph_node_status', 'postTrt_lymph_node_status', 'tumor_stage_postTrt',
#                      'tumor_stage_preTrt', 'pam50', 'pCR_spectrum', 'RCB', 'menopausal_status',
#                      'HER2_IHC_score_preTrt', 'ploidy', 'estrogen_receptor', 'therapy']

#     data = {}

#     # Generate OS
#     data['OS'] = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])

#     # Generate treatment decisions based on OS and important features
#     data['surgery'] = np.where((data['OS'] == 1) & (data['PR_percentage_preTrt'] > 0.5) & (data['metastasis_months_2'] < 5) & (data['preTrt_numPosLymphNodes'] < 3) & (data['PR_fmolmg_preTrt'] > 200) & (data['preTrt_totalLymphNodes'] < 10),
#                                 np.random.choice([0, 1], size=num_samples, p=[0.1, 0.9]),
#                                 np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1]))

#     data['chemotherapy'] = np.where((data['OS'] == 1) & (data['metastasis_months_2'] < 5) & (data['preTrt_numPosLymphNodes'] < 3) & (data['postTrt_numPosLymphNodes'] < 3) & (data['topoihc_preTrt'] < 50) & (data['age'] < 60),
#                                      np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8]),
#                                      np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2]))

#     data['hormone_therapy'] = np.where((data['OS'] == 1) & (data['ER_fmolmg_preTrt'] > 200) & (data['pCR_postTrt_days'] < 30) & (data['postTrt_numPosLymphNodes'] < 3) & (data['topoihc_preTrt'] < 50) & (data['metastasis_months_2'] < 5),
#                                         np.random.choice([0, 1], size=num_samples, p=[0.1, 0.9]),
#                                         np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1]))

#     data['radiotherapy'] = np.where((data['OS'] == 1) & (data['PR_fmolmg_preTrt'] > 200) & (data['age'] < 60) & (data['metastasis_months_2'] < 5) & (data['ER_fmolmg_preTrt'] > 200) & (data['preTrt_numPosLymphNodes'] < 3),
#                                     np.random.choice([0, 1], size=num_samples, p=[0.1, 0.9]),
#                                     np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1]))

#     # Generate important features for surgery
#     data['PR_percentage_preTrt'] = np.where(data['surgery'] == 1, np.random.randint(60, 100, num_samples), np.random.randint(0, 40, num_samples))
#     data['metastasis_months_2'] = np.where(data['surgery'] == 1, np.random.randint(0, 5, num_samples), np.random.randint(10, 20, num_samples))
#     data['preTrt_numPosLymphNodes'] = np.where(data['surgery'] == 1, np.random.randint(0, 3, num_samples), np.random.randint(5, 10, num_samples))
#     data['PR_fmolmg_preTrt'] = np.where(data['surgery'] == 1, np.random.randint(200, 500, num_samples), np.random.randint(0, 100, num_samples))
#     data['preTrt_totalLymphNodes'] = np.where(data['surgery'] == 1, np.random.randint(0, 10, num_samples), np.random.randint(15, 30, num_samples))

#     # Generate important features for chemotherapy
#     data['metastasis_months_2'] = np.where(data['chemotherapy'] == 1, np.random.randint(0, 5, num_samples), np.random.randint(10, 20, num_samples))
#     data['preTrt_numPosLymphNodes'] = np.where(data['chemotherapy'] == 1, np.random.randint(0, 3, num_samples), np.random.randint(5, 10, num_samples))
#     data['postTrt_numPosLymphNodes'] = np.where(data['chemotherapy'] == 1, np.random.randint(0, 3, num_samples), np.random.randint(5, 10, num_samples))
#     data['topoihc_preTrt'] = np.where(data['chemotherapy'] == 1, np.random.randint(0, 50, num_samples), np.random.randint(70, 100, num_samples))
#     data['age'] = np.where(data['chemotherapy'] == 1, np.random.randint(30, 60, num_samples), np.random.randint(70, 90, num_samples))

#     # Generate important features for hormone therapy
#     data['ER_fmolmg_preTrt'] = np.where(data['hormone_therapy'] == 1, np.random.randint(200, 500, num_samples), np.random.randint(0, 100, num_samples))
#     data['pCR_postTrt_days'] = np.where(data['hormone_therapy'] == 1, np.random.randint(0, 30, num_samples), np.random.randint(60, 100, num_samples))
#     data['postTrt_numPosLymphNodes'] = np.where(data['hormone_therapy'] == 1, np.random.randint(0, 3, num_samples), np.random.randint(5, 10, num_samples))
#     data['topoihc_preTrt'] = np.where(data['hormone_therapy'] == 1, np.random.randint(0, 50, num_samples), np.random.randint(70, 100, num_samples))
#     data['metastasis_months_2'] = np.where(data['hormone_therapy'] == 1, np.random.randint(0, 5, num_samples), np.random.randint(10, 20, num_samples))

#     # Generate important features for radiotherapy
#     data['PR_fmolmg_preTrt'] = np.where(data['radiotherapy'] == 1, np.random.randint(200, 500, num_samples), np.random.randint(0, 100, num_samples))
#     data['age'] = np.where(data['radiotherapy'] == 1, np.random.randint(30, 60, num_samples), np.random.randint(70, 90, num_samples))
#     data['metastasis_months_2'] = np.where(data['radiotherapy'] == 1, np.random.randint(0, 5, num_samples), np.random.randint(10, 20, num_samples))
#     data['ER_fmolmg_preTrt'] = np.where(data['radiotherapy'] == 1, np.random.randint(200, 500, num_samples), np.random.randint(0, 100, num_samples))
#     data['preTrt_numPosLymphNodes'] = np.where(data['radiotherapy'] == 1, np.random.randint(0, 3, num_samples), np.random.randint(5, 10, num_samples))

#     # Generate remaining features
#     for feature in feature_names:
#         if feature not in data:
#             data[feature] = np.random.choice(2, size=num_samples)

#     df = pd.DataFrame(data)

#     return df

# synthetic_data = generate_synthetic_data(num_samples=1000)
# synthetic_data.to_csv('synthetic_dataset.csv', index=False)
# print("Synthetic dataset generated and saved as 'synthetic_dataset.csv'.")

import pandas as pd
import numpy as np

# def generate_synthetic_data(num_samples):
#     feature_names = ['biopsy_preTreat', 'pCR_postTrt_days', 'tumor_size_cm_preTrt_preSurgery',
#                      'tumor_size_cm_secondAxis_preTrt_preSurgery', 'tumor_size_cm_postTrt',
#                      'preTrt_totalLymphNodes', 'preTrt_numPosLymphNodes', 'postTrt_numPosLymphNodes',
#                      'preTrt_posDichLymphNodes', 'hist_grade', 'nuclear_grade_preTrt', 'pCR', 'near_pCR',
#                      'RFS', 'DFS', 'OS', 'metastasis', 'metastasis_months', 'died_from_cancer_if_dead', 'age',
#                      'ER_preTrt', 'ER_fmolmg_preTrt', 'ESR1_preTrt', 'ERbb2_preTrt', 'Erbeta_preTrt',
#                      'ERBB2_CPN_amplified', 'PR_preTrt', 'PR_percentage_preTrt', 'PR_fmolmg_preTrt',
#                      'HER2_preTrt', 'HER2_fish_cont_score_preTrt', 'cytokeratin5_pos', 'top2atri_preTrt',
#                      'topoihc_preTrt', 'S_phase', 'radiotherapy', 'postmenopausal_only', 'anthracycline',
#                      'taxane', 'anti_estrogen', 'aromatase_inhibitor', 'anti_HER2', 'tamoxifen', 'doxorubicin',
#                      'epirubicin', 'docetaxel', 'capecitabine', 'fluorouracil', 'paclitaxel', 'cyclophosphamide',
#                      'anastrozole', 'fulvestrant', 'gefitinib', 'trastuzumab', 'letrozole', 'chemotherapy',
#                      'hormone_therapy', 'methotrexate', 'cetuximab', 'carboplatin', 'other_treatment',
#                      'tumor_size_cm_postTrt_1', 'postTrt_numPosLymphNodes_1', 'metastasis_1',
#                      'metastasis_months_1', 'tumor_size_cm_postTrt_2', 'postTrt_numPosLymphNodes_2',
#                      'metastasis_2', 'metastasis_months_2', 'clinical_AJCC_stage', 'treatment_admin',
#                      'Race', 'preTrt_lymph_node_status', 'postTrt_lymph_node_status', 'tumor_stage_postTrt',
#                      'tumor_stage_preTrt', 'pam50', 'pCR_spectrum', 'RCB', 'menopausal_status',
#                      'HER2_IHC_score_preTrt', 'ploidy', 'estrogen_receptor', 'therapy']

#     data = {}

#     # Generate OS
#     data['OS'] = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])

#     # Generate features used in treatment decision conditions
#     # data['PR_percentage_preTrt'] = np.random.uniform(0, 1, size=num_samples)
#     # data['metastasis_months_2'] = np.random.randint(0, 20, size=num_samples)
#     # data['preTrt_numPosLymphNodes'] = np.random.randint(0, 10, size=num_samples)
#     # data['PR_fmolmg_preTrt'] = np.random.randint(0, 1000, size=num_samples)
#     # data['preTrt_totalLymphNodes'] = np.random.randint(0, 20, size=num_samples)
#     data['ER_fmolmg_preTrt'] = np.random.randint(0, 1000, size=num_samples)
#     data['pCR_postTrt_days'] = np.random.randint(0, 100, size=num_samples)
#     data['topoihc_preTrt'] = np.random.randint(0, 100, size=num_samples)
#     data['age'] = np.random.randint(18, 80, size=num_samples)
#     data['postTrt_numPosLymphNodes'] = np.random.randint(0, 10, size=num_samples)
#     data['metastasis_months'] = np.random.randint(0, 20, size=num_samples)
#     data['tumor_size_cm_secondAxis_preTrt_preSurgery'] = np.random.uniform(0, 10, size=num_samples)
#     data['HER2_fish_cont_score_preTrt'] = np.random.randint(0, 10, size=num_samples)
#     data['S_phase'] = np.random.randint(0, 20, size=num_samples)
#     data['metastasis_months_1'] = np.random.randint(0, 20, size=num_samples)


#     data['surgery'] = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
#     data['chemotherapy'] = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
#     data['hormone_therapy'] = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
#     data['radiotherapy'] = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])

#     # Generate features based on treatment decisions
#     data['PR_percentage_preTrt'] = np.where(data['surgery'] == 1, np.random.uniform(0.6, 1, num_samples), np.random.uniform(0, 0.4, num_samples))
#     data['metastasis_months_2'] = np.where(data['surgery'] == 1, np.random.randint(0, 3, num_samples), np.random.randint(5, 10, num_samples))
#     data['preTrt_numPosLymphNodes'] = np.where(data['surgery'] == 1, np.random.randint(0, 2, num_samples), np.random.randint(3, 6, num_samples))
#     data['PR_fmolmg_preTrt'] = np.where(data['surgery'] == 1, np.random.randint(400, 800, num_samples), np.random.randint(0, 300, num_samples))
#     data['preTrt_totalLymphNodes'] = np.where(data['surgery'] == 1, np.random.randint(0, 5, num_samples), np.random.randint(10, 20, num_samples))

#     data['ER_fmolmg_preTrt'] = np.where(data['hormone_therapy'] == 1, np.random.randint(400, 800, num_samples), np.random.randint(0, 300, num_samples))
#     data['pCR_postTrt_days'] = np.where(data['hormone_therapy'] == 1, np.random.randint(0, 20, num_samples), np.random.randint(40, 60, num_samples))
#     data['postTrt_numPosLymphNodes'] = np.where(data['hormone_therapy'] == 1, np.random.randint(0, 2, num_samples), np.random.randint(3, 6, num_samples))
#     data['topoihc_preTrt'] = np.where(data['hormone_therapy'] == 1, np.random.randint(0, 30, num_samples), np.random.randint(60, 100, num_samples))
#     data['metastasis_months'] = np.where(data['hormone_therapy'] == 1, np.random.randint(0, 3, num_samples), np.random.randint(5, 10, num_samples))

#     data['age'] = np.where(data['chemotherapy'] == 1, np.random.randint(30, 60, num_samples), np.random.randint(65, 80, num_samples))
#     data['metastasis_months_2'] = np.where(data['chemotherapy'] == 1, np.random.randint(0, 3, num_samples), np.random.randint(5, 10, num_samples))
#     data['preTrt_numPosLymphNodes'] = np.where(data['chemotherapy'] == 1, np.random.randint(0, 2, num_samples), np.random.randint(3, 6, num_samples))
#     data['postTrt_numPosLymphNodes'] = np.where(data['chemotherapy'] == 1, np.random.randint(0, 2, num_samples), np.random.randint(3, 6, num_samples))
#     data['topoihc_preTrt'] = np.where(data['chemotherapy'] == 1, np.random.randint(0, 30, num_samples), np.random.randint(60, 100, num_samples))

#     data['PR_fmolmg_preTrt'] = np.where(data['radiotherapy'] == 1, np.random.randint(400, 800, num_samples), np.random.randint(0, 300, num_samples))
#     data['age'] = np.where(data['radiotherapy'] == 1, np.random.randint(30, 60, num_samples), np.random.randint(65, 80, num_samples))
#     data['metastasis_months_2'] = np.where(data['radiotherapy'] == 1, np.random.randint(0, 3, num_samples), np.random.randint(5, 10, num_samples))
#     data['ER_fmolmg_preTrt'] = np.where(data['radiotherapy'] == 1, np.random.randint(400, 800, num_samples), np.random.randint(0, 300, num_samples))
#     data['preTrt_numPosLymphNodes'] = np.where(data['radiotherapy'] == 1, np.random.randint(0, 2, num_samples), np.random.randint(3, 6, num_samples))
    

#     # Generate treatment decisions based on important features from the feature importance lists
#     # data['surgery'] = np.where((data['PR_percentage_preTrt'] > 0.8) | (data['metastasis_months_2'] < 1) | (data['preTrt_numPosLymphNodes'] < 1) | (data['PR_fmolmg_preTrt'] > 500) | (data['preTrt_totalLymphNodes'] < 3),
#     #                             np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8]),
#     #                             np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2]))

#     # data['chemotherapy'] = np.where((data['metastasis_months_2'] < 1) | (data['preTrt_numPosLymphNodes'] < 1) | (data['postTrt_numPosLymphNodes'] < 1) | (data['topoihc_preTrt'] < 20) | (data['age'] < 50),
#     #                                  np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8]),
#     #                                  np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2]))

#     # data['hormone_therapy'] = np.where((data['ER_fmolmg_preTrt'] > 500) | (data['pCR_postTrt_days'] < 10) | (data['postTrt_numPosLymphNodes'] < 1) | (data['topoihc_preTrt'] < 20) | (data['metastasis_months_2'] < 1),
#     #                                     np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8]),
#     #                                     np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2]))

#     # data['radiotherapy'] = np.where((data['PR_fmolmg_preTrt'] > 500) | (data['age'] < 50) | (data['metastasis_months_2'] < 1) | (data['ER_fmolmg_preTrt'] > 500) | (data['preTrt_numPosLymphNodes'] < 1),
#     #                                 np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8]),
#     #                                 np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2]))



#     # Generate remaining features
#     for feature in feature_names:
#         if feature not in data:
#             data[feature] = np.random.choice([0, 1], size=num_samples)

#     df = pd.DataFrame(data)

#     return df

# synthetic_data = generate_synthetic_data(num_samples=2000)
# synthetic_data.to_csv('synthetic_dataset.csv', index=False)
# print("Synthetic dataset generated and saved as 'synthetic_dataset.csv'.")



# generate.py

import numpy as np
import pandas as pd

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
                     'HER2_IHC_score_preTrt', 'ploidy', 'estrogen_receptor', 'therapy']

    data = {}

    # Generate OS
    data['OS'] = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
    data['RFS'] = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
    data['DFS'] = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
    
    # Generate treatment decisions
    data['surgery'] = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
    data['chemotherapy'] = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
    data['hormone_therapy'] = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
    data['radiotherapy'] = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
    
    # Generate features based on treatment decisions
    data['PR_percentage_preTrt'] = np.where(data['surgery'] == 1, np.random.uniform(0.6, 1, num_samples), np.random.uniform(0, 0.4, num_samples))
    data['metastasis_months_2'] = np.where(data['surgery'] == 1, np.random.randint(0, 3, num_samples), np.random.randint(5, 10, num_samples))
    data['preTrt_numPosLymphNodes'] = np.where(data['surgery'] == 1, np.random.randint(0, 2, num_samples), np.random.randint(3, 6, num_samples))
    data['PR_fmolmg_preTrt'] = np.where(data['surgery'] == 1, np.random.randint(400, 800, num_samples), np.random.randint(0, 300, num_samples))
    data['preTrt_totalLymphNodes'] = np.where(data['surgery'] == 1, np.random.randint(0, 5, num_samples), np.random.randint(10, 20, num_samples))
    
    data['ER_fmolmg_preTrt'] = np.where(data['hormone_therapy'] == 1, np.random.randint(400, 800, num_samples), np.random.randint(0, 300, num_samples))
    data['pCR_postTrt_days'] = np.where(data['hormone_therapy'] == 1, np.random.randint(0, 20, num_samples), np.random.randint(40, 60, num_samples))
    data['postTrt_numPosLymphNodes'] = np.where(data['hormone_therapy'] == 1, np.random.randint(0, 2, num_samples), np.random.randint(3, 6, num_samples))
    data['topoihc_preTrt'] = np.where(data['hormone_therapy'] == 1, np.random.randint(0, 30, num_samples), np.random.randint(60, 100, num_samples))
    data['metastasis_months'] = np.where(data['hormone_therapy'] == 1, np.random.randint(0, 3, num_samples), np.random.randint(5, 10, num_samples))
    
    data['age'] = np.where(data['chemotherapy'] == 1, np.random.randint(30, 60, num_samples), np.random.randint(65, 80, num_samples))
    data['metastasis_months_2'] = np.where(data['chemotherapy'] == 1, np.random.randint(0, 3, num_samples), np.random.randint(5, 10, num_samples))
    data['preTrt_numPosLymphNodes'] = np.where(data['chemotherapy'] == 1, np.random.randint(0, 2, num_samples), np.random.randint(3, 6, num_samples))
    data['postTrt_numPosLymphNodes'] = np.where(data['chemotherapy'] == 1, np.random.randint(0, 2, num_samples), np.random.randint(3, 6, num_samples))
    data['topoihc_preTrt'] = np.where(data['chemotherapy'] == 1, np.random.randint(0, 30, num_samples), np.random.randint(60, 100, num_samples))
    
    data['PR_fmolmg_preTrt'] = np.where(data['radiotherapy'] == 1, np.random.randint(400, 800, num_samples), np.random.randint(0, 300, num_samples))
    data['age'] = np.where(data['radiotherapy'] == 1, np.random.randint(30, 60, num_samples), np.random.randint(65, 80, num_samples))
    data['metastasis_months_2'] = np.where(data['radiotherapy'] == 1, np.random.randint(0, 3, num_samples), np.random.randint(5, 10, num_samples))
    data['ER_fmolmg_preTrt'] = np.where(data['radiotherapy'] == 1, np.random.randint(400, 800, num_samples), np.random.randint(0, 300, num_samples))
    data['preTrt_numPosLymphNodes'] = np.where(data['radiotherapy'] == 1, np.random.randint(0, 2, num_samples), np.random.randint(3, 6, num_samples))
    
    # Generate remaining features
    for feature in feature_names:
        if feature not in data:
            data[feature] = np.random.choice([0, 1], size=num_samples)
    
    df = pd.DataFrame(data)
    
    return df

if __name__ == "__main__":
    synthetic_data = generate_synthetic_data(num_samples=2000)
    synthetic_data.to_csv('synthetic_dataset.csv', index=False)
    print("Synthetic dataset generated and saved as 'synthetic_dataset.csv'.")