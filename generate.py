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


# THIS IS GOOD ALREADY

# generate.py

# import numpy as np
# import pandas as pd

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
#                      'HER2_IHC_score_preTrt', 'ploidy', 'estrogen_receptor', 'therapy',
#                      'hormone_receptor_status', 'lymph_nodes', 'tumor_size', 'her2_status', 'grade', 'stage', 
#                      'histology', 'Ki67_expression', 'oncotype_score']

#     data = {}

#     # Generate OS
#     # data['OS'] = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
#     # data['RFS'] = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
#     # data['DFS'] = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
    
#     # # Generate treatment decisions
#     # data['surgery'] = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
#     # data['chemotherapy'] = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
#     # data['hormone_therapy'] = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
#     # data['radiotherapy'] = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])

#     data['surgery'] = np.random.choice(['Lumpectomy', 'Mastectomy'], size=num_samples, p=[0.6, 0.4])
#     data['chemotherapy'] = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.5, 0.5])
#     data['hormone_therapy'] = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.7, 0.3])
#     data['radiotherapy'] = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.6, 0.4])

    
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
    
#     # data['age'] = np.where(data['chemotherapy'] == 1, np.random.randint(30, 60, num_samples), np.random.randint(65, 80, num_samples))
#     data['age'] = np.random.normal(loc=55, scale=10, size=num_samples).astype(int)
#     data['tumor_size'] = np.random.normal(loc=3.0, scale=1.5, size=num_samples).clip(0.1, None)
#     data['lymph_nodes'] = np.random.binomial(n=10, p=0.3, size=num_samples)
#     data['hormone_receptor_status'] = np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.7, 0.3])
#     data['her2_status'] = np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.2, 0.8])
#     data['grade'] = np.random.choice([1, 2, 3], size=num_samples, p=[0.1, 0.5, 0.4])
#     data['stage'] = np.random.choice([1, 2, 3, 4], size=num_samples, p=[0.3, 0.4, 0.2, 0.1])
#     data['histology'] = np.random.choice(['Ductal', 'Lobular', 'Mixed', 'Other'], size=num_samples, p=[0.7, 0.2, 0.05, 0.05])
#     data['Ki67_expression'] = np.random.normal(loc=30, scale=10, size=num_samples).clip(0, 100)
#     data['oncotype_score'] = np.random.normal(loc=20, scale=5, size=num_samples).clip(0, 100)

#     data['metastasis_months_2'] = np.where(data['chemotherapy'] == 1, np.random.randint(0, 3, num_samples), np.random.randint(5, 10, num_samples))
#     data['preTrt_numPosLymphNodes'] = np.where(data['chemotherapy'] == 1, np.random.randint(0, 2, num_samples), np.random.randint(3, 6, num_samples))
#     data['postTrt_numPosLymphNodes'] = np.where(data['chemotherapy'] == 1, np.random.randint(0, 2, num_samples), np.random.randint(3, 6, num_samples))
#     data['topoihc_preTrt'] = np.where(data['chemotherapy'] == 1, np.random.randint(0, 30, num_samples), np.random.randint(60, 100, num_samples))
    
#     data['PR_fmolmg_preTrt'] = np.where(data['radiotherapy'] == 1, np.random.randint(400, 800, num_samples), np.random.randint(0, 300, num_samples))
#     data['age'] = np.where(data['radiotherapy'] == 1, np.random.randint(30, 60, num_samples), np.random.randint(65, 80, num_samples))
#     data['metastasis_months_2'] = np.where(data['radiotherapy'] == 1, np.random.randint(0, 3, num_samples), np.random.randint(5, 10, num_samples))
#     data['ER_fmolmg_preTrt'] = np.where(data['radiotherapy'] == 1, np.random.randint(400, 800, num_samples), np.random.randint(0, 300, num_samples))
#     data['preTrt_numPosLymphNodes'] = np.where(data['radiotherapy'] == 1, np.random.randint(0, 2, num_samples), np.random.randint(3, 6, num_samples))



#     data['OS'] = np.random.binomial(n=1, p=np.where(data['stage'] < 3, 0.9, 0.6), size=num_samples)
#     data['DFS'] = np.random.binomial(n=1, p=np.where(data['stage'] < 3, 0.8, 0.5), size=num_samples)
#     data['RFS'] = np.random.binomial(n=1, p=np.where(data['stage'] < 3, 0.7, 0.4), size=num_samples)
    
#     # Generate remaining features
#     for feature in feature_names:
#         if feature not in data:
#             data[feature] = np.random.choice([0, 1], size=num_samples)
    
#     df = pd.DataFrame(data)
    
#     return df

# if __name__ == "__main__":
#     synthetic_data = generate_synthetic_data(num_samples=2000)
#     synthetic_data.to_csv('synthetic_dataset.csv', index=False)
#     print("Synthetic dataset generated and saved as 'synthetic_dataset.csv'.")



import numpy as np
import pandas as pd

# def generate_synthetic_data(num_samples):
#     feature_names = ['age', 'menopausal_status', 'tumor_size', 'lymph_nodes', 'stage', 'histology', 'grade',
#                      'er_status', 'pr_status', 'her2_status', 'ki67_expression', 'oncotype_score',
#                      'surgery', 'tumor_size_post_surgery', 'lymph_nodes_post_surgery',
#                      'chemotherapy', 'tumor_size_post_chemo', 'lymph_nodes_post_chemo', 'recurrence_post_chemo',
#                      'hormone_therapy', 'tumor_size_post_hormone', 'lymph_nodes_post_hormone', 'recurrence_post_hormone',
#                      'radiotherapy', 'tumor_size_post_radio', 'lymph_nodes_post_radio', 'recurrence_post_radio',
#                      'OS', 'DFS', 'RFS']

#     data = {}

#     # Pre-treatment features
#     data['age'] = np.random.normal(loc=55, scale=10, size=num_samples).astype(int)
#     data['menopausal_status'] = np.random.choice(['Pre', 'Post'], size=num_samples, p=[0.4, 0.6])
#     data['tumor_size'] = np.random.normal(loc=3.0, scale=1.5, size=num_samples).clip(0.1, None)
#     data['lymph_nodes'] = np.random.binomial(n=10, p=0.3, size=num_samples)
#     data['stage'] = np.random.choice([1, 2, 3, 4], size=num_samples, p=[0.3, 0.4, 0.2, 0.1])
#     data['histology'] = np.random.choice(['Ductal', 'Lobular', 'Mixed', 'Other'], size=num_samples, p=[0.7, 0.2, 0.05, 0.05])
#     data['grade'] = np.random.choice([1, 2, 3], size=num_samples, p=[0.1, 0.5, 0.4])
#     data['er_status'] = np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.7, 0.3])
#     data['pr_status'] = np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.6, 0.4])
#     data['her2_status'] = np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.2, 0.8])
#     data['ki67_expression'] = np.random.normal(loc=30, scale=10, size=num_samples).clip(0, 100)
#     data['oncotype_score'] = np.random.normal(loc=20, scale=5, size=num_samples).clip(0, 100)

#     # Treatment decision 1: Surgery
#     data['surgery'] = np.where((data['tumor_size'] < 3) & (data['lymph_nodes'] < 3),
#                                np.random.choice(['Lumpectomy', 'Mastectomy'], size=num_samples, p=[0.8, 0.2]),
#                                np.random.choice(['Lumpectomy', 'Mastectomy'], size=num_samples, p=[0.3, 0.7]))
#     data['surgery'] = np.where(np.random.rand(num_samples) < 0.1, np.random.permutation(data['surgery']), data['surgery'])

#     # Post-surgery features
#     data['tumor_size_post_surgery'] = np.where(data['surgery'] == 'Lumpectomy',
#                                                data['tumor_size'] * np.random.normal(loc=0.2, scale=0.1, size=num_samples),
#                                                0.0)
#     data['lymph_nodes_post_surgery'] = np.where(data['surgery'] == 'Lumpectomy',
#                                                 data['lymph_nodes'] * np.random.binomial(n=1, p=0.2, size=num_samples),
#                                                 0)

#     # Treatment decision 2: Chemotherapy
#     data['chemotherapy'] = np.where((data['stage'] > 2) | (data['grade'] > 2) | (data['ki67_expression'] > 30),
#                                     np.random.choice(['Yes', 'No'], size=num_samples, p=[0.8, 0.2]),
#                                     np.random.choice(['Yes', 'No'], size=num_samples, p=[0.3, 0.7]))
#     data['chemotherapy'] = np.where(np.random.rand(num_samples) < 0.1, np.random.permutation(data['chemotherapy']), data['chemotherapy'])

#     # Post-chemotherapy features
#     data['tumor_size_post_chemo'] = np.where(data['chemotherapy'] == 'Yes',
#                                              data['tumor_size_post_surgery'] * np.random.normal(loc=0.5, scale=0.1, size=num_samples),
#                                              data['tumor_size_post_surgery'])
#     data['lymph_nodes_post_chemo'] = np.where(data['chemotherapy'] == 'Yes',
#                                               data['lymph_nodes_post_surgery'] * np.random.binomial(n=1, p=0.5, size=num_samples),
#                                               data['lymph_nodes_post_surgery'])
#     data['recurrence_post_chemo'] = np.random.binomial(n=1, p=0.1, size=num_samples)

#     # Treatment decision 3: Hormone therapy
#     data['hormone_therapy'] = np.where((data['er_status'] == 'Positive') | (data['pr_status'] == 'Positive'),
#                                        np.random.choice(['Yes', 'No'], size=num_samples, p=[0.9, 0.1]),
#                                        np.random.choice(['Yes', 'No'], size=num_samples, p=[0.1, 0.9]))
#     data['hormone_therapy'] = np.where(np.random.rand(num_samples) < 0.1, np.random.permutation(data['hormone_therapy']), data['hormone_therapy'])

#     # Post-hormone therapy features
#     data['tumor_size_post_hormone'] = np.where(data['hormone_therapy'] == 'Yes',
#                                                data['tumor_size_post_chemo'] * np.random.normal(loc=0.8, scale=0.1, size=num_samples),
#                                                data['tumor_size_post_chemo'])
#     data['lymph_nodes_post_hormone'] = np.where(data['hormone_therapy'] == 'Yes',
#                                                 data['lymph_nodes_post_chemo'] * np.random.binomial(n=1, p=0.8, size=num_samples),
#                                                 data['lymph_nodes_post_chemo'])
#     data['recurrence_post_hormone'] = np.random.binomial(n=1, p=0.05, size=num_samples)

#     # Treatment decision 4: Radiotherapy
#     data['radiotherapy'] = np.where((data['surgery'] == 'Lumpectomy') | (data['lymph_nodes_post_hormone'] > 0),
#                                     np.random.choice(['Yes', 'No'], size=num_samples, p=[0.8, 0.2]),
#                                     np.random.choice(['Yes', 'No'], size=num_samples, p=[0.2, 0.8]))
#     data['radiotherapy'] = np.where(np.random.rand(num_samples) < 0.1, np.random.permutation(data['radiotherapy']), data['radiotherapy'])

#     # Post-radiotherapy features
#     data['tumor_size_post_radio'] = np.where(data['radiotherapy'] == 'Yes',
#                                              data['tumor_size_post_hormone'] * np.random.normal(loc=0.9, scale=0.05, size=num_samples),
#                                              data['tumor_size_post_hormone'])
#     data['lymph_nodes_post_radio'] = np.where(data['radiotherapy'] == 'Yes',
#                                               data['lymph_nodes_post_hormone'] * np.random.binomial(n=1, p=0.9, size=num_samples),
#                                               data['lymph_nodes_post_hormone'])
#     data['recurrence_post_radio'] = np.random.binomial(n=1, p=0.02, size=num_samples)

#     # Outcomes
#     data['OS'] = np.random.binomial(n=1, p=np.where(data['stage'] < 3, 0.9, 0.6), size=num_samples)
#     data['DFS'] = np.random.binomial(n=1, p=np.where(data['stage'] < 3, 0.8, 0.5), size=num_samples)
#     data['RFS'] = np.random.binomial(n=1, p=np.where(data['stage'] < 3, 0.7, 0.4), size=num_samples)

#     # Create a DataFrame from the generated data
#     df = pd.DataFrame(data)

#     return df


# THIS IS ALSO GOOD WITH > results

# def generate_synthetic_data(num_samples):
#     feature_names = ['age', 'menopausal_status', 'tumor_size', 'lymph_nodes', 'stage', 'histology', 'grade',
#                      'er_status', 'pr_status', 'her2_status', 'ki67_expression', 'oncotype_score',
#                      'surgery', 'tumor_size_post_surgery', 'lymph_nodes_post_surgery',
#                      'chemotherapy', 'tumor_size_post_chemo', 'lymph_nodes_post_chemo', 'recurrence_post_chemo',
#                      'hormone_therapy', 'tumor_size_post_hormone', 'lymph_nodes_post_hormone', 'recurrence_post_hormone',
#                      'radiotherapy', 'tumor_size_post_radio', 'lymph_nodes_post_radio', 'recurrence_post_radio',
#                      'OS', 'DFS', 'RFS']

#     data = {}

#     # Pre-treatment features
#     data['age'] = np.random.normal(loc=55, scale=10, size=num_samples).astype(int)
#     data['menopausal_status'] = np.random.choice(['Pre', 'Post'], size=num_samples, p=[0.4, 0.6])
#     data['stage'] = np.random.choice([1, 2, 3, 4], size=num_samples, p=[0.3, 0.4, 0.2, 0.1])
#     data['tumor_size'] = np.where(data['stage'] < 3, np.random.normal(loc=2.0, scale=1.0, size=num_samples), np.random.normal(loc=4.0, scale=1.5, size=num_samples)).clip(0.1, None)
#     data['lymph_nodes'] = np.where(data['stage'] < 3, np.random.binomial(n=5, p=0.2, size=num_samples), np.random.binomial(n=15, p=0.4, size=num_samples))
#     data['histology'] = np.random.choice(['Ductal', 'Lobular', 'Mixed', 'Other'], size=num_samples, p=[0.7, 0.2, 0.05, 0.05])
#     data['grade'] = np.where(data['stage'] < 3, np.random.choice([1, 2, 3], size=num_samples, p=[0.2, 0.6, 0.2]), np.random.choice([1, 2, 3], size=num_samples, p=[0.05, 0.35, 0.6]))
#     data['er_status'] = np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.7, 0.3])
#     data['pr_status'] = np.where(data['er_status'] == 'Positive', np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.8, 0.2]), np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.2, 0.8]))
#     data['her2_status'] = np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.2, 0.8])
#     data['ki67_expression'] = np.where(data['grade'] < 3, np.random.normal(loc=20, scale=5, size=num_samples).clip(0, 100), np.random.normal(loc=40, scale=10, size=num_samples).clip(0, 100))
#     data['oncotype_score'] = np.where((data['er_status'] == 'Positive') & (data['grade'] < 3), np.random.normal(loc=15, scale=5, size=num_samples).clip(0, 100), np.random.normal(loc=30, scale=10, size=num_samples).clip(0, 100))

#     # Treatment decision 1: Surgery
#     data['surgery'] = np.where((data['tumor_size'] < 3) & (data['lymph_nodes'] < 3), np.random.choice(['Lumpectomy', 'Mastectomy'], size=num_samples, p=[0.9, 0.1]), np.random.choice(['Lumpectomy', 'Mastectomy'], size=num_samples, p=[0.1, 0.9]))

#     # Post-surgery features
#     data['tumor_size_post_surgery'] = np.where(data['surgery'] == 'Lumpectomy', data['tumor_size'] * np.random.normal(loc=0.2, scale=0.1, size=num_samples), 0.0)
#     data['lymph_nodes_post_surgery'] = np.where(data['surgery'] == 'Lumpectomy', data['lymph_nodes'] * np.random.binomial(n=1, p=0.2, size=num_samples), 0)

#     # Treatment decision 2: Chemotherapy
#     data['chemotherapy'] = np.where((data['stage'] > 2) | (data['grade'] > 2) | (data['ki67_expression'] > 30) | (data['er_status'] == 'Negative'), np.random.choice(['Yes', 'No'], size=num_samples, p=[0.9, 0.1]), np.random.choice(['Yes', 'No'], size=num_samples, p=[0.1, 0.9]))

#     # Post-chemotherapy features
#     data['tumor_size_post_chemo'] = np.where(data['chemotherapy'] == 'Yes', data['tumor_size_post_surgery'] * np.random.normal(loc=0.5, scale=0.1, size=num_samples), data['tumor_size_post_surgery'])
#     data['lymph_nodes_post_chemo'] = np.where(data['chemotherapy'] == 'Yes', data['lymph_nodes_post_surgery'] * np.random.binomial(n=1, p=0.5, size=num_samples), data['lymph_nodes_post_surgery'])
#     data['recurrence_post_chemo'] = np.where(data['chemotherapy'] == 'Yes', np.random.binomial(n=1, p=0.1, size=num_samples), np.random.binomial(n=1, p=0.3, size=num_samples))

#     # Treatment decision 3: Hormone therapy
#     data['hormone_therapy'] = np.where((data['er_status'] == 'Positive') | (data['pr_status'] == 'Positive'), np.random.choice(['Yes', 'No'], size=num_samples, p=[0.9, 0.1]), np.random.choice(['Yes', 'No'], size=num_samples, p=[0.1, 0.9]))

#     # Post-hormone therapy features
#     data['tumor_size_post_hormone'] = np.where(data['hormone_therapy'] == 'Yes', data['tumor_size_post_chemo'] * np.random.normal(loc=0.8, scale=0.1, size=num_samples), data['tumor_size_post_chemo'])
#     data['lymph_nodes_post_hormone'] = np.where(data['hormone_therapy'] == 'Yes', data['lymph_nodes_post_chemo'] * np.random.binomial(n=1, p=0.8, size=num_samples), data['lymph_nodes_post_chemo'])
#     data['recurrence_post_hormone'] = np.where(data['hormone_therapy'] == 'Yes', np.random.binomial(n=1, p=0.05, size=num_samples), np.random.binomial(n=1, p=0.2, size=num_samples))

#     # Treatment decision 4: Radiotherapy
#     data['radiotherapy'] = np.where((data['surgery'] == 'Lumpectomy') | (data['lymph_nodes_post_hormone'] > 0) | (data['tumor_size_post_hormone'] > 2), np.random.choice(['Yes', 'No'], size=num_samples, p=[0.9, 0.1]), np.random.choice(['Yes', 'No'], size=num_samples, p=[0.1, 0.9]))

#     # Post-radiotherapy features
#     data['tumor_size_post_radio'] = np.where(data['radiotherapy'] == 'Yes', data['tumor_size_post_hormone'] * np.random.normal(loc=0.9, scale=0.05, size=num_samples), data['tumor_size_post_hormone'])
#     data['lymph_nodes_post_radio'] = np.where(data['radiotherapy'] == 'Yes', data['lymph_nodes_post_hormone'] * np.random.binomial(n=1, p=0.9, size=num_samples), data['lymph_nodes_post_hormone'])
#     data['recurrence_post_radio'] = np.where(data['radiotherapy'] == 'Yes', np.random.binomial(n=1, p=0.02, size=num_samples), np.random.binomial(n=1, p=0.1, size=num_samples))

#     # Outcomes
#     data['OS'] = np.where((data['stage'] < 3) & (data['recurrence_post_radio'] == 0), np.random.binomial(n=1, p=0.95, size=num_samples), np.random.binomial(n=1, p=0.6, size=num_samples))
#     data['DFS'] = np.where((data['stage'] < 3) & (data['recurrence_post_radio'] == 0), np.random.binomial(n=1, p=0.9, size=num_samples), np.random.binomial(n=1, p=0.5, size=num_samples))
#     data['RFS'] = np.where((data['stage'] < 3) & (data['recurrence_post_radio'] == 0), np.random.binomial(n=1, p=0.85, size=num_samples), np.random.binomial(n=1, p=0.4, size=num_samples))

#     # Create a DataFrame from the generated data
#     df = pd.DataFrame(data)

#     return df

# if __name__ == "__main__":
#     synthetic_data = generate_synthetic_data(num_samples=2000)
#     synthetic_data.to_csv('synthetic_dataset.csv', index=False)
#     print("Synthetic dataset generated and saved as 'synthetic_dataset.csv'.")



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

    # Treatment decision 1: Surgery
    data['surgery'] = np.where((data['tumor_size'] < 2) & (data['lymph_nodes'] < 2) & (data['stage'] < 3), 'Lumpectomy', 'Mastectomy')
    data['surgery'] = np.where(data['tumor_size'] < 1, 'Lumpectomy', data['surgery'])

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

    # Treatment decision 4: Radiotherapy
    data['radiotherapy'] = np.where((data['surgery'] == 'Lumpectomy') | ((data['stage'] > 1) & ((data['lymph_nodes_post_hormone'] > 1) | (data['tumor_size_post_hormone'] > 2))), 'Yes', 'No')
    data['radiotherapy'] = np.where((data['lymph_nodes_post_hormone'] > 1) | (data['tumor_size_post_hormone'] > 2), 'Yes', data['radiotherapy'])

    # Post-radiotherapy features
    data['tumor_size_post_radio'] = np.where(data['radiotherapy'] == 'Yes', data['tumor_size_post_hormone'] * np.random.normal(loc=0.9, scale=0.05, size=num_samples), data['tumor_size_post_hormone'])
    data['lymph_nodes_post_radio'] = np.where(data['radiotherapy'] == 'Yes', data['lymph_nodes_post_hormone'] * np.random.binomial(n=1, p=0.95, size=num_samples), data['lymph_nodes_post_hormone'])
    data['recurrence_post_radio'] = np.where(data['radiotherapy'] == 'Yes', np.random.binomial(n=1, p=0.01, size=num_samples), np.random.binomial(n=1, p=0.05, size=num_samples))

    # Outcomes
    data['OS'] = np.where((data['stage'] < 3) & (data['recurrence_post_radio'] == 0), np.random.binomial(n=1, p=0.98, size=num_samples), np.random.binomial(n=1, p=0.7, size=num_samples))
    data['DFS'] = np.where((data['stage'] < 3) & (data['recurrence_post_radio'] == 0), np.random.binomial(n=1, p=0.95, size=num_samples), np.random.binomial(n=1, p=0.6, size=num_samples))
    data['RFS'] = np.where((data['stage'] < 3) & (data['recurrence_post_radio'] == 0) & (data['recurrence_post_chemo'] == 0) & (data['recurrence_post_hormone'] == 0), np.random.binomial(n=1, p=0.95, size=num_samples), np.random.binomial(n=1, p=0.6, size=num_samples))

    # Create a DataFrame from the generated data
    df = pd.DataFrame(data)

    return df



# def generate_synthetic_data(num_samples):
#     feature_names = ['age', 'menopausal_status', 'tumor_size', 'lymph_nodes', 'stage', 'histology', 'grade',
#                      'er_status', 'pr_status', 'her2_status', 'ki67_expression', 'oncotype_score',
#                      'surgery', 'tumor_size_post_surgery', 'lymph_nodes_post_surgery',
#                      'chemotherapy', 'tumor_size_post_chemo', 'lymph_nodes_post_chemo', 'recurrence_post_chemo',
#                      'hormone_therapy', 'tumor_size_post_hormone', 'lymph_nodes_post_hormone', 'recurrence_post_hormone',
#                      'radiotherapy', 'tumor_size_post_radio', 'lymph_nodes_post_radio', 'recurrence_post_radio',
#                      'OS', 'DFS', 'RFS']

#     data = {}

#     # Pre-treatment features
#     data['age'] = np.random.normal(loc=55, scale=10, size=num_samples).astype(int)
#     data['menopausal_status'] = np.where(data['age'] > 50, 'Post', 'Pre')
#     data['stage'] = np.random.choice([1, 2, 3, 4], size=num_samples, p=[0.4, 0.3, 0.2, 0.1])
#     data['tumor_size'] = np.where(data['stage'] < 3, np.random.normal(loc=1.5, scale=0.5, size=num_samples), np.random.normal(loc=4.5, scale=1.0, size=num_samples)).clip(0.1, None)
#     data['lymph_nodes'] = np.where(data['stage'] < 3, np.random.binomial(n=3, p=0.2, size=num_samples), np.random.binomial(n=10, p=0.6, size=num_samples))
#     data['histology'] = np.random.choice(['Ductal', 'Lobular', 'Mixed', 'Other'], size=num_samples, p=[0.8, 0.1, 0.05, 0.05])
#     data['grade'] = np.where(data['stage'] < 3, np.random.choice([1, 2, 3], size=num_samples, p=[0.3, 0.5, 0.2]), np.random.choice([1, 2, 3], size=num_samples, p=[0.1, 0.3, 0.6]))
#     data['er_status'] = np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.8, 0.2])
#     data['pr_status'] = np.where(data['er_status'] == 'Positive', np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.9, 0.1]), np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.1, 0.9]))
#     data['her2_status'] = np.random.choice(['Positive', 'Negative'], size=num_samples, p=[0.2, 0.8])
#     data['ki67_expression'] = np.where(data['grade'] < 3, np.random.normal(loc=15, scale=5, size=num_samples).clip(0, 100), np.random.normal(loc=40, scale=10, size=num_samples).clip(0, 100))
#     data['oncotype_score'] = np.where((data['er_status'] == 'Positive') & (data['grade'] < 3), np.random.normal(loc=10, scale=5, size=num_samples).clip(0, 100), np.random.normal(loc=30, scale=10, size=num_samples).clip(0, 100))

#     # Treatment decision 1: Surgery
#     data['surgery'] = np.where((data['tumor_size'] < 2) & (data['lymph_nodes'] < 2), 'Lumpectomy', 'Mastectomy')

#     # Post-surgery features
#     data['tumor_size_post_surgery'] = np.where(data['surgery'] == 'Lumpectomy', data['tumor_size'] * np.random.normal(loc=0.1, scale=0.05, size=num_samples), 0.0)
#     data['lymph_nodes_post_surgery'] = np.where(data['surgery'] == 'Lumpectomy', data['lymph_nodes'] * np.random.binomial(n=1, p=0.1, size=num_samples), 0)

#     # Treatment decision 2: Chemotherapy
#     data['chemotherapy'] = np.where((data['stage'] > 1) | (data['grade'] > 2) | (data['ki67_expression'] > 20) | (data['er_status'] == 'Negative') | (data['her2_status'] == 'Positive'), 'Yes', 'No')

#     # Post-chemotherapy features
#     data['tumor_size_post_chemo'] = np.where(data['chemotherapy'] == 'Yes', data['tumor_size_post_surgery'] * np.random.normal(loc=0.3, scale=0.1, size=num_samples), data['tumor_size_post_surgery'])
#     data['lymph_nodes_post_chemo'] = np.where(data['chemotherapy'] == 'Yes', data['lymph_nodes_post_surgery'] * np.random.binomial(n=1, p=0.3, size=num_samples), data['lymph_nodes_post_surgery'])
#     data['recurrence_post_chemo'] = np.where(data['chemotherapy'] == 'Yes', np.random.binomial(n=1, p=0.05, size=num_samples), np.random.binomial(n=1, p=0.2, size=num_samples))

#     # Treatment decision 3: Hormone therapy
#     data['hormone_therapy'] = np.where((data['er_status'] == 'Positive') | (data['pr_status'] == 'Positive'), 'Yes', 'No')

#     # Post-hormone therapy features
#     data['tumor_size_post_hormone'] = np.where(data['hormone_therapy'] == 'Yes', data['tumor_size_post_chemo'] * np.random.normal(loc=0.8, scale=0.1, size=num_samples), data['tumor_size_post_chemo'])
#     data['lymph_nodes_post_hormone'] = np.where(data['hormone_therapy'] == 'Yes', data['lymph_nodes_post_chemo'] * np.random.binomial(n=1, p=0.9, size=num_samples), data['lymph_nodes_post_chemo'])
#     data['recurrence_post_hormone'] = np.where(data['hormone_therapy'] == 'Yes', np.random.binomial(n=1, p=0.02, size=num_samples), np.random.binomial(n=1, p=0.1, size=num_samples))

#     # Treatment decision 4: Radiotherapy
#     data['radiotherapy'] = np.where((data['surgery'] == 'Lumpectomy') | (data['lymph_nodes_post_hormone'] > 0) | (data['tumor_size_post_hormone'] > 1), 'Yes', 'No')

#     # Post-radiotherapy features
#     data['tumor_size_post_radio'] = np.where(data['radiotherapy'] == 'Yes', data['tumor_size_post_hormone'] * np.random.normal(loc=0.9, scale=0.05, size=num_samples), data['tumor_size_post_hormone'])
#     data['lymph_nodes_post_radio'] = np.where(data['radiotherapy'] == 'Yes', data['lymph_nodes_post_hormone'] * np.random.binomial(n=1, p=0.95, size=num_samples), data['lymph_nodes_post_hormone'])
#     data['recurrence_post_radio'] = np.where(data['radiotherapy'] == 'Yes', np.random.binomial(n=1, p=0.01, size=num_samples), np.random.binomial(n=1, p=0.05, size=num_samples))

#     # Outcomes
#     data['OS'] = np.where((data['stage'] < 3) & (data['recurrence_post_radio'] == 0), np.random.binomial(n=1, p=0.98, size=num_samples), np.random.binomial(n=1, p=0.7, size=num_samples))
#     data['DFS'] = np.where((data['stage'] < 3) & (data['recurrence_post_radio'] == 0), np.random.binomial(n=1, p=0.95, size=num_samples), np.random.binomial(n=1, p=0.6, size=num_samples))
#     data['RFS'] = np.where((data['stage'] < 3) & (data['recurrence_post_radio'] == 0) & (data['recurrence_post_chemo'] == 0) & (data['recurrence_post_hormone'] == 0), np.random.binomial(n=1, p=0.95, size=num_samples), np.random.binomial(n=1, p=0.6, size=num_samples))

#     # Create a DataFrame from the generated data
#     df = pd.DataFrame(data)

#     return df





if __name__ == "__main__":
    synthetic_data = generate_synthetic_data(num_samples=2000)
    synthetic_data.to_csv('synthetic_dataset.csv', index=False)
    print("Synthetic dataset generated and saved as 'synthetic_dataset.csv'.")