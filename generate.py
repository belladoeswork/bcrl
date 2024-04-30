import pandas as pd
import numpy as np

def generate_synthetic_data(num_samples):
    # Define the feature names
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
    for feature in feature_names:
        if feature in ['OS', 'RFS', 'DFS']:
            data[feature] = np.random.choice([0, 1], size=num_samples, p=[0.2, 0.8])
        elif feature in ['surgery', 'chemotherapy', 'hormone_therapy', 'radiotherapy']:
            data[feature] = np.random.choice([0, 1], size=num_samples, p=[0.3, 0.7])
        elif feature in ['biopsy_preTreat', 'pCR', 'near_pCR', 'metastasis',  'died_from_cancer_if_dead', 'ER_preTrt', 'ESR1_preTrt',  'ERbb2_preTrt', 'Erbeta_preTrt', 'ERBB2_CPN_amplified', 'PR_preTrt', 'HER2_preTrt', 'cytokeratin5_pos',  
'anthracycline','taxane', 'anti_estrogen', 'aromatase_inhibitor', 'anti_HER2',  'doxorubicin', 'epirubicin', 'docetaxel', 'capecitabine', 'fluorouracil', 'paclitaxel', 'cyclophosphamide',
'anastrozole', 'fulvestrant', 'gefitinib', 'trastuzumab', 'letrozole', 'methotrexate', 'cetuximab', 'carboplatin', 'other_treatment', 'metastasis_1', 'metastasis_2']:
            data[feature] = np.random.choice(['Y', 'N'], size=num_samples)
        elif feature in ['top2atri_preTrt']: 
            data[feature] = np.random.choice([-1, 0, 1], size=num_samples)
        elif feature in ['tumor_size_cm_postTrt_2', 'tumor_size_cm_postTrt_1']:
            data[feature] = np.random.choice([0, 1, 2], size=num_samples)
        elif feature in ['preTrt_posDichLymphNodes', 'tamoxifen']:
            data[feature] = np.random.choice([0, 1], size=num_samples)
        elif feature in ['hist_grade', 'nuclear_grade_preTrt']:
            data[feature] = np.random.choice([1, 2, 3], size=num_samples)
        elif feature in ['age']:
            data[feature] = np.random.randint(20, 90, num_samples)
        elif feature in ['postmenopausal_only']:
            data[feature] = np.where(data['age'] >= 45, 1, 0)
        elif feature in ['menopausal_status']:
            data[feature] = np.where(data['postmenopausal_only'] == 1, 'post', 'pre')
        elif feature in ['ER_fmolmg_preTrt']:
            data[feature] = np.random.randint(0, 500, num_samples)
        elif feature in ['S_phase']:
            data[feature] = np.random.randint(1, 40, num_samples)
        elif feature in ['topoihc_preTrt']:
            data[feature] = np.random.randint(0, 100, num_samples)
        elif feature in ['HER2_fish_cont_score_preTrt']:
            data[feature] = np.random.randint(0, 20, num_samples)
        elif feature in ['postTrt_numPosLymphNodes_1']:
            data[feature] = np.random.randint(1, 15, num_samples)
        elif feature in ['postTrt_numPosLymphNodes_2']:
            data[feature] = np.random.randint(0, 15, num_samples)
        elif feature in ['pCR_postTrt_days', 'PR_percentage_preTrt']:
            data[feature] = np.random.randint(0, 100, num_samples)
        elif feature in ['metastasis_months', 'metastasis_months_1', 'metastasis_months_2']:
            data[feature] = np.random.randint(1, 100, num_samples)
        elif feature in ['tumor_size_cm_preTrt_preSurgery', 'tumor_size_cm_secondAxis_preTrt_preSurgery','tumor_size_cm_postTrt']:
            data[feature] = np.random.randint(0, 25, num_samples)
        elif feature in ['preTrt_totalLymphNodes', 'preTrt_numPosLymphNodes', 'postTrt_numPosLymphNodes', 'PR_fmolmg_preTrt']:
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
        # elif feature in ['menopausal_status']:
        #     data[feature] = np.random.choice(['post', 'pre'], size=num_samples)
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
            data[feature] = np.random.rand(num_samples)
    
    # Create a DataFrame from the generated data
    df = pd.DataFrame(data)
    
    return df

# Generate the synthetic dataset
synthetic_data = generate_synthetic_data(num_samples=1000)
synthetic_data.to_csv('synthetic_dataset.csv', index=False)
print("Synthetic dataset generated and saved as 'synthetic_dataset.csv'.")



# 'pCR_postTrt_days'
# gen whole numbers between 0 and 100 days
# 'metastasis_months', 'metastasis_months_1', 'metastasis_months_2' whole number between 1 and 100 in months
# 'PR_percentage_preTrt', whole number between 0 and 100 %
# 'tumor_size_cm_preTrt_preSurgery', 'tumor_size_cm_secondAxis_preTrt_preSurgery','tumor_size_cm_postTrt', 
#  from 0 > 25 whole and decimal numbers

# 'tumor_size_cm_postTrt_2', 'tumor_size_cm_postTrt_1' whole numbers 0 1 2
# 'preTrt_totalLymphNodes', 'preTrt_numPosLymphNodes', 'postTrt_numPosLymphNodes', 'PR_fmolmg_preTrt'
# gen whole numbers can be from 0 - 500
# 'postTrt_numPosLymphNodes_2' whole numbers between 0 and 15
# 'HER2_fish_cont_score_preTrt' whole numbers bet 0 and 20
# 'preTrt_posDichLymphNodes', 'tamoxifen', numbers  0 1
# 'hist_grade', 'nuclear_grade_preTrt' numbers  1 2 3
# 'postmenopausal_only' should be based on age (from the age generat who is, then give 0 or 1)

# 'biopsy_preTreat', 'pCR', 'near_pCR', 'metastasis',  'died_from_cancer_if_dead', 'ER_preTrt', 'ESR1_preTrt',  'ERbb2_preTrt', 'Erbeta_preTrt', 'ERBB2_CPN_amplified', 'PR_preTrt', 'HER2_preTrt', 'cytokeratin5_pos',  
# 'anthracycline','taxane', 'anti_estrogen', 'aromatase_inhibitor', 'anti_HER2',  'doxorubicin', 'epirubicin', 'docetaxel', 'capecitabine', 'fluorouracil', 'paclitaxel', 'cyclophosphamide',
# 'anastrozole', 'fulvestrant', 'gefitinib', 'trastuzumab', 'letrozole', 'methotrexate', 'cetuximab', 'carboplatin', 'other_treatment', 'metastasis_1', 'metastasis_2'
# Y N

#  'age', whole numbers between 20 and 90
# 'ER_fmolmg_preTrt', 
# whole numbers between 0 and 500
# 'top2atri_preTrt' whole numbers  -1 0 1
# 'topoihc_preTrt' whole num bet 0 and 100
# 'S_phase' whole num bet 1 and 40
# 'postTrt_numPosLymphNodes_1' whole numbers between 1 and  15
# 'clinical_AJCC_stage' I II III IV IIB IIA  IIIA IIIB IIIC 
# 'treatment_admin' oral NOS intarvenous
# 'Race' White Hispanic Black Asian Mixed
# 'preTrt_lymph_node_status' N0 N1 positive N2 N3 ND
# 'postTrt_lymph_node_status' N0 N1 positive N2 N3 ND N1a N2a
# 'tumor_stage_postTrt ' T1c T0 T1b T1a T2 Tis T1 T1mic
# 'tumor_stage_preTrt' T2 T1 T3 T4 T0
# 'pam50' Basal Luminal A Luminal B Her2 Normal Claudin
# 'pCR_spectrum' PR CR NCR PD SD EPD NE
# 'RCB' II 0/I 2 III 3 0 1
# 'menopausal_status' post pre based on menapause age
# 'HER2_IHC_score_preTrt' between NEG 0 1 2 3 ND
# 'ploidy' aneuploid diploid multiploid
# 'estrogen_receptor' NOS block_and_stop block_and_eliminate
# 'therapy' neo adj mixed unspecified
