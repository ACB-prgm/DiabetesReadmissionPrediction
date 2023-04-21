# Hospital Readmission Prediction

### Repository Navigation
[Notebook](https://github.com/ACB-prgm/HospitalReadmissionPrediction/blob/main/base.ipynb) • [Data](https://github.com/ACB-prgm/HospitalReadmissionPrediction/tree/main/dataset_diabetes) • [Presentation](https://github.com/ACB-prgm/HospitalReadmissionPrediction/blob/main/Presentation/Presentation.pdf)

## Abstract
This project focuses on predicting the likelihood of readmission within 30 days of initial discharge for diabetes patients using machine learning algorithms. I developed and evaluated various models, including random forests, SVC with RBF kernel, gradient boosting, and neural networks, to identify high-risk patients and provide targeted interventions to reduce the likelihood of readmission. By analyzing admission, demographic, clinical, medication, and discharge data, we were able to develop a predictive model that can be used by healthcare providers to optimize resource allocation, improve care coordination, and inform policy and practice in the field of diabetes care and readmission prevention. The insights gained from this project demonstrate the potential of data science and machine learning in healthcare and contribute to efforts to improve patient outcomes and reduce healthcare costs.

## Introduction
Hospital readmissions within 30 days of discharge are a common problem, especially for patients with chronic conditions like diabetes. Not only do readmissions negatively impact patient outcomes, but they also increase healthcare costs and strain the capacity of the healthcare system. In response to this issue, Medicare established the Hospital Readmissions Reduction Program (HRRP), which financially penalizes hospitals with higher-than-expected rates of readmissions. To address this problem, I developed machine learning models that can identify patients who are at risk for 30-day readmissions. These models analyze key features and patterns to inform treatment plans and reduce the likelihood of readmissions. By developing and implementing these models, we aim to address the challenges of preventable readmissions, improve patient outcomes, and reduce healthcare costs. Additionally, our models can be used in real-time to predict the probability of a patient's readmission, enabling providers to modify treatment plans dynamically and optimize patient care.

### Project Goals
1. **Develop** a machine learning model to predict <30-day readmission based on patient treatment and discharge data
2. **Identify** key patient and treatment features that are most predictive of readmissions
3. **Provide recommendations** to help reduce the number of preventable readmissions

## Methods
### Data
[Source](https://archive-beta.ics.uci.edu/dataset/296/diabetes+130+us+hospitals+for+years+1999+2008) [1]

#### Overview
The "Diabetes 130-US hospitals for years 1999-2008" dataset from the University of California - Irvine, Machine Learning Repository represents 10 years of clinical care at 130 US hospitals and integrated delivery networks. It includes over 101,000 instances with more than 50 features, such as demographic information, health history, admission and treatment information, and discharge information. 
#### Preparation
The dataset includes three classes for readmission, which were simplified into a binary classification for the purpose of the analysis: Class 0 includes NO and >30 readmissions, while Class 1 includes <30 readmissions. The dataset underwent several preprocessing steps, including de-noising, feature engineering, and undersampling to address class imbalance. These steps were taken to ensure the quality and accuracy of the data and to improve the performance of the machine learning models used in the analysis. The data was split into train (80%) and validation (20%) sets.

<img src="https://user-images.githubusercontent.com/63984796/233452006-59e2a030-5de2-4c83-8b4e-4a1d197c8ee0.png" alt="Undersampling" width="850" align="center"/>

### Models
I developed and evaluated several machine learning models, including Random Forest, SVC (Support Vector Classifier), LightGBM & CatBoost (Gradient Boosting), and Recurrent Neural Network (RNN) on their accuracy, precision, recall, and f-1 scores.

## Results
### Models' Performance
<img src="https://user-images.githubusercontent.com/63984796/233452981-cdf0a7cd-853c-496a-8069-cbe771d10d4a.png" alt="Undersampling" width="850" align="center"/>
After training, tuning, and evaluation, all models achieved similar metrics, indicating that I have likely extracted the maximum amount of information from the dataset. Interestingly, the simple random forest model achieved the same accuracy as the more complex LGBM and Catboost models, but achieved a higher target F-1 score. Given the task of predicting the likelihood of readmission within 30 days of initial discharge for diabetes patients, I prioritized minimizing false negative errors over false positive errors. Therefore, I chose the random forest model for further feature analysis.  
### Feature Importances
<img src="https://user-images.githubusercontent.com/63984796/233454146-b67bd2e4-f3d1-4afd-a750-ac057bd305c7.png" alt="Undersampling" width="850" align="center"/>
It's important to note that the relationships observed between certain variables and the risk of readmission may be correlational rather than causal. For example, hospital visits, the number of diagnoses, the number of medications, and the discharge facility may be secondary to a patient's health status, which may be the primary predictor of readmission risk. However, it's interesting to note that more procedures appear to be associated with a lower risk of readmission, suggesting that this may be a potentially important factor to consider when developing strategies for preventing readmissions, although more research is necessary.  

### Discharge Facilities by Readmission Probability
I realized there were several important features relating to the discharge facility, so I investigated this further:  

<img src="https://user-images.githubusercontent.com/63984796/233455184-93370427-028c-445b-88f6-2faadd98a9e4.png" alt="Undersampling" width="500" align="center"/>

My analysis revealed that rehab facilities had the highest predicted readmission probability, while discharge home had the lowest predicted readmission probability. Additionally, transferring a patient to inpatient care at the same hospital was associated with a lower readmission probability compared to transferring to a different hospital. It's important to note that these findings demonstrate a correlation between certain variables and the risk of readmission, but do not necessarily indicate causation. Nonetheless, these insights can inform clinical decision-making and help me as a healthcare provider develop targeted interventions to reduce the risk of readmissions for diabetes patients.

## Application and Value
### Deployment
The machine learning model can be integrated with electronic health records (EHRs) to assess a patient's risk of readmission in real-time. This integration would allow healthcare providers to intervene early and provide targeted interventions and informed care plans, potentially reducing the risk of readmission for diabetes patients. By leveraging the predictive capabilities of the model and integrating it with EHRs, healthcare providers can take a proactive approach to managing the health of their patients and improving health outcomes.
### Value
The machine learning model's real-time risk assessment capabilities can enable healthcare providers to develop informed care and discharge plans for high-risk patients. This targeted approach to patient care can potentially reduce the risk of readmission for diabetes patients and optimize resource allocation within healthcare systems. In addition, leveraging the predictive capabilities of the model can inform quality improvement (QI) initiatives, allowing healthcare providers to continuously improve patient outcomes and reduce healthcare costs associated with readmissions.

## Conclusion
In conclusion, hospital readmissions within 30 days of discharge negatively impact patient outcomes, decrease profits through Medicare's Hospital Readmissions Reduction Program (HRRP), increase healthcare costs, and strain healthcare system capacity. Our machine learning model provides healthcare providers with a valuable tool to address these challenges by identifying high-risk patients and enabling the development of informed care and discharge plans. By leveraging the model's predictive capabilities, healthcare providers can improve patient outcomes, reduce healthcare costs, and optimize resource allocation, ultimately improving the quality of care for diabetes patients.




















## Sources

[1] Clore,John, Cios,Krzysztof, DeShazo,Jon & Strack,Beata. (2014). Diabetes 130-US hospitals for years 1999-2008. UCI Machine Learning Repository. https://doi.org/10.24432/C5230J.
