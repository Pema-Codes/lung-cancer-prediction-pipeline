# Lung Cancer Prediction Analysis
High Precision Diagnostic Classification using Logistic Regression & UMAP

## Project Overview:
This project implements a robust machine learning pipeline to predict lung cancer presence based on patient lifestyle and symptom data. This project leverages an optimized Logistic Regression model and advanced preprocessing techniques like SMOTE to achieve high sensitivity, ensuring minimal false negatives in a clinical context.

## Project Structure

lung_cancer_analysis.ipynb: Full EDA, training, and validation logic.

predict.py: Production-ready script for real-time inference.

lung_cancer_pipeline.pkl: The trained Logistic Regression model pipeline.

requirements.txt: Environment dependencies.

## Technical Deep-Dive

-   **Pipeline architecture:** Implemented `sklearn.pipeline` to integrate `StandardScaler` and Logistic Regression, ensuring zero data leakage and deployment-ready code.

-   **Class Imbalance Handling:** Utilized SMOTE (Synthetic Minority Over-sampling Technique) to balance clinical classes, improving model sensitivity.

-   **Feature Correlation:** Identified several symptoms with varying degrees of positive correlation to Lung Cancer, including `YELLOW_FINGERS`, `ANXIETY`, `CHRONIC DISEASE`, `FATIGUE`, `ALLERGY`, `WHEEZING`, `ALCOHOL CONSUMING`, `COUGHING`, `SHORTNESS OF BREATH`, `SWALLOWING DIFFICULTY`, and `CHEST PAIN`.

-   **Manifold Learning:** Employed UMAP (Uniform Manifold Approximation and Projection) to project 15-dimensional symptom data into 2D, visually validating class separation post-SMOTE and scaling.

-   **Optimization:** Executed GridSearchCV for Hyperparameter Tuning (C parameter and penalty types).

-   **Stability Testing:** Validated performance using 10-Fold Cross-Validation to ensure the model generalizes to new patients.

-   **Error Analysis:** Implemented a 'Confidence Check' that identified **Patient 227** as a critical false negative case (10.60% confidence when actual was cancer).
  
## Model Performance

- **Model Selection**: Logistic Regression was chosen as the final model over XGBoost and Random Forest due to its superior stability and interpretability.
- 
- **Optimized Performance**: The Logistic Regression pipeline achieved high performance on a test set with a custom threshold of `0.5433`:
  
    - **Accuracy**: 95%
      
    - **Recall (Lung Cancer)**: 98%
      
    - **Recall (Healthy)**: 75% (improved from 50% with threshold optimization)
      
    - **Mean CV AUC**: 0.9870

**Model Performance Summary**
| Metric              | Value    |
|---------------------|----------|
| Accuracy            | 95.00%   |
| ROC-AUC Score (Mean CV) | 0.9870   |
| Stability (Std Dev) | 0.0300   |
| Precision (Cancer)  | 0.96     |
| Recall (Cancer)     | 0.98     |
| F1-score (Cancer)   | 0.97     |

**Clinical Impact:** The model demonstrated a **98% Sensitivity (Recall)** for Lung Cancer, successfully identifying **47 out of 48** positive cancer cases in the test set. With **2 False Positives** (healthy individuals misclassified as cancer) and **1 False Negative** (cancer case misclassified as healthy), the model maintains high specificity while prioritizing the detection of actual cancer cases.

**Confusion Matrix Analysis:**

As shown in the Confusion Matrix (with optimized threshold), the model highly prioritizes True Positives (47).

-   **Recall (Sensitivity):** The model correctly identified 47 out of 48 cancer cases (98% Sensitivity).

-   **Precision:** 2 healthy patients were misclassified as having cancer (False Positives).

The model demonstrates an elite ability to minimize unnecessary clinical anxiety while maintaining a high catch rate for cancer cases.

## Key Clinical Insights

1.  **Model Selection** While complex boosters like XGBoost was evaluated, Logistic Regression was selected as the final production model. 

Rationale: In clinical settings, "black-box" models are difficult to validate. Logistic Regression provided a high-performing, transparent framework where the Log-Odds of each symptom could be clinically audited. 

Result: Demonstrated a strong linear relationship between reported clinical symptoms and diagnostic outcomes.

2. **Advanced Dimensionality Reduction (UMAP)**

Using UMAP (Uniform Manifold Approximation and Projection), I mapped the 15- dimensional patient profiles into 2-D space.   

Observation: Distinct spatial clustering was observed between the 'Healthy' (Blue) and 'Cancer' (Red) cohorts

Validation: The distinct separation confirms that the selected features (symptoms) are high-qualtiy biological markers for this classification task. 

3. **Strategic Threshold Optimization**

Rather than using the default 0.5 probability, I implemented a Custom Decision Threshold (0.5433). 

Goal: To balance Precision and Recall. 

Clinical Impact: This optimization significantly improved teh identification of the 'Healthy' class while maintaining high sensitivity for detecting 'Lung Cancer' cases, reducing unnecessary clincal alarm. 

4. **Clinical Error Analysis: the "Borderline" Patient**

My pipeline identified Patient 227 as a critical False Negative (Actual: Cancer | Predicted: Healthy).

Finding: The model reported a very low confidence score (10.6%). 

This highlights the necessity of Confidence Score Monitoring. In real-world pipeline, any patinet falling within a "Low Confidence" interval (e.g., 40%-60%) would be automatically flagged for Manual Clinical Review by a Medical Lead.

5. **Top Clinical Predictors** 

The model’s feature importance ranking identified the top 5 indicators for this dataset:

    Chronic Disease (Comorbidity impact)

    Allergy

    Peer Pressure (Social determinant of health)

    Alcohol Consumption

    Fatigue
    
##Future Work

    Integrate additional patient data sources for richer feature engineering.
    Explore advanced interpretability techniques (e.g., SHAP, LIME).
    Develop a user-friendly interface for clinicians to interact with the model.


## How to run

1.  Install dependencies:
    `pip install -r requirements.txt`

2.  Run the analysis:
    Open `lung_cancer_analysis.ipynb` in Jupyter or VS Code.

3.  Test a single prediction:
    `python predict.py`

## Technical Implementation Roadmap

    [x] Data Engineering: Schema normalization, categorical encoding, and null value verification.

    [x] Exploratory Data Analysis (EDA): Correlation heatmaps and symptom distribution profiling.

    [x] Preprocessing: Implemented SMOTE for class balance and StandardScaler for feature normalization.

    [x] Dimensionality Reduction: Utilized UMAP for manifold learning and cluster validation.

    [x] Architecture: Developed a unified sklearn.pipeline for robust data flow.

    [x] Optimization: Performed GridSearchCV for hyperparameter tuning and Threshold Optimization to maximize clinical recall.

    [x] Validation: 10-Fold Cross-Validation and ROC-AUC performance benchmarking.

    [x] Clinical Analysis: In-depth error analysis of "Borderline" cases and model confidence distribution.

    [x] Deployment Ready: Model serialization via joblib and production scripting for real-time inference.
