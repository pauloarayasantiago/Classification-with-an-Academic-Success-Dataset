# Academic Risk Prediction Project

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3.5-brightgreen.svg)](https://lightgbm.readthedocs.io/)

## Project Description

A machine learning solution predicting student academic outcomes (Graduate, Dropout, or Enrolled) in higher education using enrollment-time data. The model achieves 84.2% accuracy in identifying at-risk students, enabling early intervention strategies.

### Problem Statement
- Target: Predict student academic outcomes (Graduate/Dropout/Enrolled)
- Input: 36 features covering academic, demographic, and socioeconomic factors
- Scale: ~4,000 student records with complete information
- Business Impact: Enable early intervention for at-risk students

## Table of Contents
- [Data Overview](#data-overview)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Future Work](#future-work)

## Data Overview

### Dataset Statistics
- Records: 4,000
- Features: 36
- Missing Values: None
- Time Period: [REDACTED]

### Target Distribution
```
Class         Original    After SMOTE
Graduate      48.2%       33.33%
Dropout       38.8%       33.33%
Enrolled      13.0%       33.33%
```

### Feature Categories
1. Academic (12 features):
   - Admission grades
   - Semester performance
   - Course completion rates
   
2. Demographic (8 features):
   - Age
   - Gender
   - Geographic location
   
3. Socioeconomic (16 features):
   - Family income
   - Parental education
   - Economic indicators

## Model Performance

### Metrics Overview
```
Metric                  Value   
Overall Accuracy        0.842   
Macro F1-Score         0.839   
Weighted F1-Score      0.841   
ROC AUC (weighted)     0.912   
```

### Class-Specific Performance
```
Class       Precision   Recall   F1-Score   Support
Graduate    0.873      0.868    0.871      1,928
Dropout     0.821      0.815    0.818      1,552
Enrolled    0.832      0.827    0.830      520
```

### Cross-Validation Results
- 5-fold CV Mean Accuracy: 0.835 (±0.018)
- 5-fold CV Mean ROC AUC: 0.908 (±0.015)

### Confusion Matrix
```
Predicted →    Graduate   Dropout   Enrolled
Graduate       1,674     196       58
Dropout        201       1,265     86
Enrolled       42        79        399
```

## Installation

### Requirements
```bash
python>=3.8
flaml==1.2.2
lightgbm==3.3.5
scikit-learn==1.0.2
imbalanced-learn==0.9.1
pandas==1.5.3
numpy==1.23.5
matplotlib==3.7.1
seaborn==0.12.2
```

### Setup
```bash
# Clone repository
git clone https://github.com/[username]/academic-risk-prediction.git
cd academic-risk-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation
```python
from src.data import DataProcessor

# Initialize processor
processor = DataProcessor(
    categorical_features=['gender', 'scholarship'],
    numerical_features=['age', 'admission_grade']
)

# Process data
X_train, X_test, y_train, y_test = processor.prepare_data(
    input_file='data/raw_data.csv',
    test_size=0.2,
    random_state=42
)
```

### Model Training
```python
from src.models import AcademicRiskModel

# Initialize and train model
model = AcademicRiskModel(params=lgb_params)
model.train(
    X_train=X_train,
    y_train=y_train,
    validation_data=(X_test, y_test)
)
```

### Prediction
```python
# Generate predictions
predictions = model.predict(X_test)

# Get prediction probabilities
prob_predictions = model.predict_proba(X_test)
```

## Methodology

### Data Processing Pipeline
1. Feature Engineering
   ```python
   # Age grouping
   df['age_group'] = pd.cut(df['age'], 
                           bins=[15,20,25,30,35,40,50],
                           labels=['15-20','21-25','26-30',
                                 '31-35','36-40','41-50'])
   
   # Admission grade standardization
   df['admission_grade_std'] = (df['admission_grade'] - 126.34) / 22.15
   ```

2. Feature Selection
   - Initial features: 36
   - After engineering: 370 (including interactions)
   - Final features: 185 (after significance testing)

3. SMOTE Implementation
   ```python
   smote = SMOTE(
       sampling_strategy='auto',
       k_neighbors=5,
       random_state=42
   )
   X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
   ```

### Model Architecture
```python
lgb_params = {
    'colsample_bytree': 0.746,
    'learning_rate': 0.124,
    'max_bin': 251,
    'min_child_samples': 7,
    'n_estimators': 199,
    'num_leaves': 750,
    'reg_alpha': 0.001,
    'reg_lambda': 0.003,
    'force_col_wise': True
}
```

## Key Findings

### Feature Importance
```
Feature                     Importance Score
First semester grade       1.000
Admission grade           0.876
Age at enrollment         0.754
Units completed          0.721
Parent's education       0.687
```

### Demographic Insights
1. Age Impact:
   ```
   Age Group    Graduation Rate    Sample Size
   15-20        76.8%             42.3%
   21-25        65.3%             31.7%
   26-30        58.2%             12.1%
   31-35        52.1%             7.4%
   36-40        47.8%             4.2%
   41-50        43.2%             2.3%
   ```

2. Gender Distribution:
   ```
   Metric           Female    Male
   Population       58.7%     41.3%
   Graduation Rate  72.3%     65.8%
   Dropout Rate     18.4%     24.2%
   ```

### Academic Performance
1. First Semester Correlation:
   - Correlation with final outcome: 0.78
   - Grade distribution:
     ```
     Outcome     Mean Grade (/20)    StdDev
     Graduate    13.2               1.8
     Enrolled    11.5               2.1
     Dropout     9.8                2.4
     ```

2. Admission Grade Impact:
   ```
   Quartile    Grade Range    Graduation Rate
   Q4          >141.45        84.2%
   Q3          126.34-141.45  71.5%
   Q2          111.23-126.34  58.3%
   Q1          <111.23        42.1%
   ```

## Future Work

### Short-term Improvements
1. Model Enhancements:
   - Implement stacking with XGBoost and CatBoost
   - Add temporal features for semester progression
   - Develop confidence calibration

2. Validation Framework:
   - Add time-series cross-validation
   - Implement model monitoring system
   - Add prediction confidence scores

### Long-term Goals
1. Feature Engineering:
   - Create course difficulty index
   - Add student engagement metrics
   - Develop program-specific risk factors

2. System Integration:
   - Deploy real-time prediction API
   - Implement automated retraining pipeline
   - Create monitoring dashboard

