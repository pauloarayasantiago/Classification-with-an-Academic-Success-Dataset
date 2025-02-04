# Student Academic Risk Prediction

This project implements a machine learning solution to predict academic outcomes (Graduate, Dropout, or Enrolled) for higher education students using data available at the time of enrollment. The goal is to enable early identification of at-risk students, allowing for timely intervention strategies.

## Project Overview

The project uses a comprehensive dataset containing student information across multiple dimensions:
- Academic performance indicators
- Demographic information
- Socio-economic factors
- Educational background
- Family background
- Economic indicators (GDP, Unemployment rate, Inflation rate)

## Technical Implementation

### Data Processing Pipeline

1. **Feature Engineering**
   - Created age groups using custom binning (15-20, 21-25, 26-30, 31-35, 36-40, 41-50)
   - Generated admission grade quartiles
   - Implemented polynomial feature generation (degree 2) for numerical variables
   - Applied one-hot encoding for categorical variables

2. **Data Balancing**
   - Utilized SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance
   - Generated synthetic samples for minority classes to achieve balanced class distribution

3. **Model Architecture**
   - Implemented LightGBM classifier with optimized hyperparameters
   - Applied regularization (L1 and L2) to prevent overfitting
   - Utilized early stopping during training

### Key Model Parameters

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

## Key Insights

### Feature Impact Analysis

1. **Demographic Factors**
   - Age at enrollment shows significant correlation with academic outcomes
   - Gender distribution analysis revealed statistical significance (confirmed through Chi-Square test)
   - Scholarship status demonstrates notable impact on graduation rates

2. **Academic Indicators**
   - Strong correlation between first and second semester performance
   - Admission grades show clear relationship with final academic outcomes
   - Curricular units completion rate in early semesters highly indicative of final outcome

3. **Socio-Economic Factors**
   - Scholarship holders showed distinct patterns in academic outcomes
   - Economic indicators (GDP, unemployment, inflation) demonstrated meaningful correlations
   - Parents' qualifications and occupations showed notable influence on student outcomes

## Technical Insights

### Effective Techniques

1. **Feature Engineering Impact**
   - Polynomial features significantly improved model performance
   - Age grouping and admission grade binning enhanced predictive power
   - One-hot encoding effectively handled categorical variables

2. **Model Optimization**
   - SMOTE application improved model balance and performance
   - Custom hyperparameter tuning yielded optimal results
   - Early stopping prevented overfitting while maintaining performance

### Unsuccessful Approaches

Several techniques were tested but did not improve model performance:
- Various scaling methods (Min-Max, Standard, Robust)
- Principal Component Analysis (PCA)
- Higher degree polynomial features
- More granular age grouping
- Alternative encoding techniques
- Outlier removal
- Feature selection by importance threshold

## Data Distribution Insights

The analysis revealed several important patterns:
- Clear correlation between early semester performance and final outcomes
- Significant impact of age groups on academic success
- Notable influence of scholarship status on retention
- Gender-based variations in academic outcomes
- Strong relationship between admission grades and graduation likelihood

## Future Improvements

Potential areas for enhancement:
1. Implement feature selection based on domain knowledge
2. Explore deep learning approaches for complex pattern recognition
3. Develop time-series analysis for semester-by-semester prediction
4. Incorporate additional external economic indicators
5. Implement cross-validation for more robust model evaluation

## Requirements

- Python 3.x
- Key libraries: 
  - flaml
  - lightgbm
  - scikit-learn
  - imbalanced-learn
  - pandas
  - numpy
  - matplotlib
  - seaborn

## Usage

The project includes a complete pipeline from data preprocessing to model evaluation:

1. Data preparation and feature engineering
2. Model training with optimized parameters
3. Evaluation on validation set
4. Prediction generation for test data

## Model Performance

The model achieved significant accuracy in predicting student outcomes, with robust performance across all classes after addressing class imbalance through SMOTE.

---
For detailed implementation and analysis, please refer to the Jupyter notebook in the repository.
