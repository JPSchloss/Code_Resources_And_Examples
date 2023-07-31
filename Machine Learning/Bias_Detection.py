# Python code to identify bias in ML predictions using Adult Income dataset
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def calculate_residuals(y_test, predictions, group_test):
    residuals = y_test - predictions
    residuals_df = pd.DataFrame({'group': group_test, 'residuals': residuals})
    
    results = {}
    # Check for significant discrepancies in residuals across groups
    groups = residuals_df['group'].unique()
    group_residuals = [residuals_df[residuals_df['group'] == grp]['residuals'] for grp in groups]

    for grp, res in zip(groups, group_residuals):
        results[str(grp)] = res.mean()

    # Use ANOVA to check if the difference in residuals is significant
    F, p_value = f_oneway(*group_residuals)

    # If the p-value is less than 0.05, we consider the difference to be significant
    if p_value < 0.05:
        bias_result = 'Likely To Be Biased'
    else:
        bias_result = 'Not Likely To Be Biased'
        
    return results, bias_result

def disparate_impact_analysis(predictions, group_test, favored_group, unfavored_group):
    outcomes_df = pd.DataFrame({'group': group_test, 'outcome': predictions})

    # Calculate the mean outcome for the favored and unfavored groups
    favored_outcome = outcomes_df[outcomes_df['group']==favored_group]['outcome'].mean()
    unfavored_outcome = outcomes_df[outcomes_df['group']==unfavored_group]['outcome'].mean()

    # If the following ratio is outside 0.8-1.2, model might be biased
    DIA_ratio = unfavored_outcome / favored_outcome

    # Determine if the model is likely to be biased or not
    if DIA_ratio < 0.8 or DIA_ratio > 1.2:
        bias_result = 'Likely To Be Biased'
    else:
        bias_result = 'Not Likely To Be Biased'
        
    return DIA_ratio, bias_result

# Load the Census Income dataset from UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
                'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 
                'income']
df = pd.read_csv(url, names=column_names, na_values=' ?', skipinitialspace=True)

# Preprocess the data
df = df.dropna()
df['income'] = df['income'].apply(lambda x: 1 if x=='>50K' else 0)

# Recategorize 'race' into 'White' and 'Non-White'
df['race'] = df['race'].apply(lambda x: 'White' if x=='White' else 'Non-White')

# Assuming 'race' is the group of interest
X = df.drop(columns=['income', 'race'])
y = df['income']
group = df['race']

# Convert categorical variables into numeric
le = LabelEncoder()
X = X.apply(le.fit_transform)

# Split and train the model
X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(X, y, group, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
predictions = model.predict(X_test)

# Check residuals
results, bias_result = calculate_residuals(y_test, predictions, group_test)
print(f"Residulas: {results}, \nBias: {bias_result}")

# Check disparate impact
DIA_ratio, bias_result = disparate_impact_analysis(predictions, group_test, 'White', 'Non-White')  
print(f"DIA Ratio: {DIA_ratio}, \nBias: {bias_result}")


