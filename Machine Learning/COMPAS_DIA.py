import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def load_compas_data():
    # Define the URL of the dataset
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"

    # Load the dataset
    df = pd.read_csv(url)

    # Select the features we are interested in
    df = df[['age', 'sex', 'race', 'two_year_recid']]

    # Mapping 'sex' from categorical to binary
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})

    return df

def disparate_impact_analysis(predictions, group_test, favored_group, unfavored_group):
    outcomes_df = pd.DataFrame({'group': group_test, 'outcome': predictions})

    # Count the number of positive outcomes for each group
    favored_positive = outcomes_df[(outcomes_df['group'] == favored_group) & (outcomes_df['outcome'] == 1)].shape[0]
    unfavored_positive = outcomes_df[(outcomes_df['group'] == unfavored_group) & (outcomes_df['outcome'] == 1)].shape[0]

    # Calculate the Disparate Impact Analysis (DIA) ratio
    DIA_ratio = unfavored_positive / favored_positive

    # Determine if the model is likely to be biased or not based on DIA ratio
    if DIA_ratio < 0.8 or DIA_ratio > 1.2:
        bias_result = 'Likely To Be Biased'
    else:
        bias_result = 'Not Likely To Be Biased'
        
    return DIA_ratio, bias_result

def main():
    # Load the COMPAS dataset
    compas_dataset = load_compas_data()
    print(compas_dataset['race'].unique())

    # Assuming 'race' is the group of interest
    X = compas_dataset.drop(columns=['two_year_recid', 'race'])
    y = compas_dataset['two_year_recid']  # Predict likelihood of recidivism
    group = compas_dataset['race']

    # Convert categorical variables into numeric using LabelEncoder
    le = LabelEncoder()
    X = X.apply(le.fit_transform)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(X, y, group, test_size=0.2, random_state=42)

    # Train a Logistic Regression model on the training data
    model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    # Make predictions on the test data using the trained model
    predictions = model.predict(X_test)

    # Perform Disparate Impact Analysis (DIA) to check for bias
    DIA_ratio, bias_result = disparate_impact_analysis(predictions, group_test, favored_group='Other', unfavored_group='Asian')  
    print(f"DIA Ratio: {DIA_ratio}, \nBias: {bias_result}")

if __name__ == "__main__":
    main()
