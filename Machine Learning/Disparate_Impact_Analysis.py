import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def generate_artificial_dataset(num_samples=2000):
    # Create lists to hold generated data
    age_list = []
    gender_list = []
    race_list = []
    income_list = []

    # Define possible values for gender and race
    genders = ['Male', 'Female']
    races = ['White', 'Black', 'Asian', 'Hispanic', 'Other']

    # Generate random data for each sample
    for _ in range(num_samples):
        age = random.randint(18, 65)  # Random age between 18 and 65
        gender = random.choice(genders)  # Random gender
        race = random.choice(races)  # Random race
        
        # Option to not have bias in income based on race
        #income = random.randint(2, 30) * 10000

        # Option to introduce bias in income based on race
        if race == 'Black':
            income = random.randint(2, 20) * 10000  # Lower income range for 'Black'
        else:
            income = random.randint(2, 30) * 10000  # Higher income range for others

        # Append data to respective lists
        age_list.append(age)
        gender_list.append(gender)
        race_list.append(race)
        income_list.append(income)

    # Create a DataFrame from the lists
    df = pd.DataFrame({
        'age': age_list,
        'gender': gender_list,
        'race': race_list,
        'income': income_list
    })

    df['Above 70k'] = df['income'].apply(lambda x: 1 if x > 70000 else 0)

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
    # Generate the artificial dataset
    artificial_dataset = generate_artificial_dataset(num_samples=1000)
    print(artificial_dataset.head())

    # Assuming 'race' is the group of interest
    X = artificial_dataset.drop(columns=['Above 70k'])
    y = artificial_dataset['Above 70k']  # Predict the "Above 70k" column
    group = artificial_dataset['race']

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
    DIA_ratio, bias_result = disparate_impact_analysis(predictions, group_test, favored_group='White', unfavored_group='Black')  
    print(f"DIA Ratio: {DIA_ratio}, \nBias: {bias_result}")

if __name__ == "__main__":
    main()
