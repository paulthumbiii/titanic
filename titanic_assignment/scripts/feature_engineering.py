import pandas as pd
import numpy as np

def engineer_features(input_path):
    # Load the cleaned data
    df = pd.read_csv(input_path)
    
    # 1. CREATE DERIVED FEATURES
    # FamilySize = Siblings + Parents + 1
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # IsAlone = 1 if traveling alone
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Title extraction
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    rare_titles = ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # Fare per person
    df['Fare_Per_Person'] = df['Fare'] / df['FamilySize']

    # 2. FEATURE TRANSFORMATIONS
    # Log transform Fare
    df['Fare_Log'] = np.log1p(df['Fare'])
    
    # 3. CATEGORICAL ENCODING
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
    
    # Save the engineered dataset
    output_path = "data/train_engineered.csv"
    df.to_csv(output_path, index=False)
    return output_path

if __name__ == "__main__":
    try:
        path = engineer_features('data/train_cleaned.csv')
        print(f"Feature Engineering complete! File saved to: {path}")
    except FileNotFoundError:
        print("Error: 'data/train_cleaned.csv' not found. Run data_cleaning.py first.")