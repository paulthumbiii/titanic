import pandas as pd

def clean_data(input_path):
    df = pd.read_csv(input_path)
    
    # 1. Missing Value Handling
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # 2. Outlier Handling (Cap Fare)
    fare_cap = df['Fare'].quantile(0.95)
    df['Fare'] = df['Fare'].clip(upper=fare_cap)
    
    # 3. Data Consistency
    df['Sex'] = df['Sex'].str.lower()
    
    # Save cleaned data
    output_path = "data/train_cleaned.csv"
    df.to_csv(output_path, index=False)
    return output_path

if __name__ == "__main__":
    print(f"Cleaning complete: {clean_data('data/train.csv')}")