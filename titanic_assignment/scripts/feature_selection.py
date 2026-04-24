import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def select_features(input_path):
    df = pd.read_csv(input_path)
    
    # Select only numeric columns for selection
    X = df.select_dtypes(include=['number']).drop(['Survived', 'PassengerId'], axis=1)
    y = df['Survived']
    
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Get feature importance
    importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop Selected Features:")
    print(importance.head(5))
    return importance

if __name__ == "__main__":
    select_features('data/train_engineered.csv')