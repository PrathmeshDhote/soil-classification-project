import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def train_parameter_model():
    # Load dataset with specified data types
    data_path = 'data/soil_data.csv'
    data = pd.read_csv(data_path, dtype={'SoilType': 'category'})

    # Print the first few rows of the DataFrame to check data
    print(data.head())

    # Handle missing values
    data.fillna(data.mean(numeric_only=True), inplace=True)  # Only fill numeric columns

    # Features and target
    features = ['pH', 'OrganicCarbon', 'Nitrogen', 'Phosphorus', 'Potassium', 'Sand', 'Silt', 'Clay']
    X = data[features]
    y = data['SoilType']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Test accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Parameter Model Accuracy: {accuracy:.2f}")

    # Save model
    with open('soil_params_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Parameter model saved as soil_params_model.pkl")

if __name__ == "__main__":
    train_parameter_model()
