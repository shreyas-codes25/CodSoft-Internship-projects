import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Extract features and labels, and standardize the features."""
    X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = data['species']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def train_model(X, y):
    """Train a RandomForestClassifier on the given data."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def get_user_input():
    """Get input from the user."""
    sepal_length = float(input("Enter sepal length: "))
    sepal_width = float(input("Enter sepal width: "))
    petal_length = float(input("Enter petal length: "))
    petal_width = float(input("Enter petal width: "))
    return sepal_length, sepal_width, petal_length, petal_width

def standardize_user_input(user_input, scaler):
    """Standardize the user input using the provided scaler."""
    user_input_df = pd.DataFrame([user_input], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    user_input_scaled = scaler.transform(user_input_df)
    return user_input_scaled

def make_prediction(model, user_input_scaled):
    """Make a prediction using the trained model and standardized user input."""
    prediction = model.predict(user_input_scaled)
    return prediction[0]

def main():
    file_path = 'Iris_flower_classification\irisDataset.csv'

    # Load data
    data = load_data(file_path)

    # Preprocess data
    X, y, scaler = preprocess_data(data)

    # Train model
    model = train_model(X, y)

    # Get user input
    user_input = get_user_input()

    # Standardize user input
    user_input_scaled = standardize_user_input(user_input, scaler)

    # Make prediction
    prediction = make_prediction(model, user_input_scaled)

    # Display the predicted species
    print(f"The predicted species is: {prediction}")

if __name__ == "__main__":
    main()
