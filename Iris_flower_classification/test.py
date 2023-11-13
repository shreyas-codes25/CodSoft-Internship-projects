import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = data['species']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def train_model(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def get_user_input():
    sepal_length = float(input("Enter sepal length: "))
    sepal_width = float(input("Enter sepal width: "))
    petal_length = float(input("Enter petal length: "))
    petal_width = float(input("Enter petal width: "))
    return sepal_length, sepal_width, petal_length, petal_width

def standardize_user_input(user_input, scaler):
    user_input_df = pd.DataFrame([user_input], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    user_input_scaled = scaler.transform(user_input_df)
    return user_input_scaled

def make_prediction(model, user_input_scaled):
    prediction = model.predict(user_input_scaled)
    return prediction[0]

def main():
    file_path = 'Iris_flower_classification\irisDataset.csv'
    data = load_data(file_path)  
    X, y, scaler = preprocess_data(data)    
    model = train_model(X, y)    
    user_input = get_user_input()    
    user_input_scaled = standardize_user_input(user_input, scaler)    
    prediction = make_prediction(model, user_input_scaled)    
    print(f"The species is: {prediction}")

if __name__ == "__main__":
    main()
