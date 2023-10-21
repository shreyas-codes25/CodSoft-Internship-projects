import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Define a class to encapsulate the Titanic Survival Predictor
class TitanicSurvivalPredictor:
    def __init__(self):
        # Initialize a Random Forest Classifier with 100 estimators and a fixed random state
        self.model = RandomForestClassifier(n_estimators=100, random_state=32)
        
        # Define a list of features used in the dataset
        self.data_set_list = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'PassengerId']

    # Method for preprocessing the dataset
    def preprocess_data(self, data):
        # Handle missing values by filling them with the mean
        imputer = SimpleImputer(strategy='mean')
        data['Age'] = imputer.fit_transform(data[['Age']])
        data['Fare'] = imputer.fit_transform(data[['Fare']])
        
        # Extract the cabin letter and apply one-hot encoding to categorical variables
        data['Cabin'] = data['Cabin'].str.extract('([A-Za-z])')
        data = pd.get_dummies(data, columns=['Sex', 'Embarked', 'Cabin'], drop_first=True)
        
        # Define feature (x) and target (y) variables
        self.x = data[self.data_set_list]
        self.y = data['Survived']

    # Method for training the machine learning model
    def train_model(self):
        self.model.fit(self.x, self.y)

    # Method for making passenger survival predictions
    def predict_passenger_survival(self, new_data):
        # Create a DataFrame for new passenger data and make predictions
        new_data_df = pd.DataFrame([new_data])
        prediction = self.model.predict(new_data_df)
        
        # Return the prediction as "Survive" or "Not Survive"
        return "Survive" if prediction[0] == 1 else "Not Survive"

    # Method to interactively input passenger data and get predictions
    def run_interactive_mode(self):
        pid = 1
        while True:
            try:
                print("Please provide passenger information")
                pclass = int(input("Enter passenger class (1, 2, 3): "))
                age = float(input("Enter passenger age: "))
                Sibsp = int(input("Enter the number of siblings/spouses the passenger has onboard: "))
                parch = int(input("Enter the number of parents or children the passenger has on board: "))
                fare = float(input("Enter the amount paid for the ticket: "))
                sex = input("What is the gender of the passenger (male/female): ").lower()
                embarked = input("Enter the port of embarkation (C/Q/S): ").upper()

                # Create a dictionary for new passenger data
                new_data1 = {
                    'Pclass': pclass,
                    'Age': age,
                    'SibSp': Sibsp,
                    'Parch': parch,
                    'Fare': fare,
                    'Sex_male': 1 if sex == 'male' else 0,
                    'Embarked_Q': 1 if embarked == 'Q' else 0,
                    'Embarked_S': 1 if embarked == 'S' else 0,
                    'PassengerId': pid,
                }

                # Get the prediction for the passenger's survival
                result = self.predict_passenger_survival(new_data1)
                print(f'\nPassenger {pid} will  {result}')

            except (ValueError, KeyError, AssertionError):
                print("Please enter correct values")
                continue

            another = input("Do you want to check for another passenger? (yes/no): ").lower()
            if another != 'yes':
                break
            print('\n')
            pid += 1

# Entry point for the program
if __name__ == "__main__":
    # Create an instance of the TitanicSurvivalPredictor class
    predictor = TitanicSurvivalPredictor()
    
    # Preprocess the dataset, train the model, and run interactive mode for predictions
    predictor.preprocess_data(pd.read_csv('tested.csv'))
    predictor.train_model()
    predictor.run_interactive_mode()
