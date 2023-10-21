import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

class TitanicSurvivalPredictor:
    
   
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=32)
        self.features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male','Embarked_C' ,'Embarked_Q', 'Embarked_S', 'PassengerId']

    def preprocess_data(self, data):
        imputer = SimpleImputer(strategy='mean')
        data['Age'] = imputer.fit_transform(data[['Age']])
        data['Fare'] = imputer.fit_transform(data[['Fare']])
        data['Cabin'] = data['Cabin'].str.extract('([A-Za-z])')
        data = pd.get_dummies(data, columns=['Sex', 'Embarked', 'Cabin'], drop_first=True)
        self.x = data[self.features]
        self.y = data['Survived']

    def train_model(self):
        self.model.fit(self.x, self.y)

    def predict_passenger_survival(self,new_data):
        new_data_df = pd.DataFrame([new_data])
        prediction =self.model.predict(new_data_df)
        return "Survive" if prediction[0] == 1 else "Not Survive"

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
    
                new_data1 = {
                'Pclass': pclass,
                'Age': age,
                'SibSp': Sibsp,
                'Parch': parch,
                'Fare': fare,
                'Sex_male': 1 if sex == 'male' else 0,
                'Embarked_C':1 if embarked == 'C' else 0,
                'Embarked_Q': 1 if embarked == 'Q' else 0,
                'Embarked_S': 1 if embarked == 'S' else 0,
                'PassengerId': pid,
            
                }

                result = self.predict_passenger_survival(new_data1)
                print(f'\nPassenger {pid} will  {result}')

            except (ValueError, KeyError, AssertionError):
                print("Please enter correct values")
                continue

            another = input("Do you want to check for another passenger? (yes/no): ").lower()
            if another != 'yes':
                break
            print('\n')
            pid+=1


if __name__ == "__main__": 
    predictor = TitanicSurvivalPredictor()
    predictor.preprocess_data(pd.read_csv('tested.csv'))
    predictor.train_model()
    predictor.run_interactive_mode()
