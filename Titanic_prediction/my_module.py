import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

data = pd.read_csv('tested.csv')

imputer = SimpleImputer(strategy='mean')
data['Age'] = imputer.fit_transform(data[['Age']])
data['Fare'] = imputer.fit_transform(data[['Fare']])

# Extract the cabin letter from the 'Cabin' column
data['Cabin'] = data['Cabin'].str.extract('([A-Za-z])')

# Encode the 'Sex' and 'Embarked' columns using one-hot encoding
data = pd.get_dummies(data, columns=['Sex', 'Embarked', 'Cabin'], drop_first=True)

# Define the features, including 'PassengerId', 'Sex', 'Embarked', and 'Cabin'
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S', 'PassengerId', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E', 'Cabin_F', 'Cabin_G']

x = data[features]
y = data['Survived']

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x, y)

def predict_passenger_survival(new_data):
    new_data_df = pd.DataFrame([new_data])
    prediction = clf.predict(new_data_df)
    return "Survive" if prediction[0] == 1 else "Not Survive"
pid=1
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
        cabin = input("Enter the cabin: ")

        # Extract the cabin letter from the input
        cabin = cabin[0].upper() if cabin else 'X'  # Default to 'X' if no cabin is provided

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
            'Cabin_B': 1 if cabin == 'B' else 0,
            'Cabin_C': 1 if cabin == 'C' else 0,
            'Cabin_D': 1 if cabin == 'D' else 0,
            'Cabin_E': 1 if cabin == 'E' else 0,
            'Cabin_F': 1 if cabin == 'F' else 0,
            'Cabin_G': 1 if cabin == 'G' else 0,
        }

        result = predict_passenger_survival(new_data1)
        print(f'Passenger {pid} will  {result}')

    except (ValueError, KeyError):
        print("Please enter correct values")
        continue

    another = input("Do you want to check for another passenger? (yes/no): ").lower()
    if another != 'yes':
        break
    pid+=1
    
