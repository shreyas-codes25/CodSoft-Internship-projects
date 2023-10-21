import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

data = pd.read_csv('tested.csv')

imputer = SimpleImputer(strategy='mean')
data['Age'] = imputer.fit_transform(data[['Age']])
data['Fare'] = imputer.fit_transform(data[['Fare']])

data = data.drop(columns=['Cabin'])
data = pd.get_dummies(data,columns=['Sex'],drop_first=True)

features=['Pclass','Age','SibSp','Parch','Fare','Sex_male']
x=data[features]
y=data['Survived']
clf=RandomForestClassifier(n_estimators=100,random_state=42)
clf.fit(x,y)

def predict_passenger_survival(new_data):
    new_data_df = pd.DataFrame([new_data])
    prediction=clf.predict(new_data_df)
    return "Survived" if prediction[0]==1 else "Not Survived"

while True:
    try:
        print("Please provide passenger information")
        pclass =int(input("Enter passenger class (1,2,3) :"))
        age = float(input("Enter passenger age : "))
        Sibsp = int(input("Enter the no. of siblings/ spouses the passenger has onboard : "))
        parch=int(input("Enter the no. of parents or childern the passenger has on board : "))
        fare = float(input("Enter the amont paid for the ticket : "))
        sex=input("what is the gender of the passenger : ").lower()
        
        new_data1={
            'Pclass':pclass,
            'Age':age,
            'SibSp':Sibsp,
            'Parch':parch,
            'Fare':fare,
            'Sex_male': 1 if sex == 'male' else 0
        }
        result = predict_passenger_survival(new_data1)
        print(f'prediction survival:{result}')
        
    except(ValueError,KeyError):
        print("please enter correct values")
        continue
    another = input("do you want to check for another passenger ? (yes/no) : ").lower()
    if another!='yes':
        break
    