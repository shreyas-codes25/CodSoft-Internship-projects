import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer



data = pd.read_csv("Movie_Rating_Prediction\IMDbMoviesIndia.csv")

data = data.dropna(subset=["Rating"])
data = data.drop_duplicates(subset=["Name"])
data["Votes"] = data["Votes"].str.replace(',', '', regex=True).astype(float)
data["Year"] = data["Year"].str.extract('(\d+)').astype(float)
data["Duration"] = data["Duration"].str.extract('(\d+)').astype(float)


label_encoders = {}
categorical_features = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]
for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])
    label_encoders[feature] = le


X = data[["Year", "Duration", "Genre", "Votes", "Director", "Actor 1", "Actor 2", "Actor 3"]]
y = data["Rating"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
imputer = SimpleImputer(strategy='mean')


X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
model = LinearRegression()
model.fit(X_train_imputed, y_train)

def predict_movie_rating(movie_data, label_encoders):

    user_movie_features = [
        float(movie_data['Year']),
        label_encoders['Genre'].transform([movie_data['Genre'].capitalize()])[0],
        float(movie_data['Votes']),
        label_encoders['Director'].transform([movie_data['Director']])[0],
        label_encoders['Actor 1'].transform([movie_data['Actor 1']])[0],
        label_encoders['Actor 2'].transform([movie_data['Actor 2']])[0],
        label_encoders['Actor 3'].transform([movie_data['Actor 3']])[0],
        float(movie_data['Duration'])
    ]
    

    movie_rating = model.predict([user_movie_features])
    return movie_rating[0]


user_movie_data = {}
try:
    user_movie_data['Year'] = float(input('Enter the Year: '))
    user_movie_data['Genre'] = input('Enter the Genre: ')
    user_movie_data['Votes'] = float(input('Enter the Votes: '))
    user_movie_data['Director'] = input('Enter the Director: ')
    user_movie_data['Actor 1'] = input('Enter Actor 1: ')
    user_movie_data['Actor 2'] = input('Enter Actor 2: ')
    user_movie_data['Actor 3'] = input('Enter Actor 3: ')
    user_movie_data['Name'] = input('Enter Movie Title: ')
    user_movie_data['Duration'] = float(input('Enter Duration (e.g., 120): '))
except ValueError:
    print("Invalid input. Please enter numeric values for 'Year', 'Votes', and 'Duration'.")
    exit(1)


predicted_rating = predict_movie_rating(user_movie_data, label_encoders)
print(f'\n\n\nThe Predicted Movie May Get a Rating of : {predicted_rating:.2f}\n\n')
