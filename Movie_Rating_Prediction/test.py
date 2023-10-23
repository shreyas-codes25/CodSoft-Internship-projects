import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("Movie_Rating_Prediction\IMDbMoviesIndia.csv")

# Data preprocessing
data = data.dropna(subset=["Rating"])
data = data.drop_duplicates(subset=["Name"])

# Remove the dollar sign and 'M' from the box office column and convert it to float
data["Votes"] = data["Votes"].str.replace(',', '', regex=True).astype(float)

# Extract the year from the 'Year' column and convert it to float
data["Year"] = data["Year"].str.extract('(\d+)').astype(float)

# Convert the duration column to integer
data["Duration"] = data["Duration"].str.extract('(\d+)').astype(float)

# Label encode categorical columns
label_encoders = {}
categorical_features = ["Genre", "Director", "Actor 1", "Actor 2", "Actor 3"]
for feature in categorical_features:
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])
    label_encoders[feature] = le

# Define the features and target variable
X = data[["Year", "Duration", "Genre", "Votes", "Director", "Actor 1", "Actor 2", "Actor 3"]]
y = data["Rating"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Function to predict movie rating
def predict_movie_rating(movie_data, label_encoders):
    # Convert the user input features into the same format as the dataset
    user_movie_features = [
        float(movie_data['Year']),
        label_encoders['Genre'].transform([movie_data['Genre']])[0],
        float(movie_data['Votes']),
        label_encoders['Director'].transform([movie_data['Director']])[0],
        label_encoders['Actor 1'].transform([movie_data['Actor 1']])[0],
        label_encoders['Actor 2'].transform([movie_data['Actor 2']])[0],
        label_encoders['Actor 3'].transform([movie_data['Actor 3']])[0],
        float(movie_data['Duration'])
    ]
    
    # Use the trained model to predict the movie rating
    movie_rating = model.predict([user_movie_features])
    return movie_rating[0]

# User input for movie data with error handling
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

# Predict the movie rating
predicted_rating = predict_movie_rating(user_movie_data, label_encoders)
print(f'Predicted Movie Rating: {predicted_rating:.2f}')
