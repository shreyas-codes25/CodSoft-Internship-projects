import pandas as pd
data = {
    'name': ["Movie 1", "Movie 2", "Movie 3", "Movie 4"],
    'year': ["(2020)", "(2019)", "(2022)", "(2018)"],
    'duration': ["120 min", "105 min", "130 min", "95 min"],
    'genre': ["Action", "Drama", "Comedy", "Sci-Fi"],
    'rating': [7.5, 8.2, 6.9, 7.8],
    'votes': [1000, 1500, 800, 1200],
    'director': ["Director 1", "Director 2", "Director 3", "Director 4"],
    'actor 1': ["Actor A", "Actor B", "Actor C", "Actor D"],
    'actor 2': ["Actor E", "Actor F", "Actor G", "Actor H"],
    'actor 3': ["Actor I", "Actor J", "Actor K", "Actor L"]
}

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)