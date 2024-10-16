import pandas as pd

# Load CSV files
ratings = pd.read_csv('ml-latest-small/ml-latest-small/ratings.csv')
tags = pd.read_csv('ml-latest-small/ml-latest-small/tags.csv')
movies = pd.read_csv('ml-latest-small/ml-latest-small/movies.csv')
links = pd.read_csv('ml-latest-small/ml-latest-small/links.csv')

# Merge dataframes
merged_df = ratings.merge(tags, on=['userId', 'movieId'], how='left')
merged_df = merged_df.merge(movies, on='movieId', how='left')
merged_df = merged_df.merge(links, on='movieId', how='left')

# Save the combined dataframe
merged_df.to_csv('combined_data.csv', index=False)