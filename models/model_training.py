import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

# Load training data
train_data = pd.read_csv('../data/raw/train_data.csv')

# Prepare data for Surprise library
reader = Reader(rating_scale=(train_data['rating'].min(), train_data['rating'].max()))
data = Dataset.load_from_df(train_data[['userId', 'movieId', 'rating']], reader)

# Build and evaluate model
algo = SVD()
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Train the model on the whole dataset
trainset = data.build_full_trainset()
algo.fit(trainset)

# Save the trained model
import pickle
with open('recommendation_model.pkl', 'wb') as f:
    pickle.dump(algo, f)