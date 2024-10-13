import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import pickle

# Load training data
train_data = pd.read_csv('../data/raw/train_data.csv')

# Prepare data for Surprise library
reader = Reader(rating_scale=(train_data['rating'].min(), train_data['rating'].max()))
data = Dataset.load_from_df(train_data[['userId', 'movieId', 'rating']], reader)

# Build and evaluate model
algo = SVD()
cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Train the model on the whole dataset
trainset = data.build_full_trainset()
algo.fit(trainset)

# Save the trained model
with open('recommendation_model.pkl', 'wb') as f:
    pickle.dump(algo, f)

# Additional evaluation
testset = trainset.build_testset()
predictions = algo.test(testset)

# Extract true labels and predictions
y_true = [pred.r_ui for pred in predictions]
y_pred = [pred.est for pred in predictions]

# Calculate RMSE and MAE
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)

print(f'RMSE: {rmse}')
print(f'MAE: {mae}')