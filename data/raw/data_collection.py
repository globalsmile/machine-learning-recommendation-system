import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Clean data
def clean_data(data):
    data.dropna(inplace=True)
    return data

# Split data
def split_data(data):
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    return train, test

if __name__ == '__main__':
    # Example using the MovieLens dataset
    data = load_data('combined_data.csv')
    cleaned_data = clean_data(data)
    train_data, test_data = split_data(cleaned_data)
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)