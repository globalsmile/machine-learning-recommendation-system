# Machine Learning Recommendation System
## Project Description
### Overview
This project involves building a machine learning recommendation system, including data collection and preparation, model training, and model evaluation and validation.

### Goals
- Collect and prepare high-quality data for training.
- Develop and train a machine learning model to meet specific business requirements.
- Evaluate and validate the model to ensure it meets performance criteria.
- Deploy a robust and scalable model for production use.
### Deliverables
- Cleaned and processed dataset (can be found in data/raw/train_data.csv).
- Trained machine learning model (can be found in models/recommedation_model.pkl).
- Deployment-ready model.
## Installation
### Clone the repository:
```git clone https://github.com/globalsmile/machine-learning-recommendation-system.git
cd machine-learning-recommendation-system
```
### Create and activate a virtual environment:
```python -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
```
### Install the dependencies:
```pip install -r requirements.txt```
## Usage
### Train the model:
```python models/model_training.py```
### Load and use the trained model:
```
with open('models/recommendation_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Example usage
user_id = 1
item_id = 1
prediction = model.predict(user_id, item_id)
print(f'Predicted rating for user {user_id} and item {item_id}: {prediction.est}')
```
## Evaluation Metrics
### Cross-validation Results (5 splits)
- RMSE (Mean): 0.5911, Std: 0.0353
- MAE (Mean): 0.3789, Std: 0.0145
### Additional Evaluation
- RMSE: 0.4029
- MAE: 0.2454
## Contributing
- Fork the repository.
- Create your feature branch (git checkout -b feature/your-feature).
- Commit your changes (git commit -m 'Add some feature').
- Push to the branch (git push origin feature/your-feature).
- Open a pull request.