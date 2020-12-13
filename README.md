# Disaster Response
Disaster Response Project for Udacity Data Science Nanodegree

## Installation & Requirements
* Download or clone repository
* Install the following python packages:
    * Pandas
    * SQLAlchemy
    * scikit-learn
    * nltk
    * flask
    * plotly


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        ```bash
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```
    - To run ML pipeline that trains classifier and saves
        ```bash
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

2. Run the following command in the app's directory to run your web app.
    ```bash
    python run.py
    ```

3. Go to `http://0.0.0.0:3001/`

## Project Components

This project has three main components.

1. ETL Pipeline: `data/process_data.py`  
    * Load messages and categories datasets
    * Merge and clean the data
    * Save the cleaned data in a SQLite database
    
2. Model Training: `models/train_classifier.py`
    * Load messages and categories from SQLite database
    * Build a preprocessing and ML model pipeline
    * Train the model and grid search for optimal hyperparameters
    * Evaluate the model (accuracy, f1 score, precision, recall)
    * Save the model to disk
    
3. Web app `app/run.py`
A flask app that
    * accepts new messages from the user 
    * classifies the user message and provides classification results
    * displays some statistics of the training dataset
