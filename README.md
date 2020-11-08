# Disaster Response Pipeline

This is a webapp that provides a visualization tool for categorizing a message in 36 different categories. 

## Background and motivation

This project is part of Udacity's Data Science nanodegree. Its aim is 

1. Clean and use data gathered from two datasets and store it in a database;
2. Build and train a Machine Learning model using Natural Language Process that is able to classify a message into 36 different categories
3. Use this trained moded to classify a message inputed by user.

This project is divided in the following key sections:

- Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB (data folder)
- Machine learning pipeline to train the which can classify text message in 36 categories (model folder)
- Web app which can show model results for an inputed message (app folder)

## Running the Project

### Instructions:
(this part of the text was extracted from the exercise)

1. Run the following commands in the project's root directory to set up database and model:

    - To run ETL pipeline that cleans data and stores in database
    
        ```
        python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```
        
    - To run ML pipeline that trains classifier and saves it to a model
    
        ```
        python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
