# Disaster Response Pipeline Project

### Project Motivation
As per the requirements for this project I have tried to understand the classes for data engineering and apply it to create a disaster response message classifier using the data provided by Figure Eight.
___
### Files in the repository
    .
    ├── app     
    │   ├── run.py                           # Initializer for flask app
    │   └── templates   
    │       ├── go.html                      # Results page 
    │       └── master.html                  # Home page
    ├── data                   
    │   ├── disaster_categories.csv          # Categories Dataset
    │   ├── disaster_messages.csv            # Messages Dataset
    │   └── process_data.py                  # Wrangling data
    ├── models
    │   └── train_classifier.py              # MLP model
    └── README.md
___
### Requirements:
Python, Flask, Plotly, NLTK, Pandas, Sklearn, SQLalchemy
___
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
