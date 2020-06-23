# Starbucks Capstone Challenge

### Project Motivation
The Project is made after understanding the concepts explained in the Udacity Data Science Nanodegree Course.

This data set contains simulated data that mimics customer behavior on the Starbucks rewards mobile app. Once every few days, Starbucks sends out an offer to users of the mobile app. An offer can be merely an advertisement for a drink or an actual offer such as a discount or BOGO (buy one get one free). Some users might not receive any offer during certain weeks. 

Not all users receive the same offer, and that is the challenge to solve with this data set. The task is to create a model that can help to identify whether an offer would be successful if sent to a given user looking at their profile and offer details.

Three datasets are provided in the project in json format which needs to be processed and combined together to create meaningful realtionship in the data.

### Files in the repository

    .
    ├── Starbucks_Capstone_notebook.ipynb #jupyter notebook  
    ├── data                   
    │   ├── transcript.json               # Transcript Dataset
    │   ├── profile.json                  # Customer Profile Dataset
    |   ├── portfolio.json                # Offer description Dataset
    │   └── processed_offers.csv          # Combined Dataset
    └── README.md

### Datasets:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)
 
**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record

### Requirements:
    Python
    Numpy
    Pandas
    Sklearn
    Seaborn
    MatplotLib
    Json 
    Regular expression (re)
