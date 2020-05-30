import sys
# import libraries
import pandas as pd
import numpy as np
import pickle
import re
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt','wordnet','stopwords'])
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report,accuracy_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    engine = create_engine(database_filepath)
    df = pd.read_sql_table(database_filepath.rsplit('/',1)[1].split('.')[0], engine)
    X = df['message'].values
    Y = df[df.columns[4:]].values
    category_names = list(df.columns[4:])
    
    return X, Y, category_names

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    lemmatizer=WordNetLemmatizer()
    # Detect URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    # Normalize and tokenize
    tokens = nltk.word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens

def build_model():
    #create a pipeline
    pipeline = Pipeline([
    ('count',CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC(random_state=0))))
    ])
    
    parameters = {
    'count__ngram_range':((1,1),(1,2)),
    'tfidf__smooth_idf':[True, False],
    'clf__estimator__estimator__C':[1,0.1,5]
    }
    
    cv = GridSearchCV(pipeline,param_grid=parameters, cv =5, verbose=0)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred=pipeline.predict(X_test)
    for column in range(0,Y_test.shape[1]):
    print(column)
    print(classification_report(Y_test[:,column], y_pred[:,column], output_dict=True))



def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()