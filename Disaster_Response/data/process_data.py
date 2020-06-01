import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
	'''
    input:
        messages_filepath: Message data filepath.
        categories_filepath: Category data filepath.
    output:
        merged_data: merged data of the two provided datasets.
    '''
    #read message data
    messages_data = pd.read_csv(messages_filepath) 
    # read category data
    categories_data = pd.read_csv(categories_filepath)
    #merge the two datasets
    merged_data = pd.merge(messages_data, categories_data, on='id')
    
    return merged_data


def clean_data(df):
	'''
	input:
		df: mergerd dataframe obtained from load_data().
	output:
		df: cleaned dataframe.
	'''
    #separating the categories into individual columns
    categories = pd.DataFrame(df.categories.str.split(';', expand=True))
    
    row = categories.iloc[0]
    category_colnames = [category.split('-')[0] for category in row]
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(1)
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    df = pd.concat([df.drop(columns=['categories']), categories], axis=1)
    
    return df


def save_data(df, database_filename):
	'''
	input:
		df: cleaned dataframe obtained from clean_data().
		database_filename: name for database file. example: Database.db
	output:
		None
	'''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(database_filename.split('.')[0], engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()