import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    original_categories = categories.copy()
    df_new = pd.merge(messages,categories,on='id')
    categories = df_new['categories'].str.split(';',expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.str.split('-').transform(lambda x: x[0])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    #categories = categories.related.replace(2,1,inplace=True)
    
    #drop the original categories column from `df`
    df_new = df_new.drop('categories',1)
    
    # concatenate the original dataframe with the new `categories` dataframe
    categories = pd.concat([original_categories,categories],axis=1).drop('categories',1)
    
    df_new = pd.merge(df_new,categories,on='id')
    df_new = df_new.drop(df_new[df_new.related >1].index)
    return df_new


def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    table_name = database_filename.split("/")[1].replace(".db","")
    df.to_sql(table_name, engine, index=False, if_exists='replace')


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