import sys
import warnings
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories from csv files
    :param messages_filepath: path to csv file with messages
    :param categories_filepath: path to csv file with categories
    :return: (messages, categories) DataFrames with messages and categries
    """

    # load messages
    messages = pd.read_csv(messages_filepath, index_col='id')
    messages = messages[~messages.index.duplicated()]

    # load categories and remove entries with duplicate indices
    categories = pd.read_csv(categories_filepath, index_col='id')

    return messages, categories


def clean_data(messages, categories):
    """
    Clean messages and categories data:
        * Drop rows with duplicate indices
        * Append categories as integer columns with 0s and 1s to messages dataframe
    :param messages: DataFrame with messages
    :param categories: DataFrame with categories
    :return:
    """

    # remove messages and categories with duplicate indices
    messages   = messages[~messages.index.duplicated()]
    categories = categories[~categories.index.duplicated()]

    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0, :]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.str.split('-', expand=True)[0]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-', expand=True)[1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # the 'related' category has a few values of 2. set them to the mode of 'related'
    categories.loc[categories["related"] == 2, "related"] = categories["related"].mode().iloc[0]

    # merge messages and categories (inner merge with one-to-one validation to ensure there are categories for each message)
    df = pd.merge(messages, categories, how="left", left_index=True, right_index=True, validate='one_to_one')

    # check for remaining duplicates
    if df.index.duplicated().sum() > 0:
        warnings.warn(f"There are {df.index.duplicated().sum()} duplicates in the index after cleaning", RuntimeWarning)

    return df


def save_data(df, database_filename):
    """
    Save messages and categories in dataframe to SQLite DB table 'messages'
    :param df: DataFrame with messages and category columns
    :param database_filename: Filename of the SQLite DB
    :return:
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages, categories = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages, categories)
        
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