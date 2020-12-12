import sys
import pandas as pd
import pickle
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization


def load_data(database_filepath):
    """
    Load messages and categories from SQLite DB
    :param database_filepath: relative SQLite DB filepath
    :return: X, Y, category_names: DataFrames with messages and categories and a list with category names
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('messages', con=engine)
    X = df[['message']]
    Y = df.iloc[:, 4:]
    categoriy_names = Y.columns
    return X, Y, categoriy_names


def tokenize(text):
    """Tokenize the provided text by
            * removing punctuation
            * tokenizing into words
            * removing stopwords
            * lemmatizing, including conversion to lowercase
        """

    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # tokenize into words
    tokens = word_tokenize(text)

    # remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words("english")]

    # lemmatize, including conversion to lowercase
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

    return clean_tokens


def build_model():
    """
    Construct ML pipeline with count vectorization, TFIDF transformation and a MultiOuput RandomForest classifier
    :return: ML pipeline
    """
    pipeline = Pipeline([
        ('countVectorizer', CountVectorizer(tokenizer=tokenize, max_df=0.75, max_features=500)),
        ('TFIDF', TfidfTransformer()),
        ('MultiRandomForest', MultiOutputClassifier(RandomForestClassifier(n_estimators=100, min_samples_split=4), n_jobs=4))
    ])
    return pipeline


def grid_search(model):
    """
    Perform a grid search on the model to optimize parameters.
    The paramters space searched is very limited because runtimes are long.
    For production use this should be optimized!
    :return: GridSearchCV object
    """
    parameters = {
        'MultiRandomForest__estimator__n_estimators': [100, 200],
        'MultiRandomForest__estimator__min_samples_split': [4, 5],
    }

    # Score by f1 score for a balanced model between precision and recall
    f1_weighted_score = make_scorer(f1_score, average='weighted', zero_division=0)

    cv = GridSearchCV(model, param_grid=parameters, scoring=f1_weighted_score, n_jobs=1, cv=2)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Predict the test data with the given model and evalutate the result
    :param model: ML pipeline
    :param X_test: Test data messages
    :param Y_test: Test data categories
    :param category_names: category names of Y_test
    :return: None
    """
    Y_pred = model.predict(X_test)
    print(f"Model accuracy: {accuracy_score(Y_test, Y_pred):.2f}")
    report = classification_report(Y_test, Y_pred, target_names=category_names, zero_division=0)
    print(report)


def save_model(model, model_filepath):
    """
    Pickle ML model to disk
    :param model: ML pipeline
    :param model_filepath: path to pickled file
    :return: None
    """
    with open(model_filepath, 'wb') as outfile:
        pickle.dump(model, outfile)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        model = grid_search(model)
        
        print('Training model...')
        model.fit(X_train["message"], Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test["message"], Y_test, category_names)

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