import sys
import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization

app = Flask(__name__)


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


if len(sys.argv) == 3:
    database_filepath, model_filepath = sys.argv[1:]
else:
    print('Please provide the filepath of the disaster messages database ' \
          'as the first argument and the filepath of the pickled model file' \
          'as the second argument. \n\nExample: python ' \
          'run.py ../data/DisasterResponse.db classifier.pkl')
    sys.exit(1)

# load data
engine = create_engine(f'sqlite:///{database_filepath}')
df = pd.read_sql_table('messages', engine)


# load model
model = joblib.load(model_filepath)


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()