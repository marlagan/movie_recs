from copy import deepcopy

import nltk
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
import json

#collaborative filtering recommendation system using pytorch

#for tokenization
nltk.download('punkt_tab')

#for lemmanization
nltk.download('wordnet')

#downloading stopwords
nltk.download('stopwords')
stopwords_list = set(stopwords.words('english'))


path = os.path.abspath('')
path_movies = path.replace('model', 'data\\movies_metadata.csv')
path_ratings = path.replace('model', 'data\\links.csv')
path_users = path.replace('model', 'data\\ratings.csv')

movies = pd.read_csv(path_movies)
ratings = pd.read_csv(path_ratings)
users = pd.read_csv(path_users)

#Getting rid of NaN(missing) values
movies.dropna(inplace=True)
ratings.dropna(inplace=True)
users.dropna(inplace=True)

print(f"All columns in movies dataset: \n {list(movies.columns)}")

print(f"All columns in ratings dataset:\n {list(ratings.columns)}")

print(f"All columns in users dataset: \n {list(users.columns)}")


print(f"The first ten rows in the movie dataset: \n {movies.head()}")

print(f"The first ten rows in the ratings dataset: \n {ratings.head()}")

print(f"The first ten rows in the users dataset: \n {users.head()}")


def processing_text_data(data, lemmanization=True, stopwords=True, lowercasing=True, punctuation=True):
    '''

    :param data: the data we want to process
    :param lemmanization: boiling down the word to its root
    :param stopwords: words that are not that important
    :param lowercasing: changing the size of the letter
    :param punctuation: the choice whether the punctuation is needed
    :return: processed data
    '''

    tokenized_data = []

    for row in range(0, len(data)):
        tokenized_row = word_tokenize(data.iloc[row])
        if punctuation:
            tokenized_row = [word for word in tokenized_row if word not in string.punctuation]
        if lowercasing:
            tokenized_row = [str.lower(word) for word in tokenized_row]
        if stopwords:
            tokenized_row = [word for word in tokenized_row if word not in stopwords_list]
        if lemmanization:
            lem = WordNetLemmatizer()
            tokenized_row = [lem.lemmatize(word) for word in tokenized_row]

        tokenized_data.append(tokenized_row)

    return tokenized_data

def normalization(data):
    '''
    :param data: the data we want to normalize
    :return: normalized data
    '''

    norm_data = deepcopy(data)

    max_value = max(data)
    min_value = min(data)

    for x in range(len(data)):
        norm_data.iloc[x] = (data.iloc[x] - min_value) / (max_value - min_value)

    return norm_data.values.tolist()

def process_date(dates):
    '''
    :param dates: the date we want to process
    :return: DataFrame object containing year, month, day
    '''

    date_row = pd.DataFrame(columns=['year', 'month', 'day'])

    for date in dates:
        row = len(date_row)
        year, month, day = date.split('-')
        date_row.loc[row, 'year'] = year
        date_row.loc[row, 'month'] = month
        date_row.loc[row, 'day'] = day

    return date_row

#preparing text data

movies_desc = processing_text_data(movies['overview'])
movies_title = processing_text_data(movies['title'])
movies_org_language = processing_text_data(movies['original_language'])

#preapring data containing dates

movies_release_date = process_date(movies['release_date'])
movies_release_year = movies_release_date['year'].values.tolist()

#preparing numerical data using normalisation

movies_runtime = normalization(movies['runtime'])
movies_vote_avg = normalization(movies['vote_average'])

print(f'\n How the data looks like after processing: \n ')
print(f'Description: {movies_desc[0]}')
print(f'Title: {movies_title[0]}')
print(f'Original langauge: {movies_org_language[0]}')
print(f'Release year: {movies_release_year[0]}')
print(f'Runtime: {movies_runtime[0]}')
print(f'Vote average: {movies_vote_avg[0]}')

#combinig all movies' processed features into one dataset

movie_dataset = pd.DataFrame({
    'description': movies_desc,
    'title': movies_title,
    'org_language': movies_org_language,
    'release_date': movies_release_year,
    'runtime': movies_runtime,
    'vote_average': movies_vote_avg,
})

#The first ten rows of the movie dataset

print(movie_dataset.head())

#creating custom DataSet

class MoviesDataSet():
    def __init__(self, movies, users, ratings, transform=True):
        self.movies = movies
        self.users = users
        self.ratings = ratings
        self.transform = transform

    def __len__(self):
        return len(self.movies)

    def __getitem__(self, idx):

        movie_item = self.movies.iloc[idx]
        user_item = self.users.iloc[idx]
        rating = self.ratings.iloc[idx]

        return movie_item, user_item, rating


