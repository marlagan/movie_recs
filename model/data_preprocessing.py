import nltk
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer

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

def processing_text_data(data, lemmanization=True, stopwords=True, lowercasing=True, punctuaction=True):
    tokenized_data = []

    for row in range(1, len(data)):
        tokenized_row = word_tokenize(data.iloc[row])
        if punctuaction:
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

movies_desc = processing_text_data(movies['overview'])
print(movies_desc[0])

#moies_desc = processing_text_data(movies['de'], lowercasing=True)






#creating custom DataSet

'''
class MoviesDataSet():
    def __init__(self, movies, users, ratings):
'''