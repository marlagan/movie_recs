from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from copy import deepcopy

import pandas as pd
import numpy as np
import string
import os
import re

#content-based recommendation system

#for tokenization
#nltk.download('punkt_tab')

#for lemmanization
#nltk.download('wordnet')

#downloading stopwords
#nltk.download('stopwords')
stopwords_list = set(stopwords.words('english'))


path = os.path.abspath('')
path = os.path.dirname(path)


path_movies = os.path.join(path, r'data\movies_metadata.csv')

movies = pd.read_csv(path_movies)

#Getting rid of NaN(missing) values
movies.dropna(inplace=True)


print(f"All columns in movies dataset: \n {list(movies.columns)}")

print(f"The first ten rows in the movie dataset: \n {movies.head()}")


def processing_text_data(data, tokenization=True, lemmanization=True, stopwords=True, lowercasing=True, punctuation=True):
    '''
    Processing text data
    :param data: the data we want to process
    :param tokenization: whether we want to tokenize the data
    :param lemmanization: boiling down the word to its root
    :param stopwords: words that are not that important
    :param lowercasing: changing the size of the letter
    :param punctuation: the choice whether the punctuation is needed
    :return: processed data
    '''

    tokenized_data = []

    for row in range(0, len(data)):

        tokenized_row = data.iloc[row]
        if tokenization:
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

#Preparing text data

movies_desc = processing_text_data(movies['overview'])
movies_title = processing_text_data(movies['title'])
movies_org_language = processing_text_data(movies['original_language'])
movies_genres = movies['genres']


pattern = r"'name':\s*'(\w+)'"

for x in range(len(movies_genres)):

    genres = re.findall(pattern, movies_genres.iloc[x])
    movies_genres.iloc[x] = genres

movies_genres = processing_text_data(movies_genres, tokenization=False)


print(f'\n How the data looks like after processing: \n ')
print(f'Description: {movies_desc[0]}')
print(f'Title: {movies_title[0]}')
print(f'Original langauge: {movies_org_language[0]}')
print(f'Genres: {movies_genres[0]}')

#Changing the categorical values into numerical values

def assigning_indices(dataset):
    '''
    Assigning indices to the data
    :param dataset: the data we want to turn into indices
    :return: dictionary, indices assigned to words
    '''
    dictionary = {}
    dataset_indices = deepcopy(dataset)
    for x in range(len(dataset)):

        if type(dataset[x]) != float:
            for y in range(len(dataset[x])):

                vocab_values = dictionary.values()
                if dictionary.get(dataset[x][y]):
                    dataset_indices[x][y] = dictionary.get(dataset[x][y])
                else:
                    if len(vocab_values) > 0:
                        dictionary[dataset[x][y]] = max(vocab_values) + 1
                        dataset_indices[x][y] = dictionary.get(dataset[x][y])
                    else:
                        dictionary[dataset[x][y]] = 1
                        dataset_indices[x][y] = dictionary.get(dataset[x][y])
        else:
            vocab_values = dictionary.values()
            if dictionary.get(dataset[x]):
                dataset_indices[x] = dictionary.get(dataset[x])
            else:
                if len(vocab_values) > 0:
                    dictionary[dataset[x]] = max(vocab_values) + 1
                    dataset_indices[x] = dictionary.get(dataset[x])
                else:
                    dictionary[dataset[x]] = 1
                    dataset_indices[x] = dictionary.get(dataset[x])

    return dictionary, dataset_indices

vocab_titles, movies_title = assigning_indices(movies_title)
vocab_genres, movies_genres = assigning_indices(movies_genres)
vocab_lang, movies_org_language= assigning_indices(movies_org_language)
vocab_desc, movies_desc = assigning_indices(movies_desc)

def create_bag_of_words(data, vocab):
    '''
    Creating a bag of words
    :param data: The data we want to transform into a bag of words
    :param vocab: All tokens that were created from the dataset
    :return: Bag of words
    '''

    bag_of_words = []

    for x in range(len(data)):
        row = []
        for key, value in vocab.items():
            row_value = 0

            for y in data[x]:
                if  value == y:
                    row_value += 1
            row.append(row_value)

        bag_of_words.append(row)
    return bag_of_words

bag_of_words_genres= create_bag_of_words(movies_genres, vocab_genres)
bag_of_words_desc = create_bag_of_words(movies_desc, vocab_desc)
bag_of_words_lang = create_bag_of_words(movies_org_language, vocab_lang)

movies_final = np.concatenate((bag_of_words_genres, bag_of_words_desc, bag_of_words_lang), axis=1)


def get_titles():
    '''
    Method returning all movie titles
    :return: movie titles
    '''
    return movies['title']


