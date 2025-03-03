import movie_recs.model.data_preprocessing as dp
from copy import deepcopy
import numpy as np


full_dataset = dp.movies

def find_movie_index(title):
    '''
    Finding the index for the movie title
    :param title: movie title
    :return: index of the movie title in the dataset
    '''

    movies = dp.movies['title']

    for idx, value in enumerate(movies):
        if movies.iloc[idx] == title:
            return idx

    return None

def euclidian_distance(word1, word2):
    '''
    Calculating an Euclidian distance between two movies transformed into a bag of words
    :param word1: the first bag of word
    :param word2: the second bag of word
    :return: the Euclidian distance
    '''
    sum_col = 0

    for col in range(len(word1)):

        sum_col += (word1[col] - word2[col]) ** 2

    return np.sqrt(sum_col)

def similarity(movie, list_of_movies):
    '''
    Collecting distances between movies
    :param movie: the movie user has chosen
    :param list_of_movies: all the movies from the dataset
    :return: distances between chosen movie and movies from the dataset
    '''

    distances = []

    for row in range(len(list_of_movies)):
        distance = euclidian_distance(movie, list_of_movies[row])
        distances.append(distance)

    return distances

def argmin_n(amount, distances):
    '''
    Returning indexes of n smallest values in distances list
    :param amount: number of values to return
    :param distances: the distances list
    :return: indexes of n smallest values in distances list
    '''
    list_copy = deepcopy(distances)
    list_copy = np.array(list_copy, dtype=object)

    indexes = []

    for number in range(amount + 1):

        index = np.argmin(list_copy)
        if number != 0:
            indexes.append(index)
        list_copy[index] = 10000

    return indexes


def find_most_similar(movie, list_of_movies, amount):
    '''
    Choosing n most similar movies from the dataset
    :param movie: the movie user has chosen
    :param list_of_movies: list of the movies from the dataset
    :param amount: amount of movies to return
    :return: most similar movies from the dataset
    '''

    distances = similarity(movie, list_of_movies)

    indexes = argmin_n(amount, distances)

    return indexes

def get_content(indexes):
    '''
    Returning information about similar movies
    :param indexes: indexes of n most similar movies
    :return: movie's description
    '''

    descriptions = []
    images = []
    titles = []
    date_of_release = []
    genres = []

    for index in indexes:

        descriptions.append(full_dataset.iloc[index]['overview'])
        titles.append(full_dataset.iloc[index]['title'])
        date_of_release.append(full_dataset.iloc[index]['release_date'])
        genres.append(full_dataset.iloc[index]['genres'])

    return descriptions, images, titles, date_of_release, genres


def movie_recommendation(title):
    '''
    Finding 5 most similar movies from the dataset
    :param title: the title of the movie user has chosen
    :return: 5 most similar movies from the dataset
    '''

    index = find_movie_index(title)
    dataset = dp.movies_final
    movie = dataset[index]
    amount = 5
    indexes = find_most_similar(movie, dataset, amount)
    content = get_content(indexes)

    return content

def get_full_titles():
    '''
    Getting the full titles from the dataset
    :return: movie titles
    '''
    return dp.get_titles()


