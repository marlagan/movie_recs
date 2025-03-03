We all once had a moment when we enjoyed a particular movie so much 
that we wanted to find something similar to watch. That's why I decided to bring to life the movie recommendation system, which allows a user to discover movies that could pique their interest. Additionally, I developed a user-friendly interface using HTML, CSS, and JavaScript, facilitating the process of finding movies that match the user's interest and making it a better experience. Flask-based API backend written in Python is responsible for creating the connection between the Frontend and the recommendation system. The technique that I implemented for the recommendation system is content-based filtering. This approach concentrates on movies' features and uses them to predict the right movies.
In order to do that, I had to preprocess the data using Natural Language Processing(NLP) and create the bag of words from processed tokens. Such prepared data can be used to calculate a Euclidean distance between movies and find the most similar ones.

The dataset that I used for my project:
https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=links_small.csv
