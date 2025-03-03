from flask import Flask, request, jsonify, render_template, redirect, url_for
from movie_recs.model.interaction_with_gui import get_content, get_full_titles, movie_recommendation

app = Flask(__name__,  template_folder='../frontend', static_folder='../frontend')

titles_suggestions =  get_full_titles()
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def autocomplete():

    data = request.json

    query = data.get('query', '')

    suggestions = [text for text in titles_suggestions if text.lower().startswith(query.lower())]
    return jsonify(suggestions)

@app.route('/', methods=['GET'])
def get_movie_title():

    title = request.args.get('title_choice')

    return redirect(url_for('get_movies', x = title))

@app.route('/get-movies/<x>', methods=['GET'])
def get_movies(x):

    descriptions, images, titles, date_of_release, genres = movie_recommendation(x)

    return render_template('index1.html', descriptions=descriptions, titles=titles, genres=genres, date_of_release=date_of_release)

if __name__ == '__main__':
    app.run(debug=True, port=5500)