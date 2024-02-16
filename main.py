import os

from flask import Flask, render_template, request, redirect, url_for, jsonify
from boxoffice_linear import create_dash_application
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Create app
app = Flask(__name__)

# Static pages ------------------------------------

# Main page
@app.route('/', endpoint='index')
def index():
    return render_template('index.html')

# Presentation/Communication
@app.route('/teaching', endpoint='teaching')
def teaching():
    return render_template('teaching.html')

# Research
@app.route('/academic', endpoint='academic')
def teaching():
    return render_template('academic.html')

# Resume
@app.route('/resume_interactive', endpoint='resume_interactive')
def image_website():
    return render_template('interactive_resume.html')

# Resume
@app.route('/resume', endpoint='resume')
def image_website():
    return render_template('resume.html')

# Image Credit
@app.route('/credit', endpoint='credit')
def image_credit():
    return render_template('image_credit.html')

# Notebooks -----------------------------------------------------------

# Forecast - Boxoffice
@app.route('/boxoffice_notebook', endpoint='notebook_boxoffice')
def boxoffice():
    return render_template('boxoffice_notebook.html')

# Image Classification
@app.route('/dogvcat_notebook', endpoint='dogvcat_notebook')
def dogvcat():
    return render_template('dogvcat_notebook.html')

# Movie Recommendation
@app.route('/movies_notebook', endpoint='movies_notebook')
def dogvcat():
    return render_template('movies_notebook.html')

# Demos ---------------------------------------------------------------

# Forecast - Box Office ------------
@app.route('/boxoffice_dashboard', endpoint='boxoffice_dashboard')
def boxoffice():
    return render_template('boxoffice_demo.html')

# Movie Recommendation----------------
# Content-based Filtering-----------
# Get the data in a pandas dataframe
file_path = 'static/csv_files/movies_data_imdb1000.csv'
df_movies = pd.read_csv(file_path)

# Set the features
features = ['Overview', 'Director', 'Genre', 'Star1', 'Star2', 'Star3', 'Star4']

# Define weights for each feature
weights = {'Overview': 1, 'Director': 0.8, 'Genre': 0.5, 'Star1': 0.1, 'Star2': 0.1, 'Star3': 0.1, 'Star4': 0.1}

# Fill NaN values with an empty string
df_movies[features] = df_movies[features].fillna('')

# Combine text features into a single column
df_movies['Combined_Text'] = df_movies[features].apply(lambda row: ' '.join(row), axis=1)

# Fit the TfidfVectorizer to the combined text
tfidf_vectorizer = TfidfVectorizer()
combined_tfidf_matrix = tfidf_vectorizer.fit_transform(df_movies['Combined_Text'])

# Function to transform individual features based on the fitted vectorizer and apply weights
def transform_feature(feature):
    feature_tfidf_matrix = tfidf_vectorizer.transform(df_movies[feature])
    return feature_tfidf_matrix * weights[feature]

# Transform individual features
tfidf_matrices = {feature: transform_feature(feature) for feature in features}

# Combine individual feature matrices
combined_tfidf_matrix = sum(tfidf_matrices.values())

# Calculate cosine similarity
cosine_sim = cosine_similarity(combined_tfidf_matrix, combined_tfidf_matrix)

# Function for movie recommendations
def get_recommendations_content(input_movies):
    input_indices = []
    for movie_title in input_movies:
        idx = df_movies[df_movies['Series_Title'] == movie_title].index
        if not idx.empty:
            input_indices.append(idx[0])

    if not input_indices:
        return "No matches found for input movies."

    sim_scores = cosine_sim[input_indices].sum(axis=0)

    # Exclude input movies from recommendations
    exclude_indices = np.array(input_indices)
    sim_scores[exclude_indices] = -1

    # Get top recommendations
    movie_indices = sim_scores.argsort()[::-1][:5]
    return df_movies['Series_Title'].iloc[movie_indices]

# Suggestions for the text input
@app.route('/get_suggestions-content', methods=['GET'], endpoint='suggestions_content')
def get_suggestions():
    # Fetch suggestions from your pandas dataframe or any other data source
    suggestions = df_movies['Series_Title'].values.tolist()
    return jsonify(suggestions)


# Content based recommendations are generated
@app.route('/movie_recommendation_content', methods=['GET', 'POST'], endpoint='rec_content')
def movie_recommendation():
    recommendations = []  # Initialize an empty list for recommendations
    selected_movies = []  # Initialize an empty list for selected movies
    page = request.args.get('page', 'content')  # Get the 'page' parameter from the URL, default to 'content'

    if request.method == 'POST':
        # Get selected movies from the form
        selected_movies = [request.form['selectedMovie1'], request.form['selectedMovie2'], request.form['selectedMovie3']]
        print(selected_movies)

        # Get recommendations
        recommendations = get_recommendations_content(selected_movies)

    # Render the template with the form to select movies and recommendations
    return render_template('movies_demo.html', movies=df_movies['Series_Title'].tolist(), suggestions=df_movies['Series_Title'].tolist(),
                           recommendations_content=recommendations, selected_movies_content=selected_movies, page=page)


## Collaborative-Filtering ----------------------------------------------------------
# Get the data in a pandas dataframe
file_path = 'static/csv_files/collab_item_similarity.csv'
item_similarity_df = pd.read_csv(file_path)

def get_movie_recommendations_collaborative(user_movies, user_ratings = [1,1,1]):
    # Create an empty DataFrame to store the recommendations
    movies = item_similarity_df.columns.tolist()
    movies.pop(0)
    recommendations = pd.DataFrame({'Movie': movies,'Score': [0 for k in range(len(item_similarity_df.index))]})

    # Loop through each movie in the user's list
    movie_count = 0
    for movie in user_movies:
        # Get the similarity scores between the current movie and all other movies
        similarity_scores = item_similarity_df[movie]
        
        # Multiply the similarity scores by the user's ratings
        weighted_scores = similarity_scores * user_ratings[movie_count]
        movie_count += 1
        # Add the weighted scores to the recommendations DataFrame
        recommendations['Score'] += weighted_scores.values

    # Filter out prefered movies
    recommendations = recommendations[~recommendations.Movie.isin(user_movies)]
    
    # Sort the recommendations by score in descending order
    recommendations = recommendations.sort_values(by='Score', ascending=False)[:5]


    return recommendations.Movie.tolist()

# Suggestions for the text input
@app.route('/get_suggestions-collaborative', methods=['GET'], endpoint='suggestions_collaborative')
def get_suggestions():
    # Fetch suggestions from your pandas dataframe or any other data source
    suggestions = item_similarity_df.columns.tolist()
    return jsonify(suggestions)

# Collaborative-filtering recommendations are generated
@app.route('/movie_recommendation_collaborative', methods=['GET', 'POST'],endpoint='rec_collaborative')
def movie_recommendation():
    recommendations = []  # Initialize an empty list for recommendations
    selected_movies = []  # Initialize an empty list for selected movies
    page = request.args.get('page', 'collaborative')

    if request.method == 'POST':
        # Get selected movies from the form
        selected_movies = [request.form['selectedMovie4'], request.form['selectedMovie5'], request.form['selectedMovie6']]
        print(selected_movies)

        # Get recommendations
        recommendations = get_movie_recommendations_collaborative(selected_movies)

    # Render the template with the form to select movies and recommendations
    return render_template('movies_demo.html', movies=df_movies['Series_Title'].tolist(), suggestions=df_movies['Series_Title'].tolist(), recommendations_collaborative=recommendations,selected_movies_collaborative=selected_movies,page='collaborative')


# Interactive Resume - Illustrations ------------------------------------

# Forecast - Box Office
@app.route('/resume_boxoffice', endpoint='image_box')
def image_boxoffice():
    return render_template('image_box.html')

# Image Classification - Cats and Dogs
@app.route('/resume_dog', endpoint='image_dog')
def image_dog():
    return render_template('image_dog.html')

# Recommendation System - Movies
@app.route('/resume_movie', endpoint='image_movie')
def image_movie():
    return render_template('image_movie.html')

# Website - Flask/Dash app
@app.route('/resume_website', endpoint='image_website')
def image_website():
    return render_template('image_website.html')

# Main app -------------------------------------------------------------

if __name__ == "__main__":
    # Create Dashboard
    dash_app = create_dash_application(app)
    # Create Flask App
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))