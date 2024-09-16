from flask import Flask, request, render_template_string
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the dataset
url=https://drive.google.com/file/d/1cCkwiVv4mgfl20ntgY3n4yApcWqqZQe6/view?usp=sharing
movies_data = pd.read_csv(url, low_memory=False)

selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Convert movie titles to lowercase
movies_data['title'] = movies_data['title'].astype(str).str.lower()

# Combine the selected features into a single string
combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data['tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

# Convert the text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# Compute cosine similarity scores
similarity = cosine_similarity(feature_vectors)

# HTML Templates with CSS and Background Images from Unsplash
index_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Movie Recommendation System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-image: url('https://wallpapercave.com/wp/wp1945898.jpg');
            background-size: cover;
            color: white;
            text-align: center;
            padding: 50px;
        }
        .container {
            background-color: rgba(0, 0, 0, 0.8);
            padding: 20px;
            border-radius: 10px;
            max-width: 500px;
            margin: 0 auto;
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 20px;
        }
        input[type="text"], input[type="submit"] {
            padding: 10px;
            width: 80%;
            margin: 10px 0;
            border-radius: 5px;
            border: none;
            transition: box-shadow 0.3s ease, background-color 0.3s ease;
            box-shadow: 0 0 15px rgba(40, 167, 69, 0.9); /* Stronger glow */
        }
        input[type="submit"] {
            background-color: #28a745;
            color: white;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #218838;
            box-shadow: 0 0 25px rgba(40, 167, 69, 1); /* Enhanced glow on hover */
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Movie Recommendation System</h1>
        <form action="/recommend" method="post">
            <label for="movie_name">Enter your favourite movie name:</label><br>
            <input type="text" id="movie_name" name="movie_name" required><br>
            <input type="submit" value="Get Recommendations">
        </form>
    </div>
</body>
</html>
'''

results_html = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recommendations</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-image: url('https://wallpapercave.com/wp/wp1945898.jpg');
            background-size: cover;
            color: white;
            text-align: center;
            padding: 50px;
        }
        .container {
            background-color: rgba(0, 0, 0, 0.8);
            padding: 20px;
            border-radius: 10px;
            max-width: 600px;
            margin: 0 auto;
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 20px;
        }
        ul {
            list-style-type: none;
            padding: 0;
        }
        li {
            background-color: #17a2b8;
            padding: 10px;
            margin: 5px;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        }
        a {
            color: #28a745;
            text-decoration: none;
            font-size: 1.2em;
            border: 2px solid #28a745;
            padding: 10px;
            border-radius: 5px;
            display: inline-block;
            margin-top: 20px;
            box-shadow: 0 0 15px rgba(40, 167, 69, 0.9); /* Glowing effect */
            transition: background 0.3s ease, color 0.3s ease, box-shadow 0.3s ease;
        }
        a:hover {
            background-color: #28a745;
            color: #fff;
            box-shadow: 0 0 25px rgba(40, 167, 69, 1); /* Enhanced glow on hover */
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Recommendations for "{{ movie_name }}"</h1>
        {% if recommendations %}
            <ul>
                {% for movie in recommendations %}
                    <li>{{ movie }}</li>
                {% endfor %}
            </ul>
        {% else %}
            <p>No recommendations found.</p>
        {% endif %}
        <a href="/">Back to Home</a>
    </div>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(index_html)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name'].lower()
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles, n=1)

    if find_close_match:
        close_match = find_close_match[0]
        index_of_the_movie = movies_data[movies_data.title == close_match].index[0]
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        recommendations = []
        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = movies_data.iloc[index]['title']
            if len(recommendations) < 30:
                recommendations.append(title_from_index.title())
            else:
                break

        return render_template_string(results_html, movie_name=movie_name.title(), recommendations=recommendations)
    else:
        return render_template_string(results_html, movie_name=movie_name.title(), recommendations=["Sorry, no close match found."])

if __name__ == '__main__':
    app.run(debug=True)
