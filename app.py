import os
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_data(x):
    return str.lower(x.replace(" ", ""))

def create_soup(x):
    return x['title'] + ' ' + x['director'] + ' ' + x['cast'] + ' ' + x['listed_in'] + ' ' + x['description']

def get_recommendations(title, cosine_sim):
    title = title.replace(' ', '').lower()
    if title not in indices:
        return pd.DataFrame({'Error': ['Title not found in the dataset']})
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    result = netflix_overall['title'].iloc[movie_indices]
    result = result.to_frame()
    result = result.reset_index(drop=True)
    return result

# Load data
file_path = os.path.join(os.path.dirname(__file__), 'netflix_titles.csv')
netflix_overall = pd.read_csv(file_path)
netflix_data = netflix_overall.fillna('')

# Preprocess data
new_features = ['title', 'director', 'cast', 'listed_in', 'description']
netflix_data = netflix_data[new_features]
for feature in new_features:
    netflix_data[feature] = netflix_data[feature].apply(clean_data)
netflix_data['soup'] = netflix_data.apply(create_soup, axis=1)

# Compute cosine similarity
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(netflix_data['soup'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
netflix_data = netflix_data.reset_index()
indices = pd.Series(netflix_data.index, index=netflix_data['title'])

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about', methods=['POST'])
def getvalue():
    moviename = request.form['moviename']
    df = get_recommendations(moviename, cosine_sim2)
    return render_template('result.html', tables=[df.to_html(classes='data')], titles=df.columns.values)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
