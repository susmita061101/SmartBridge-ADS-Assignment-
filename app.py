import numpy as np
import pandas as pd
import streamlit as st
import re
import string
import warnings
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# Load the dataset
zomato_real = pd.read_csv("zomato.csv")

# Deleting Unnecessary Columns
zomato = zomato_real.drop(['url', 'dish_liked', 'phone'], axis=1)

# Dropping duplicates
zomato.drop_duplicates(inplace=True)

# Remove NaN values from the dataset
zomato.dropna(how='any', inplace=True)

# Changing column names
zomato = zomato.rename(columns={'approx_cost(for two people)': 'cost', 'listed_in(type)': 'type', 'listed_in(city)': 'city'})

# Transformations
zomato['cost'] = zomato['cost'].astype(str)
zomato['cost'] = zomato['cost'].apply(lambda x: x.replace(',', '.'))
zomato['cost'] = zomato['cost'].astype(float)

# Removing '/5' from Rates
zomato = zomato.loc[zomato.rate != 'NEW']
zomato = zomato.loc[zomato.rate != '-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x
zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')

# Adjusting column names
zomato.name = zomato.name.apply(lambda x: x.title())
zomato.online_order.replace(('Yes', 'No'), (True, False), inplace=True)
zomato.book_table.replace(('Yes', 'No'), (True, False), inplace=True)

# Computing Mean Rating
restaurants = list(zomato['name'].unique())
zomato['Mean Rating'] = 0

for i in range(len(restaurants)):
    zomato['Mean Rating'][zomato['name'] == restaurants[i]] = zomato['rate'][zomato['name'] == restaurants[i]].mean()

# Scaling Mean Rating
scaler = MinMaxScaler(feature_range=(1, 5))
zomato[['Mean Rating']] = scaler.fit_transform(zomato[['Mean Rating']]).round(2)

# Lowercasing and removing punctuation, stopwords, and URLs
PUNCT_TO_REMOVE = string.punctuation
STOPWORDS = set(stopwords.words('english'))

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def remove_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

zomato["reviews_list"] = zomato["reviews_list"].str.lower()
zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_punctuation(text))
zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_stopwords(text))
zomato["reviews_list"] = zomato["reviews_list"].apply(lambda text: remove_urls(text))

# Dropping unnecessary columns
zomato = zomato.drop(['address', 'rest_type', 'type', 'menu_item', 'votes'], axis=1)

# Randomly sampling 50% of the dataframe
df_percent = zomato.sample(frac=0.5)
df_percent.set_index('name', inplace=True)
indices = pd.Series(df_percent.index)

# Creating tf-idf matrix
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_percent['reviews_list'])

# Calculating cosine similarities
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

def recommend(name, cosine_similarities=cosine_similarities):
    # Create a list to store top restaurants
    recommend_restaurant = []
    
    # Find the index of the entered restaurant
    idx = indices[indices == name].index[0]
    
    # Find the restaurants with similar cosine similarity values and order them from highest to lowest
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    
    # Extract top 30 restaurant indexes with similar cosine similarity values
    top30_indexes = list(score_series.iloc[0:31].index)
    
    # Get the names of the top 30 restaurants
    for each in top30_indexes:
        recommend_restaurant.append(list(df_percent.index)[each])
    
    # Create a new dataset to show similar restaurants
    df_new = pd.DataFrame(columns=['cuisines', 'Mean Rating', 'cost'])
    
    # Create the top 30 similar restaurants with some of their columns
    for each in recommend_restaurant:
        df_new = df_new.append(pd.DataFrame(df_percent[['cuisines', 'Mean Rating', 'cost']][df_percent.index == each].sample()))
    
    # Drop the same named restaurants and sort only the top 10 by the highest rating
    df_new = df_new.drop_duplicates(subset=['cuisines', 'Mean Rating', 'cost'], keep=False)
    df_new = df_new.sort_values(by='Mean Rating', ascending=False).head(10)
    
    print('TOP %s RESTAURANTS SIMILAR TO %s WITH SIMILAR REVIEWS: ' % (str(len(df_new)), name))
    
    return df_new

recommend('Pai Vihar')

# -----deploy in Flask code -------
import flask
import pandas as pd

app = flask.Flask(__name__)

@app.route("/")
def index():
  return flask.render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend_form():
  restaurant_name = flask.request.form["restaurant_name"]
  recommended_restaurants = recommend(restaurant_name)

  # Return the recommendations to the user
  return flask.render_template("recommendations.html", recommendations=recommended_restaurants.to_html())


if __name__ == "__main__":
  app.run(debug=True)

