import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("Can you build an Oscar Nominated Film?")

### Loading Data
url = "https://raw.githubusercontent.com/carsonp4/Final-Project/main/maindf.csv"

df = pd.read_csv(url, index_col=0)

### Cleaning for Regression
# Add the columns that don't contain the word 'Oscar' to the list
df = df.drop('award_year', axis=1)
columns_to_keep = ["('Oscar', 'Film')_nominated"]
columns_to_keep.extend([col for col in df.columns if 'Oscar' not in col])
df = df[columns_to_keep]

df = df.fillna(df.mean())


### Making logistic Regression
# Assuming the first column is the target variable, and the rest are features
X = df.iloc[:, 1:]  # Features (excluding the first column)
y = df.iloc[:, 0]   # Target variable (the first column)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)


### Make your own movie
movie = pd.DataFrame(0, index=[0], columns=df.columns).iloc[:, 1:]


movie["Runtime"] = st.slider('Movie Runtime in Minutes', 0, 300, 90)
movie["IMDB"] = st.slider('IMDB Rating', 0, 10, 7)
movie["Rotten_Tomatoes"] = st.slider('Rotten Tomatoes Score', 0, 100, 80)
movie["Metascore"] = st.slider('Meta critic score', 0, 100, 70)
movie['IMDB_Votes'] = st.slider('Number of Watched On IMDB', 0, 3000000, 100000)





### Show Probability
if st.button('Find Out Movie Probability Of Being Oscar Best Film Nominated'):
    nom_prob = model.predict_proba(movie)[:, 1][0]
    st.title("Probability of being nominated:")
    st.title(nom_prob)
