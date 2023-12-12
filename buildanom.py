import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import datetime
import pickle

st.set_page_config(page_title="Build an Oscar Nominated Film")

st.sidebar.markdown(" ## About")
st.sidebar.markdown("This prediction model allows you to build a movie based on various parameters. Once built, click on the button on the buttom of the page to find the probability that your film would be nominated for an Oscar for best film.")


st.sidebar.markdown(" ## Resources")
st.sidebar.markdown(
    """
- [Streamlit Documentation](https://docs.streamlit.io/)
- [IMDB](https://help.imdb.com/article/contribution/other-submission-guides/awards/G5KGRJURZFQGHJQH#)
- [OMDB](https://www.omdbapi.com/)
- [BYU Professor Dr. Tass](https://statistics.byu.edu/directory/tass-shannon)
""")

st.sidebar.markdown(" ## Blog Posts")
st.sidebar.markdown(
    """
- [Data Scrapping](https://carsonp4.github.io/2023/11/16/movie-scrape.html)
- [EDA With This Dataset](https://carsonp4.github.io/2023/12/07/movie-eda.html)
- [Building Movie Recommendation Model](https://carsonp4.github.io/2023/10/05/movie-ml.html)
""")

st.sidebar.markdown(" ## Info")
st.sidebar.info("Read more about how the model works and see the code on my [Github](https://github.com/carsonp4/Final-Project).", icon="ℹ️")




st.title("Can you build an Oscar Nominated Film?")

### Loading Data
#url = "https://raw.githubusercontent.com/carsonp4/Final-Project/main/streamlitdash.csv"

#df = pd.read_csv(url, index_col=0)

### Making logistic Regression
# Assuming the first column is the target variable, and the rest are features
#X = df.iloc[:, 1:]  # Features (excluding the first column)
#y = df.iloc[:, 0]   # Target variable (the first column)

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
#model = LogisticRegression()

# Train the model on the training data
#model.fit(X_train, y_train)

model = pickle.load(open('https://raw.githubusercontent.com/carsonp4/Final-Project/main/streamlitmodel.pkl', 'rb'))


### Make your own movie
#movie = pd.DataFrame(0, index=[0], columns=df.columns).iloc[:, 1:]
movie = pd.read_csv("https://raw.githubusercontent.com/carsonp4/Final-Project/main/streamlitblank.csv", index_col=[0])


movie["Runtime"] = st.slider('What is the Movie Runtime in Minutes?', 0, 300, 90)

movie["IMDB"] = st.slider('What is the IMDB Rating?', 0, 10, 7)
movie["Rotten_Tomatoes"] = st.slider('What is the Rotten Tomatoes Score?', 0, 100, 80)
movie["Metascore"] = st.slider('What is the Meta critic score?', 0, 100, 70)

movie['IMDB_Votes'] = st.slider('How Many People Watched Your Film On IMDB?', 0, 3000000, 100000)

movie["Boxoffice"] = st.slider('How Much Money Did the Film Make at the Boxoffice?', 0, 1000000000, 500000000)

date = st.date_input("When Was The Film Released? (Keep 2000-2024)", datetime.date(2023, 3, 16))
movie["Release_Month"] = date.month
movie["Release_Year"] = date.year
movie["Release_DOY"] = date.timetuple().tm_yday

rating_columns = [col for col in df.columns if col.startswith("Rating_")]
ratings = [rating.split('_')[1] if '_' in rating else rating for rating in rating_columns]
rating_sums = [df[col].sum() for col in rating_columns]
rating_sum_tuples = list(zip(ratings, rating_sums))
sorted_ratings = sorted(rating_sum_tuples, key=lambda x: x[1], reverse=True)
sorted_ratings = [rating for rating, _ in sorted_ratings]
selected_rating = st.selectbox('What is The Film Rated?', sorted_ratings)
movie["Rating_" + selected_rating] = 1

Genre_columns = [col for col in df.columns if col.startswith("Genre_")]
Genres = [Genre.split('_')[1] if '_' in Genre else Genre for Genre in Genre_columns]
Genre_sums = [df[col].sum() for col in Genre_columns]
Genre_sum_tuples = list(zip(Genres, Genre_sums))
sorted_Genres = sorted(Genre_sum_tuples, key=lambda x: x[1], reverse=True)
sorted_Genres = [Genre for Genre, _ in sorted_Genres]
selected_Genres = st.multiselect("What Genre(s) is Your Film?", sorted_Genres, ["Drama", "Action"])
for i in range(len(selected_Genres)):
    movie["Genre_" + selected_Genres[i]] = 1

Director_columns = [col for col in df.columns if col.startswith("Director_")]
Directors = [Director.split('_')[1] if '_' in Director else Director for Director in Director_columns]
selected_Directors = st.multiselect("Who Directed Your Film?", Directors)
for i in range(len(selected_Directors)):
    movie["Director_" + selected_Directors[i]] = 1

Writer_columns = [col for col in df.columns if col.startswith("Writer_")]
Writers = [Writer.split('_')[1] if '_' in Writer else Writer for Writer in Writer_columns]
selected_Writers = st.multiselect("Who Wrote Your Film?", Writers)
for i in range(len(selected_Writers)):
    movie["Writer_" + selected_Writers[i]] = 1

Language_columns = [col for col in df.columns if col.startswith("Language_")]
Languages = [Language.split('_')[1] if '_' in Language else Language for Language in Language_columns]
selected_Languages = st.multiselect("What Language(s) is Your Film In?", Languages, ["English"])
for i in range(len(selected_Languages)):
    movie["Language_" + selected_Languages[i]] = 1

Country_columns = [col for col in df.columns if col.startswith("Country_")]
Countrys = [Country.split('_')[1] if '_' in Country else Country for Country in Country_columns]
selected_Countrys = st.multiselect("What Countries is Your Film From?", Countrys, ["United States"])
for i in range(len(selected_Countrys)):
    movie["Country_" + selected_Countrys[i]] = 1

movie["Total_Noms"] = st.slider('How Many Total Nominations Did The Film Receive Across All Film Awards?', 0, 400, 30)

movie["Total_Wins"] = st.slider('How Many Total Award Wins Did The Film Receive Across All Film Awards?', 0, 400, 15)

Award_columns = [col for col in movie.columns if col.startswith("(")]
selected_Awards = st.multiselect("Select Specific Awards Your Film Was Nominated For and/or Won?", Award_columns)
for i in range(len(selected_Awards)):
    movie[selected_Awards[i]] = 1


st.dataframe(movie)



### Show Probability
if st.button('Find Out Movie Probability Of Being Oscar Best Film Nominated'):
    nom_prob = model.predict_proba(movie)[:, 1][0]
    st.title("Probability of being nominated:")
    st.title(nom_prob)
