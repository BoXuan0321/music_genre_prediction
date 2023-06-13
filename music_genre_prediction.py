import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Title
st.write("""# Prediction of Music Genre""")



st.write("This app  can predicts the **Music Genre** of a song based on the *popularity*, *acousticness*, *danceability*, *instrumentalness*, *loudness*, *speechiness* of the song.")
st.write("popularity: A measure of the popularity of the track, represented as a floating-point number between 0 and 100. Higher values indicate greater popularity.")
st.write("acousticness: A measure of the acousticness of the track, represented as a floating-point number between 0 and 1. Higher values indicate greater acousticness.")
st.write("danceability: A measure of the danceability of the track, represented as a floating-point number between 0 and 1. Higher values indicate greater danceability.")
st.write("instrumentalness: A measure of the instrumentalness of the track, represented as a floating-point number between 0 and 1. Higher values indicate greater instrumentalness.")
st.write("loudness: A measure of the loudness of the track, represented as a floating-point number in decibels (dB). Higher values indicate greater loudness. Range: -47.10 to 3.8.")
st.write("speechiness: A measure of the speechiness of the track, represented as a floating-point number between 0 and 1. Higher values indicate greater speechiness.")


st.sidebar.header('User Input')

def userInput():
    popularity = st.sidebar.slider('popularity', 0.00, 100.00, 50.00)
    acousticness= st.sidebar.slider('acousticness', 0.00, 1.00, 0.50)
    danceability = st.sidebar.slider('danceability', 0.00, 1.00, 0.50)
    instrumentalness = st.sidebar.slider('instrumentalness', 0.00, 1.00, 0.50)
    loudness = st.sidebar.slider('loudness', -47.10, 3.80,-21.65)
    speechiness = st.sidebar.slider('speechiness', 0.00, 1.00, 0.50)
    
    data = {'popularity': popularity,
            'acousticness': acousticness,
            'danceability': danceability,
            'instrumentalness': instrumentalness,
            'loudness': loudness,
            'speechiness': speechiness}
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = userInput()

#Read music genre file
music_genre_raw = pd.read_csv('music_genre.csv')

music_genre_raw.drop(music_genre_raw [music_genre_raw ['tempo'] == '?'].index , inplace = True)
music_genre = music_genre_raw.drop(['obtained_date', 'valence', 'mode','energy','instance_id','key','tempo','artist_name','liveness','duration_ms', 'music_genre', 'track_name'], axis=1)
music_genre_raw  = music_genre_raw.dropna()

#Combine user input with music genre dataset

data = pd.concat([input_df,music_genre],axis=0)

#Select First row only
data = data[:1]

#Load model
load_model = pickle.load(open('KNN_music_genre.pkl', 'rb'))

prediction = load_model.predict(data)

output = np.array(['Alternative', 'Anime', 'Blues', 'Classical', 'Country', 'Electronic', 'Hip-Hop', 'Jazz', 'Rap', 'Rock'])

#{'Alternative': 0, 'Anime': 1, 'Blues': 2, 'Classical': 3, 'Country': 4, 'Electronic': 5, 'Hip-Hop': 6, 'Jazz': 7, 'Rap': 8, 'Rock': 9}
st.subheader('Prediction')
st.write('The prediction is:' ,output[prediction][0])
