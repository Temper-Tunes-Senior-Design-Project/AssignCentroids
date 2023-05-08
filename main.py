from flask import jsonify, Flask
from flask_cors import CORS, cross_origin
import json
import scipy.stats as stats
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

from sklearn.discriminant_analysis import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'POST')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

# Entry point parameters: 
@app.route('/assignCentroids')
@cross_origin()
def assignCentroids(request): #songs, user_id
    request_json = request.get_json(silent=True)
    if request_json and all(k in request_json for k in ("songs","user_id")):
        song_list = request_json["songs"]
        user_id = request_json["user_id"]
    else:
        return (jsonify({"error":"Bad Input, must pass 'songs' list, and 'user_id'"}), 400)

    #initialize creds
    global sp
    if sp == None:
        spotify_client()
        firestoreConnection()

    #Get known song_ids
    #Split ids by whether they are already labelled or not
    songs = [song.strip() for song in song_list if len(song.strip()) > 0]
    known_track_moods_dict = getAlreadyLabelled(songs)
    new_song_ids = [song_id for song_id in songs if song_id not in known_track_moods_dict.keys()]
    # Get song features of the new ids
    features_df = retrieveTrackFeatures(new_song_ids)
    track_data = {}
    if features_df is not None:
        processed_features_df = clipAndNormalizeMLP(features_df)
        if processed_features_df.shape[0] == 0 and len(known_track_moods_dict.keys()) == 0:
            return jsonify({"error": "Issue with spotify server"})
        #assign unknown songs
        (predictions, track_data) = assignLabels(new_song_ids)
        #upload unknown song predictions to DB
        addTrackMoodToDB(predictions)
        #combine arrays
        all_songs_labels = {**known_track_moods_dict, **predictions}
    else:
        all_songs_labels = known_track_moods_dict
    #retrieve track features for known track moods
    known_features_df = retrieveTrackFeatures(list(known_track_moods_dict.keys()))
    known_track_data = {}
    if known_features_df is not None: 
        processed_known_features_df = clipAndNormalizeMLP(known_features_df)
        known_track_data = processed_known_features_df.to_dict(orient='index')
        for key, value in known_track_data.items():
            known_track_data[key] = list(value.values())
    combined_track_data = {**known_track_data, **track_data}

    #upload user songs to DB
    uploadUserSongList(user_id, list(all_songs_labels.keys()))
    #calculate centroids
    centroids = classifyCentroids(all_songs_labels, combined_track_data)
    #upload centroids to user's DB
    uploadCentroidsToDB(user_id, centroids)
    #return 200 status 
    return (jsonify({"result": "success"}), 200)

#Predicting labels of songs for each model
def assignLabels(song_ids):
    global sp
    if sp == None:
        spotify_client()
    
    predictions = {}
    track_data = {}
    # Use Spotipy to retrieve track information
    features_df = retrieveTrackFeatures(song_ids)
    if features_df is None: return ({}, {})
    processed_features_df = clipAndNormalizeMLP(features_df)
    if processed_features_df.shape[0] == 0:
        return ({"error": "None of the songs passed were found. Either Spotify is down, or the song ids are incorrect"},)
    pred, _ = getMoodLabelMLP(processed_features_df)
    for i, (key, row) in enumerate(processed_features_df.iterrows()):
        predictions[key]=pred[i]
        data = row.values.reshape(1,-1)
        track_data[key] = list(data[0])

    return (predictions, track_data)

#______________________________________________
# Initialization
#______________________________________________
#Setup Spotify and Firebase Credentials
sp = None
def spotify_client():
    global sp
    sp_cred = None
    with open('spotify_credentials.json') as credentials:
        sp_cred = json.load(credentials)
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(sp_cred["client_id"],sp_cred['client_secret']))


cred,db = None,None
def firestoreConnection():
    global cred
    global db
    cred = credentials.Certificate("mood-swing-6c9d0-firebase-adminsdk-9cm02-66f39cc0dd.json")
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    
MLP_model = None
def load_mlp_model():
    global MLP_model
    with open('MLP3.pkl','rb') as f:
        MLP_model = pickle.load(f)

#______________________________________________
# Classify Centroids
#______________________________________________
moods = ['sad','angry','energetic','excited','happy','content','calm','depressed'] #Represents DB indexing of moods
centroid_features_MLP = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
                        'acousticness','instrumentalness', 'liveness', 'valence', 'tempo', 
                        'duration_ms', 'time_signature']
#Assign centroids for each model
def classifyCentroids(predicted_labels, track_data):    
    centroids = {mood:{} for mood in moods}

    for mood_index in range(len(moods)):
        mood_song_scores = {song_id:track_data[song_id] for song_id, value in predicted_labels.items() 
                            if value == mood_index}
        mood = moods[mood_index]
        fields = centroid_features_MLP
        centroid = calculateCentroid(mood_song_scores, fields)
        if centroid is not None:
            centroids[mood] = centroid 
        else: #else default value for centroid??? (currently just 0.0)
            centroids[mood] = {feature:value for feature,value in zip(fields, [0 for _ in range(len(fields))])}
    return centroids

#Calculate a single centroid
def calculateCentroid(songs, fields): #assuming songs is stored as {id: scores}
    num_songs = len(songs)
    if num_songs == 0: return None
    #get scores for the mood
    num_fields = len(fields)
    weight = 1/num_songs
    cumulative_score = [0 for _ in range(num_fields)]
    for scores in songs.values():
        cumulative_score = np.add(scores, cumulative_score)
    final_scores = np.multiply(weight, cumulative_score)
    centroid = {feature:value for feature,value in zip(fields, final_scores)}
    return centroid

#______________________________________________
# Database Operations
#______________________________________________
            
def getTrackMoodFromDB(track_id):
    doc_ref = db.collection('songs').document(track_id)
    doc_data = doc_ref.get()
    if doc_data.exists:
        return doc_data.to_dict().get('mood')
    else:
        return None

def getAlreadyLabelled(track_ids):
    already_labelled = {}
    for track_id in track_ids:
        mood = getTrackMoodFromDB(track_id)
        if mood is not None:
            already_labelled[track_id] = mood
    return already_labelled

def addTrackMoodToDB(tracks_dict):
    for track_id, mood in tracks_dict.items():
        doc_ref = db.collection('songs').document(track_id)
        doc_ref.set({
            'mood': int(mood)
        })

def uploadUserSongList(user_id, songs):
    # Define the user document reference
    user_doc_ref = db.collection('users').document(user_id)
    # Update the classifiedSongs field with the new list of songs
    user_doc_ref.update({'classifiedSongs': songs})
    

def uploadCentroidsToDB(user_id, centroids):
    # Loop through each mood in the centroids dictionary
    for mood, scores in centroids.items():
        # Define the document reference for the current mood
        mood_doc_ref = db.collection('users').document(user_id).collection('centroids').document(mood)
        # Upload the scores to the document
        mood_doc_ref.set(scores)
    
#______________________________________________
# MLP Model Classifcation
#______________________________________________

def getMoodLabelMLP(songFeatures):
    if MLP_model is None:
        load_mlp_model()
    prediction = MLP_model.predict(songFeatures.values)
    pred_probability= MLP_model.predict_proba(songFeatures.values)
    return prediction, pred_probability

def retrieveTrackFeatures(track_ids):
    dfs = []
    for i in range(0, len(track_ids), 50):
        # Retrieve track features with current offset
        features = sp.audio_features(track_ids[i:i+50])
        checked_features = [l for l in features if l is not None]
        # Convert to DataFrame
        if len(checked_features) > 0:
            df = pd.DataFrame(checked_features)
            # Remove columns that we don't need
            df = df.drop(['type', 'uri', 'analysis_url', 'track_href'], axis=1)

            # Append to list of dataframes
            dfs.append(df)
    if len(dfs) == 0: return None
    # Concatenate all dataframes into a single one
    features_df = pd.concat(dfs, ignore_index=True)
    features_df.set_index("id", inplace=True)
    #convert to dictionary, with track id as key
#     features_dict = features_df.set_index('id').T.to_dict('list')
    return features_df

def clipAndNormalizeMLP(features):
    #clip the features to the range of the training data
    features['danceability'] = features['danceability'].clip(lower=0.25336000000000003, upper=0.9188199999999997)
    features['energy'] = features['energy'].clip(lower=0.047536, upper=0.982)
    features['loudness'] = features['loudness'].clip(lower=-24.65708, upper=-0.8038200000000288)
    features['speechiness'] = features['speechiness'].clip(lower=0.0263, upper=0.5018199999999997)
    features['acousticness'] = features['acousticness'].clip(lower=1.4072e-04, upper=0.986)
    features['instrumentalness'] = features['instrumentalness'].clip(lower=0.0, upper=0.951)
    features['liveness'] = features['liveness'].clip(lower=0.044836, upper=0.7224599999999991)
    features['valence'] = features['valence'].clip(lower=0.038318, upper=0.9348199999999998)
    features['tempo'] = features['tempo'].clip(lower=66.34576, upper=189.87784)
    features['duration_ms'] = features['duration_ms'].clip(lower=86120.0, upper=341848.79999999976)
    features['time_signature'] = features['time_signature'].clip(lower=3.0, upper=5.0)
    
    columns_to_log=['liveness', 'instrumentalness', 'acousticness', 'speechiness','loudness','energy']

    for i in columns_to_log:
        if i == 'loudness':
            features[i] = features[i] + 60
        features[i] = np.log(features[i]+1)

    #normalize the data
    scaler = pickle.load(open('scaler3.pkl', 'rb'))
    #fit on all columns except the track id
    preprocessedFeatures = scaler.transform(features)

    #convert to dictionary, with track id as key
    preprocessedFeatures = pd.DataFrame(preprocessedFeatures, columns=features.columns)

    
    #apply z-score normalization
    for i in columns_to_log:
        preprocessedFeatures[i] = stats.zscore(preprocessedFeatures[i])
        preprocessedFeatures.clip(lower=-2.7, upper=2.7, inplace=True)

    preprocessedFeatures['id'] = features.index.to_list()
    preprocessedFeatures.set_index('id', inplace=True)

#     preprocessedFeatures = preprocessedFeatures.set_index('id').T.to_dict('list')
    return preprocessedFeatures




#__________________________________________
# Entry
#__________________________________________

if __name__ == '__main__':
    app = Flask(__name__)
    app.route('/closestSongs', methods=['POST'])(lambda request: assignCentroids(request))
    app.run(debug=True)