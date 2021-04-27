"""Machine learning functions"""
import logging
import random
from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field, validator
import joblib
from tfidf import dtm
from tfidf import df1
from tfidf import spotify_songs
from app.data_model.find_songs import FindSongs
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm.notebook import tqdm
from pandas import Series
from app.data_model.find_songs import FindSongs
from pandas import Series
from app.data_model.find_songs import FindSongs
from os.path import dirname

log = logging.getLogger(__name__)
router = APIRouter()

DIR = dirname(__file__)
DATA_DIR = DIR + '/../data/'
TRACKS = DATA_DIR + 'tracks_genres_lyrics_en.csv.zip'

@router.get('/find')
def FindSongs(find_song_entries):
    self.tracks_df = pd.read_csv(TRACKS)
    return self.tracks_df.loc[entries]


@router.post('/song_entry')
def find_song_entry(self, x, best_choice=True):
    df = self.find_song_entries(x)

    choice = df.index.tolist()
    if best_choice:
        choice = choice[0]

    return df.loc[choice]
# def random_artist():
#     return random.choice(['Tones and I', 'Arizona Zervas', 'Post Malone'])


@router.get('/song')
def get_recommendations(self, x):
    gvec = self.genres_tfidf.transform([tokenize(x.genres)]).todense()
    fvec = self.scaler.transform([x[self.features]])
    vec = [fvec.tolist()[0] + gvec.tolist()[0]]
    encoded_vec = self.fg_encoder.predict(vec)
    entries = self.fg_nn.kneighbors(encoded_vec)[1][0].tolist()
    entries = self.tracks_df.iloc[entries].popularity. \
        sort_values(ascending=False).index.tolist()

    return self.tracks_df.loc[entries]

@router.get('/display')
def display_song_entries(x):
    entries = [get_song_info(y) for _,y in x.iterrows()]
    for entry in entries:
        return entry

@router.post('findsongrec')
def find_song_recs(x):
    gvec = genres_tfidf.transform([tokenize(x.genres)]).todense()
    fvec = scaler.transform([x[features]])
    vec = [fvec.tolist()[0] + gvec.tolist()[0]]
    encoded_vec = fg_encoder.predict(vec)
    entries = fg_nn.kneighbors(encoded_vec)[1][0].tolist()
    entries = tracks_df.iloc[entries].popularity.sort_values(ascending=False).index.tolist()
    return tracks_df.loc[entries]

# def select_nearest_songs(artist, song):
#
#     # loaded_model = pickle.load(open('nlp_model.sav', 'rb'))
#     loaded_model = joblib.load('app/loaded_model.joblib')
#
#     # translate artist, song into doc dtm.iloc[x].values
#     artist_songs = df1.loc[df1['track_artist'] == artist]
#     selected_song = artist_songs.loc[artist_songs['track_name'] == song]
#     x = selected_song.index
#     x = x.item()
#     #x = x.tolist()
#     x = 0
#     doc = dtm.loc[x].values
#     result = loaded_model.kneighbors([doc])
#
#     song1 = result[1][0][1]  # gives the loc
#     #x = x.item().remove()
#
#     # translate the loc into an artist and song title
#     artist1 = spotify_songs.loc[song1]['track_artist']
#     song1 = spotify_songs.loc[song1]['track_name']
#
#     # translate result into song names
#     return artist1, song1

class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    x1: float = Field(..., example=3.14)
    x2: int = Field(..., example=-42)
    x3: str = Field(..., example='banjo')

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

    @validator('x1')
    def x1_must_be_positive(cls, value):
        """Validate that x1 is a positive number."""
        assert value > 0, f'x1 == {value}, must be > 0'
        return value


@router.post('/predict')
async def predict(artist, song):
    # loaded_model = pickle.load(open('nlp_model.sav', 'rb'))
    loaded_model = joblib.load('app/loaded_model.joblib')

    #translate artist, song into doc dtm.iloc[x].values
    artist_songs = df1.loc[df1['track_artist'] == artist]
    selected_song = artist_songs.loc[artist_songs['track_name'] == song]
    x = selected_song.index
    print(x)
    x = x.item()
    # x = x.tolist()
    doc = dtm.loc[x].values
    result = loaded_model.kneighbors([doc], n_neighbors=6)

    rec_songs = {"artist": [], "song": []};

    for i in range(5):
        song = result[1][0][1 + i]

        # translate the loc into an artist and song title
        artist = df1.loc[song]['track_artist']
        song = df1.loc[song]['track_name']

        rec_songs['artist'].append(artist)
        rec_songs['song'].append(song)

    return rec_songs
    #}
