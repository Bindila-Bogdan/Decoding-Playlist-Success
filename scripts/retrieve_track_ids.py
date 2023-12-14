# import SparkSession, needed PySpark functions and other packages
import time
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode

# instantiate the SparkSession class
spark = SparkSession.builder.getOrCreate()

# define used paths
PLAYLISTS_DATA_PATH = "/user/s3264424/project_group_18/data/spotify_playlists/"
STORED_DATA_PATH = "/user/s3264424/project_group_18/data/audio_features"
CREDENTIALS_FILE_PATH = "./spotify_key.config"
 

class DataRetriever:
    def __init__(self):
        self.__auth_token = self.__get_auth_token()

    def __get_auth_token(self):
        with open(CREDENTIALS_FILE_PATH, "r") as input_file:
            credentials = input_file.read()
        client_id, client_secret = credentials.split("\n")

        response = requests.post(
            "https://accounts.spotify.com/api/token",
            {
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
        )

        auth_token = response.json()["access_token"]
        
        return auth_token.strip()

    def get_audio_features(self, track_ids):
        response = requests.get(
            f"https://api.spotify.com/v1/audio-features?ids={track_ids}",
            headers={"Authorization": f"Bearer {self.__auth_token}"},
        )

        audio_features = response.json()["audio_features"]

        return audio_features

class DataManipulator:
    @staticmethod
    def get_track_ids():
        # load data that contains the playlists
        playlists_data = spark.read.json(
            PLAYLISTS_DATA_PATH, multiLine=True
        )

        track_uris = (
            playlists_data.select(explode("playlists"))
            .select(explode("col.tracks"))
            .select("col.track_uri")
            .select("track_uri")
            .distinct()
            .collect()
        )

        track_ids = [track_uri["track_uri"].split("track:")[-1] for track_uri in track_uris]

        return track_ids


    @staticmethod
    def store(audio_features):
        audio_features_df = spark.createDataFrame(audio_features)
        audio_features_df.coalesce(1).write.mode("overwrite").json(STORED_DATA_PATH)


if __name__ == "__main__":
    batch_size = 100
    request_per_minute = 100
    batches_per_file = 20

    data_retriever = DataRetriever()
    track_ids = DataManipulator.get_track_ids()
    print(len(track_ids))

    all_audio_features = []

    for batch_index in range(len(track_ids[:batch_size]) // batch_size):
        print("inside the loop")
        audio_features = data_retriever.get_audio_features(
            ",".join(track_ids[batch_index * batch_size : (batch_index + 1) * batch_size])
        )
        all_audio_features.extend(audio_features)

        time.sleep(60 / request_per_minute)

        print(all_audio_features[0])

        #if batch_index != 0 and batch_index % batches_per_file == 0: 
        #DataManipulator.store(all_audio_features)
        #del all_audio_features
        #all_audio_features = []

        print("finished")