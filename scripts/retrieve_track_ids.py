# import SparkSession, needed PySpark functions and other packages
import time
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    DoubleType,
    StringType,
)


# define used paths
PLAYLISTS_DATA_PATH = "/user/s3264424/project_group_18/data/spotify_playlists/"
STORED_DATA_PATH = "/user/s3264424/project_group_18/data/audio_features/"
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

        if response.status_code == 429:
            raise Exception("Spotify API calls limit has been reached!")

        audio_features = response.json()["audio_features"]

        float_features = [
            "danceability",
            "energy",
            "loudness",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
            "valence",
            "tempo",
        ]

        for index in range(len(audio_features)):
            for float_feature in float_features:
                audio_features[index][float_feature] = float(
                    audio_features[index][float_feature]
                )
        filtered_audio_features = [
            audio_features_
            for audio_features_ in audio_features
            if audio_features_ is not None
        ]

        return filtered_audio_features


class DataManipulator:
    schema = StructType(
        [
            StructField("danceability", DoubleType()),
            StructField("energy", DoubleType()),
            StructField("key", IntegerType()),
            StructField("loudness", DoubleType()),
            StructField("mode", IntegerType()),
            StructField("speechiness", DoubleType()),
            StructField("acousticness", DoubleType()),
            StructField("instrumentalness", DoubleType()),
            StructField("liveness", DoubleType()),
            StructField("valence", DoubleType()),
            StructField("tempo", DoubleType()),
            StructField("type", StringType()),
            StructField("id", StringType()),
            StructField("uri", StringType()),
            StructField("track_href", StringType()),
            StructField("analysis_url", StringType()),
            StructField("duration_ms", IntegerType()),
            StructField("time_signature", IntegerType()),
        ]
    )

    @staticmethod
    def get_track_ids():
        print("Get track ids...")

        # load data that contains the playlists
        playlists_data = spark.read.json(PLAYLISTS_DATA_PATH, multiLine=True)

        track_uris = (
            playlists_data.select(explode("playlists"))
            .select(explode("col.tracks"))
            .select("col.track_uri")
            .select("track_uri")
            .distinct()
            .collect()
        )

        track_ids = [
            track_uri["track_uri"].split("track:")[-1] for track_uri in track_uris
        ]

        return track_ids

    @classmethod
    def store(cls, audio_features):
        audio_features_df = spark.createDataFrame(
            audio_features, schema=DataManipulator.schema
        )
        audio_features_df.coalesce(1).write.mode("append").json(STORED_DATA_PATH)


if __name__ == "__main__":
    # parameters of the retrieval process
    batch_size = 100
    batches_per_file = 2
    request_per_minute = 100

    # instantiate the SparkSession class
    spark = SparkSession.builder.getOrCreate()

    # get track ids
    data_retriever = DataRetriever()
    track_ids = DataManipulator.get_track_ids()

    # compute number of batches
    batches_number = len(track_ids[:15]) // batch_size

    # this list stores the retrieve audio features
    all_audio_features = []

    # iterate over batches
    for batch_index in range(batches_number):
        print(f"Batch {batch_index + 1}/{batches_number}")

        # get audio features for songs from this batch
        audio_features = data_retriever.get_audio_features(
            ",".join(
                track_ids[batch_index * batch_size : (batch_index + 1) * batch_size]
            )
        )
        all_audio_features.extend(audio_features)

        # control the number of requests per minute to avoig getting the 429 error
        time.sleep(60 / request_per_minute)

        # at every 'batches_per_file' batches
        if ((batch_index + 1) % batches_per_file == 0) or (
            (batch_index + 1) == batches_number
        ):
            print("Storing audio features...")
            # store the retrieved data

            DataManipulator.store(all_audio_features)
            # deallocate memory

            del all_audio_features
            all_audio_features = []
