# import Python packages, PySpark classes and functionalities
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
    ArrayType,
)

# define used paths
PLAYLISTS_DATA_PATH = "/user/s3264424/project_group_18/data/spotify_playlists/"
ARTISTS_DATA_PATH = "/user/s3264424/project_group_18/data/artists/"
AUDIO_FEATURES_DATA_PATH = "/user/s3264424/project_group_18/data/audio_features/"
CREDENTIALS_FILE_PATH = "./spotify_key.config"

# define the type of data that is retrieved
ID_TYPE = "artist"


class DataRetriever:
    """
    This class is used to get data from the Spotify API.
    """

    def __init__(self):
        self.__auth_token = self.__get_auth_token()

    def __check_errors(self, response):
        # stop the program if the status code is 429
        if response.status_code == 429:
            raise Exception("Spotify API calls limit has been reached!")
        elif response.status_code != 200:
            print(f"Response code: {response.status_code}")

    def __get_auth_token(self):
        """
        It gets the Spotify API credentials from file and
        returns the access token.
        """

        # read credentials
        with open(CREDENTIALS_FILE_PATH, "r") as input_file:
            credentials = input_file.read()
        client_id, client_secret = credentials.split("\n")

        # request token and return it
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
        """
        This method takes as input a string with comma-separated
        track ids and returns for each track the audio features
        stored in a list.
        """

        # request audio features
        response = requests.get(
            f"https://api.spotify.com/v1/audio-features?ids={track_ids}",
            headers={"Authorization": f"Bearer {self.__auth_token}"},
        )

        # check the returned request code
        self.__check_errors(response)

        audio_features = response.json()["audio_features"]

        # convert the following features to float values
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

        # this is done to avoid PySpark data type errors when creating the Data Frame
        for index in range(len(audio_features)):
            for float_feature in float_features:
                if audio_features[index] is not None:
                    audio_features[index][float_feature] = float(
                        audio_features[index][float_feature]
                    )

        # remove None values that correspond to unfound tracks
        filtered_audio_features = [
            audio_features_
            for audio_features_ in audio_features
            if audio_features_ is not None
        ]

        return filtered_audio_features

    def get_artist_data(self, artist_ids):
        """
        This method takes as input a string with comma-separated
        artist ids and returns for each artist the retrieved info
        stored in a list.
        """

        # request artist data
        response = requests.get(
            f"https://api.spotify.com/v1/artists?ids={artist_ids}",
            headers={"Authorization": f"Bearer {self.__auth_token}"},
        )

        # check the returned request code
        self.__check_errors(response)

        artists_data = response.json()["artists"]

        # keep only features of interest and remove None values that correspond to unfound artists
        filtered_artists_data = [
            {k: artist_data_[k] for k in ["genres", "id", "name", "uri"]}
            for artist_data_ in artists_data
            if artist_data_ is not None
        ]

        return filtered_artists_data


class DataManipulator:
    """
    This class focuses on retrieving the track and artist data from the
    playlists, convert them to a PySpark Data Frame and stores
    the info as JSON files.
    """

    # define the schemas used when creating the PySpark Data Frame
    audio_features_schema = StructType(
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

    artists_schema = StructType(
        [
            StructField("genres", ArrayType(StringType())),
            StructField("id", StringType()),
            StructField("name", StringType()),
            StructField("uri", StringType()),
        ]
    )

    @staticmethod
    def get_ids():
        """
        This method loads the playlists data and returns a sorted list
        of unique track or artist ids.
        """

        print(f"Get {ID_TYPE} ids...")

        # load data that contains the playlists
        playlists_data = spark.read.json(PLAYLISTS_DATA_PATH, multiLine=True)

        # extract distinct uris
        uris = (
            playlists_data.select(explode("playlists"))
            .select(explode("col.tracks"))
            .select(f"col.{ID_TYPE}_uri")
            .select(f"{ID_TYPE}_uri")
            .distinct()
            .collect()
        )

        # get only ids and sort them
        ids = sorted([uri[f"{ID_TYPE}_uri"].split(ID_TYPE + ":")[-1] for uri in uris])

        return ids

    @classmethod
    def store(cls, data):
        """
        It creates the PySpark Data Frame from the audio features
        or artists info and stores it as a JSON file.
        """

        # pick storage path and schema based on the retrieved data type
        if ID_TYPE == "artist":
            storage_path = ARTISTS_DATA_PATH
            schema = cls.artists_schema
        elif ID_TYPE == "track":
            storage_path = AUDIO_FEATURES_DATA_PATH
            schema = cls.audio_features_schema

        # create PySpark Data Frame
        data_df = spark.createDataFrame(data, schema=schema)

        # reduce the number of partitions to 1 and store data
        data_df.coalesce(1).write.mode("append").json(storage_path)


if __name__ == "__main__":
    # set parameters of the retrieval process
    if ID_TYPE == "artist":
        batch_size = 50
    elif ID_TYPE == "track":
        batch_size = 100
    batches_per_file = 100
    request_per_minute = 50

    # instantiate the SparkSession class
    spark = SparkSession.builder.getOrCreate()

    # get track ids
    data_retriever = DataRetriever()
    ids = DataManipulator.get_ids()

    # compute number of batches
    batches_number = len(ids) // batch_size

    # this list stores the retrieved data
    all_retrieved_data = []

    # iterate over batches
    for batch_index in range(batches_number):
        # skip already retrieved batches
        if (batch_index + 1) <= 0:
            continue
        print(f"Batch {batch_index + 1}/{batches_number}")

        # get audio features or artist info for tracks from this batch
        current_ids = ",".join(
            ids[batch_index * batch_size : (batch_index + 1) * batch_size]
        )

        if ID_TYPE == "artist":
            retrieved_data = data_retriever.get_artist_data(current_ids)
        elif ID_TYPE == "track":
            retrieved_data = data_retriever.get_audio_features(current_ids)

        all_retrieved_data.extend(retrieved_data)

        # control the number of requests per minute to avoid getting the 429 error
        time.sleep(60 / request_per_minute)

        # at every 'batches_per_file' batches
        if ((batch_index + 1) % batches_per_file == 0) or (
            (batch_index + 1) == batches_number
        ):
            print(f"Storing {ID_TYPE} data...")
            # store the retrieved data
            DataManipulator.store(all_retrieved_data)
            # deallocate memory
            del all_retrieved_data
            all_retrieved_data = []
