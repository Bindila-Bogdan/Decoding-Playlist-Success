from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, lower, explode, size, avg

sc = SparkContext(appName="webApp")
sc.setLogLevel("ERROR")
#sc.addPyFile("/home/s3307891/pandas-2.1.4.tar.gz")

spark = SparkSession(sc).builder.getOrCreate()
# Clean data: s3264440
# Original data: s3264424
playlists_data = spark.read.json("/user/s3264440/project_group_18/data/spotify_playlists/", multiLine=True)
audio_features = spark.read.json("/user/s3264440/project_group_18/data/audio_features/").distinct()
artist_data = spark.read.json("/user/s3264440/project_group_18/data/artists/").distinct()

def get_aggregated_audio_features(playlists_data,audio_features):

    #exploded_df = playlists_data.select("info", explode("playlists").alias("playlist"))

    exploded_df = playlists_data.select("pid", "tracks.track_uri","num_followers")

    pid_uri = exploded_df.select("pid",explode("track_uri").alias("uri"),"num_followers")
    # join the playlists dataset with the audio features dataset
    audio_features = audio_features.select(
        "uri",
        "acousticness",
        "danceability",
        "duration_ms",
        "energy",
        "instrumentalness",
        "key",
        "liveness",
        "loudness",
        "mode",
        "speechiness",
        "tempo",
        "valence"
    )
    joined_df = pid_uri.join(audio_features, pid_uri["uri"] == audio_features["uri"], "left_outer")
    joined_df = joined_df.groupBy('pid').agg(avg(col("num_followers")).alias("num_followers"),avg(col("acousticness")).alias("acousticness"),avg(col("danceability")).alias("danceability"),avg(col("duration_ms")).alias("duration_ms"),avg(col("energy")).alias("energy"),avg(col("instrumentalness")).alias("instrumentalness"),avg(col("key")).alias("key"),avg(col("liveness")).alias("liveness"),avg(col("loudness")).alias("loudness"),avg(col("mode")).alias("mode"),avg(col("speechiness")).alias("speechiness"),avg(col("tempo")).alias("tempo"),avg(col("valence")).alias("valence"))
    aggregated_df = joined_df.groupBy('num_followers').agg(avg(col("acousticness")).alias("acousticness"),avg(col("danceability")).alias("danceability"),avg(col("duration_ms")).alias("duration_ms"),avg(col("energy")).alias("energy"),avg(col("instrumentalness")).alias("instrumentalness"),avg(col("key")).alias("key"),avg(col("liveness")).alias("liveness"),avg(col("loudness")).alias("loudness"),avg(col("mode")).alias("mode"),avg(col("speechiness")).alias("speechiness"),avg(col("tempo")).alias("tempo"),avg(col("valence")).alias("valence"))

    output_path = "/user/s3307891/data_test"
    aggregated_df.write.mode("overwrite").option("header", "true").csv(output_path)

def mean_main_features(playlists_data):
    #exploded_df = playlists_data.select("info", explode("playlists").alias("playlist"))
    exploded_df_2 = playlists_data.select("pid", "num_tracks","num_artists","num_albums","num_followers")
    grouped_df = exploded_df_2.groupBy("num_followers").agg(avg("num_tracks").alias("mean_track_length"),avg("num_artists").alias("mean_num_artists"),avg("num_albums").alias("mean_num_albums"))
    output_path = "/user/s3307891/data_track_length"
    grouped_df.write.mode("overwrite").option("header", "true").csv(output_path)

def genres_per_playlist(playlists_data,artist_data):
    # data goes boom, explode it!
    exploded_df = playlists_data.select("info", explode("playlists").alias("playlist"))
    # more explosions!
    exploded_df = exploded_df.select("info", "playlist.pid", "playlist.tracks.artist_uri","playlist.num_followers")
    # boom shakalaka
    pid_uri = exploded_df.select("pid",explode("artist_uri").alias("uri"),"num_followers")

    joined_df = pid_uri.join(artist_data, pid_uri["uri"] == artist_data["uri"], "left_outer")
    joined_exploded = joined_df.select("pid",size("genres").alias("genre_count"),"num_followers").drop("id").drop("name")

def taylor_swift_feature(playlist_data,artist_data):
    pass


get_aggregated_audio_features(playlists_data,audio_features)
