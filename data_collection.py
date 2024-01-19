from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, lower, explode, size, avg, expr, array_contains, when

sc = SparkContext(appName="webApp")
sc.setLogLevel("ERROR")
#sc.addPyFile("/home/s3307891/pandas-2.1.4.tar.gz")

spark = SparkSession(sc).builder.getOrCreate()
# Clean data: s3264440
# Original data: s3264424
playlists_data = spark.read.json("/user/s3264424/project_group_18/data/spotify_playlists/", multiLine=True)
audio_features = spark.read.json("/user/s3264440/project_group_18/data/audio_features/").distinct()
artist_data = spark.read.json("/user/s3264440/project_group_18/data/artists/").distinct()

def get_aggregated_audio_features(playlists_data,audio_features,data_type='mean'):

    exploded_df = playlists_data.select("info", explode("playlists").alias("playlist"))

    exploded_df = exploded_df.select("playlist.pid", "playlist.tracks.track_uri","playlist.num_followers")

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
    if data_type == "mean":
        joined_df = joined_df.groupBy('pid').agg(avg(col("num_followers")).alias("num_followers"),avg(col("acousticness")).alias("acousticness"),avg(col("danceability")).alias("danceability"),avg(col("duration_ms")).alias("duration_ms"),avg(col("energy")).alias("energy"),avg(col("instrumentalness")).alias("instrumentalness"),avg(col("key")).alias("key"),avg(col("liveness")).alias("liveness"),avg(col("loudness")).alias("loudness"),avg(col("mode")).alias("mode"),avg(col("speechiness")).alias("speechiness"),avg(col("tempo")).alias("tempo"),avg(col("valence")).alias("valence"))
        aggregated_df = joined_df.groupBy('num_followers').agg(avg(col("acousticness")).alias("acousticness"),avg(col("danceability")).alias("danceability"),avg(col("duration_ms")).alias("duration_ms"),avg(col("energy")).alias("energy"),avg(col("instrumentalness")).alias("instrumentalness"),avg(col("key")).alias("key"),avg(col("liveness")).alias("liveness"),avg(col("loudness")).alias("loudness"),avg(col("mode")).alias("mode"),avg(col("speechiness")).alias("speechiness"),avg(col("tempo")).alias("tempo"),avg(col("valence")).alias("valence"))
    elif data_type == "median":
        joined_df = joined_df.groupBy('pid').agg(expr('percentile(num_followers, 0.5)').alias('num_followers'),expr('percentile(acousticness, 0.5)').alias('acousticness'),expr('percentile(danceability, 0.5)').alias('danceability'),expr('percentile(duration_ms, 0.5)').alias('duration_ms'),expr('percentile(energy, 0.5)').alias('energy'),expr('percentile(instrumentalness, 0.5)').alias('instrumentalness'),expr('percentile(key, 0.5)').alias('key'),expr('percentile(liveness, 0.5)').alias('liveness'),expr('percentile(loudness, 0.5)').alias('loudness'),expr('percentile(mode, 0.5)').alias('mode'),expr('percentile(speechiness, 0.5)').alias('speechiness'),expr('percentile(tempo, 0.5)').alias('tempo'),expr('percentile(valence, 0.5)').alias('valence'))
        aggregated_df = joined_df.groupBy('num_followers').agg(expr('percentile(acousticness, 0.5)').alias('acousticness'),expr('percentile(danceability, 0.5)').alias('danceability'),expr('percentile(duration_ms, 0.5)').alias('duration_ms'),expr('percentile(energy, 0.5)').alias('energy'),expr('percentile(instrumentalness, 0.5)').alias('instrumentalness'),expr('percentile(key, 0.5)').alias('key'),expr('percentile(liveness, 0.5)').alias('liveness'),expr('percentile(loudness, 0.5)').alias('loudness'),expr('percentile(mode, 0.5)').alias('mode'),expr('percentile(speechiness, 0.5)').alias('speechiness'),expr('percentile(tempo, 0.5)').alias('tempo'),expr('percentile(valence, 0.5)').alias('valence'))
    aggregated_df.show()

    output_path = f"/user/s3307891/aggregated_audio_{data_type}"
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

    joined_df = pid_uri.join(artist_data, pid_uri["uri"] == artist_data["uri"], "left_outer").drop("id").drop("name")
    filtered_joined = joined_df.filter(size(col("genres")) > 0)
    joined_exploded = filtered_joined.select("pid",size("genres").alias("genre_count"),"num_followers")
    pid_grouped = joined_exploded.groupBy("pid").agg(avg("genre_count").alias("genre_count"),avg("num_followers").alias("num_followers"))
    followers_grouped = pid_grouped.groupBy("num_followers").agg(avg("genre_count").alias("genre_count"))

    output_path = "/user/s3307891/genres_followers"
    followers_grouped.write.mode("overwrite").option("header", "true").csv(output_path)


def taylor_swift_feature(playlists_data,artist_data):
    exploded_df = playlists_data.select(explode("playlists").alias("playlist"))
    exploded_df = exploded_df.select("playlist.pid", "playlist.tracks.artist_name","playlist.num_followers")
    #taylor_swift_df = exploded_df.withColumn("is_taylor_swift", when(array_contains(col("artist_name"), "Taylor Swift"), 1).otherwise(0))
    taylor_swift_df = exploded_df.withColumn("taylor_swift_count", expr("size(filter(artist_name, x -> x == 'Taylor Swift'))"))
    relevant_df = taylor_swift_df.groupBy("num_followers").agg(avg("taylor_swift_count").alias("avg_taylor_swift")).drop("pid").drop("artist_name")
    output_path = "/user/s3307891/taylor_feature"
    relevant_df.write.mode("overwrite").option("header", "true").csv(output_path)

get_aggregated_audio_features(playlists_data,audio_features)
