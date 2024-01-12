from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, explode

sc = SparkContext(appName="Audio Features Analysis")

spark = SparkSession(sc).builder.getOrCreate()

audio_features_df = spark.read.json("/user/s3264424/project_group_18/data/audio_features/").distinct()
playlists_df = spark.read.json("/user/s3264424/project_group_18/data/spotify_playlists/", multiLine=True)


# Analysis 1: Song Level Audio Features Analysis
audio_features = audio_features_df.select("id", "acousticness", "danceability", "energy","instrumentalness", "liveness", "loudness", "speechiness","tempo", "valence")
audio_features_summary = audio_features.describe()
#audio_features_summary.show()

# Analysis 2: Playlist Level Audio Feature Aggregation
# Assuming that each track in playlists_df is linked to audio features by 'track_uri'
exploded_playlists_df = playlists_df.select(explode("playlists").alias("playlist"))
playlist_tracks_df = exploded_playlists_df.select("playlist.pid", explode("playlist.tracks").alias("track")).select("pid", "track.track_uri")

# Joining with audio features
playlist_audio_features_df = playlist_tracks_df.join(audio_features_df, playlist_tracks_df.track_uri == audio_features_df.uri)

# Aggregating audio features for each playlist
playlist_audio_features = playlist_audio_features_df.groupBy("pid").agg(avg("acousticness").alias("avg_acousticness"),avg("danceability").alias("avg_danceability"),avg("energy").alias("avg_energy"),avg("instrumentalness").alias("avg_instrumentalness"),avg("liveness").alias("avg_liveness"),avg("loudness").alias("avg_loudness"),avg("speechiness").alias("avg_speechiness"),avg("tempo").alias("avg_tempo"),avg("valence").alias("avg_valence"))

#playlist_audio_features.show()


# Writing to csv
path_audio_features_summary = "/user/s3307913/AUDIO/SONGLEVEL"
path_playlist_audio_features = "/user/s3307913/AUDIO/PLAYLISTLEVEL"


audio_features_summary.write.csv(path_audio_features_summary, header=True, mode='overwrite')
playlist_audio_features.write.csv(path_playlist_audio_features, header=True, mode='overwrite')
