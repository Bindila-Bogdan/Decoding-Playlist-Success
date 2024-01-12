from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, sum as _sum, lower, regexp_extract, split 

sc = SparkContext(appName="Playlist Analysis")

spark = SparkSession(sc).builder.getOrCreate()

playlists_df = spark.read.json("/user/s3264424/project_group_18/data/spotify_playlists/", multiLine=True)


# Analysis 1: Playlist Composition
# Number of artists, albums, and tracks in each playlist
exploded_playlists_df = playlists_df.select(explode("playlists").alias("playlist"))
playlist_composition = exploded_playlists_df.select("playlist.pid","playlist.num_artists","playlist.num_albums", "playlist.num_tracks")

# Analysis 2: Song Length Distribution
# Distribution of song lengths within each playlist
playlist_tracks = exploded_playlists_df.select("playlist.pid", explode("playlist.tracks").alias("track"))
track_durations = playlist_tracks.select("pid", col("track.duration_ms"))

# Analysis 3: Playlist Duration
# Total duration of each playlist
playlist_duration = track_durations.groupBy("pid").agg(_sum("duration_ms").alias("total_duration"))

# Analysis 4: Playlist Name Analysis
playlist_names = exploded_playlists_df.select("playlist.pid", "playlist.name")
playlist_names_lower = playlist_names.select("pid", lower(col("name")).alias("name_lower"))

#pattern = "(\\w+)"  # for extracting words
#word_counts = playlist_names_lower.select("pid",explode(regexp_extract("name_lower", pattern, 0)).alias("word")).groupBy("word").count().orderBy('count', ascending=False)

words_df = playlist_names_lower.select("pid", split(col("name_lower"), " ").alias("words"))
word_counts = words_df.select("pid", explode("words").alias("word")).groupBy("word").count().orderBy('count', ascending=False)

# Writing to csv
path_playlist_composition = "/user/s3307913/PLAYLISTS/COMPOSITION"
path_track_durations = "/user/s3307913/PLAYLISTS/TRACKDURATION"
path_playlist_duration = "/user/s3307913/PLAYLISTS/PLAYLISTDURATION"
path_word_counts = "/user/s3307913/PLAYLISTS/PLAYLISTNAMEWC"

playlist_composition.write.csv(path_playlist_composition, header=True, mode='overwrite')
track_durations.write.csv(path_track_durations, header=True, mode='overwrite')
playlist_duration.write.csv(path_playlist_duration, header=True, mode='overwrite')
word_counts.write.csv(path_word_counts, header=True, mode='overwrite')
