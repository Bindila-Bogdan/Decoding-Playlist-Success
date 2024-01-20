from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, max, min, stddev, sum, explode, split, col, length, from_unixtime, to_date

sc = SparkContext(appName="Playlist Analysis")

spark = SparkSession(sc).builder.getOrCreate()

playlists_df = spark.read.json("/user/s3264440/project_group_18/data/spotify_playlists/", multiLine=True)

# Analysis 1: Playlist Composition Aggregation
playlist_composition_agg = playlists_df.agg(avg("num_artists").alias("avg_num_artists"),max("num_artists").alias("max_num_artists"),min("num_artists").alias("min_num_artists"),stddev("num_artists").alias("stddev_num_artists"),avg("num_albums").alias("avg_num_albums"),max("num_albums").alias("max_num_albums"),min("num_albums").alias("min_num_albums"),stddev("num_albums").alias("stddev_num_albums"),avg("num_tracks").alias("avg_num_tracks"),max("num_tracks").alias("max_num_tracks"),min("num_tracks").alias("min_num_tracks"),stddev("num_tracks").alias("stddev_num_tracks"))

path_playlist_composition = "/user/s3307913/CLEAN/PLAYLISTS/COMPOSITION"
playlist_composition_agg.write.csv(path_playlist_composition, header=True, mode='overwrite')

# Analysis 2: Playlist Duration Aggregation
playlist_duration_stats = playlists_df.describe("duration_ms")

path_playlist_duration = "/user/s3307913/CLEAN/PLAYLISTS/PLAYLISTDURATION"
playlist_duration_stats.write.csv(path_playlist_duration, header=True, mode='overwrite')

# Analysis 3: Playlist Name Analysis
word_counts = playlists_df.select(explode(split(col("name"), "\s+")).alias("word")).groupBy("word").count().orderBy("count", ascending=False)

path_word_counts = "/user/s3307913/CLEAN/PLAYLISTS/PLAYLISTNAMEWC"
word_counts.write.csv(path_word_counts, header=True, mode='overwrite')

name_length = playlists_df.withColumn("name_length", length("name"))
name_length_stats = name_length.describe("name_length")

path_name_length = "/user/s3307913/CLEAN/PLAYLISTS/PLAYLISTNAMELENGTH"
name_length_stats.write.csv(path_name_length, header=True, mode='overwrite')

# Analysis 4: Playlist Modification 
modification_date_df = playlists_df.withColumn("modified_date", to_date(from_unixtime("modified_at")))
modification_trends = modification_date_df.groupBy("modified_date").count().orderBy("modified_date")

path_modification_trends = "/user/s3307913/CLEAN/PLAYLISTS/PLAYLISTMODITRENDS"
modification_trends.write.csv(path_modification_trends, header=True, mode='overwrite')

# Analysis 5: Follower Count Statitics
follower_counts_stats = playlists_df.describe("num_followers")

path_follower_stats = "/user/s3307913/CLEAN/PLAYLISTS/PLAYLISTFOLLOWERSTATS"
follower_counts_stats.write.csv(path_follower_stats, header=True, mode='overwrite')

# Analysis 6: Playlist Descriptions  -- DIDNT END UP USING YET
# Tokenizing descriptions
words_in_description = playlists_df.withColumn("word", explode(split(col("description"), "\s+")))
# Counting word frequency
word_counts = words_in_description.groupBy("word").count().orderBy("count", ascending=False)

path_word_counts = "/user/s3307913/CLEAN/PLAYLISTS/PLAYLISTNAMEDESC"
word_counts.write.csv(path_word_counts, header=True, mode='overwrite')

# Analysis 7: Edit Stats
edit_counts_stats = playlists_df.describe("num_edits")

path_edit_counts_stats = "/user/s3307913/CLEAN/PLAYLISTS/PLAYLISTEDITS"
edit_counts_stats.write.csv(path_edit_counts_stats, header=True, mode='overwrite')

# Analysis 8: Collaborative Counts
collaborative_counts = playlists_df.groupBy("collaborative").count()

path_collaborative_counts = "/user/s3307913/CLEAN/PLAYLISTS/PLAYLISTTYPE"
collaborative_counts.write.csv(path_collaborative_counts, header=True, mode='overwrite')
