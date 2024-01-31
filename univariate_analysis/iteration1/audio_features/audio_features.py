from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, explode

sc = SparkContext(appName="Audio Features Analysis")

spark = SparkSession(sc).builder.getOrCreate()

audio_features_df = spark.read.json("/user/s3264440/project_group_18/data/audio_features/").distinct()


# Analysis 1: Song Level Audio Features Analysis
audio_features = audio_features_df.select("id", "acousticness", "danceability", "energy","instrumentalness", "liveness", "loudness", "speechiness","tempo", "valence")
audio_features_summary = audio_features.describe()
#audio_features_summary.show()
path_audio_features_summary = "/user/s3307913/FINAL/AUDIO/SONGLEVEL"
audio_features_summary.write.csv(path_audio_features_summary, header=True, mode='overwrite')

# Analysis 2: Track Duration Analysis
track_duration_stats = audio_features_df.describe("duration_ms")

path_track_duration_stats = "/user/s3307913/FINAL/AUDIO/TRACKDURATION"
track_duration_stats.write.csv(path_track_duration_stats, header=True,mode='overwrite')

# Analysis 3: Mode of Tracks
mode_distribution = audio_features_df.groupBy("mode").count()

path_mode_distribution = "/user/s3307913/FINAL/AUDIO/MODEDISTRIBUTION"
mode_distribution.write.csv(path_mode_distribution, header=True, mode='overwrite')




