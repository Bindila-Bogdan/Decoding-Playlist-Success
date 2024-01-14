from pyspark.sql import SparkSession
from pyspark.sql.functions import min, max, explode
from pyspark import SparkContext, SparkFiles

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()
sc.addPyFile('/home/s3264440/project_group_18/models/tables.py')
SparkFiles.get('tables.py')
from models.tables import *

def get_interval(df, column):
	min_value = df.agg(min(column)).collect()[0][0]
	max_value = df.agg(max(column)).collect()[0][0]
	return min_value, max_value

if __name__ == "__main__":
    playlists = spark.read.json("/user/s3264424/project_group_18/data/spotify_playlists/", multiLine=True)
    playlists = playlists.select(explode("playlists")).select("col.*")

    audio_features = spark.read.json("/user/s3264424/project_group_18/data/audio_features/").distinct()

    print("Starting interval analysis for audio features table...")
    fields_to_analyze = [AudioFeaturesTable.ACOUSTICNESS, 
                         AudioFeaturesTable.DANCEABILITY, 
                         AudioFeaturesTable.DURATION_MS, 
                         AudioFeaturesTable.ENERGY, 
                         AudioFeaturesTable.INSTRUMENTALNESS, 
                         AudioFeaturesTable.LIVENESS, 
                         AudioFeaturesTable.LOUDNESS, 
                         AudioFeaturesTable.SPEECHINESS,
                         AudioFeaturesTable.TEMPO, 
                         AudioFeaturesTable.TIME_SIGNATURE, 
                         AudioFeaturesTable.VALENCE, 
                         AudioFeaturesTable.KEY, 
                         AudioFeaturesTable.MODE]
    
    for field in fields_to_analyze:
          interval = get_interval(audio_features, field)
          print(f"Interval values for {field} is {interval}.") 

    print("Interval analysis for audio features table was done successfully")

    print("Starting interval analysis for playlist table...")
    fields_to_analyze = [PlaylistTable.NUM_ALBUMS,
                         PlaylistTable.NUM_ARTISTS,
                         PlaylistTable.NUM_EDITS,
                         PlaylistTable.NUM_FOLLOWERS,
                         PlaylistTable.NUM_TRACKS]
    
    for field in fields_to_analyze:
          interval = get_interval(playlists, field)
          print(f"Interval values for {field} is {interval}.") 

    print("Interval analysis for playlist table was done successfully")
