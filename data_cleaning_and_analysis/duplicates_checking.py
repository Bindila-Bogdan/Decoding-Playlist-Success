from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
from pyspark import SparkContext, SparkFiles

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()
sc.addPyFile('/home/s3264440/project_group_18/models/tables.py')
SparkFiles.get('tables.py')
from models.tables import *

def get_duplicate_values_count(df, column):
    return df.groupBy(column).count().filter(col("count") > 1).count()

def has_duplicate_values(df, column):
    if get_duplicate_values_count(df, column) == 0:
        return False
    return True


if __name__ == "__main__":
    playlists = spark.read.json("/user/s3264424/project_group_18/data/spotify_playlists/", multiLine=True)
    playlists = playlists.select(explode("playlists")).select("col.*")

    audio_features = spark.read.json("/user/s3264424/project_group_18/data/audio_features/").distinct()
    artists = spark.read.json("/user/s3264424/project_group_18/data/artists/").distinct()
    
    print("Starting duplicate analysis for artists table...")
    if has_duplicate_values(artists, ArtistTable.ARTIST_ID):
        print("Warning: ", "Artist id has duplicates")
    if has_duplicate_values(artists, ArtistTable.NAME):
        print("Warning: ", "Artist name has duplicates")
    if has_duplicate_values(artists, ArtistTable.URI):
        print("Warning: ", "Artist uri has duplicates")
    
    print("Duplicates analysis for artists table was done successfully")

    print("Starting duplicates analysis for audio features table...")
    if has_duplicate_values(audio_features, AudioFeaturesTable.URI):
        print("Warning: ", "Audio Features uri has duplicates")
    if has_duplicate_values(audio_features, AudioFeaturesTable.TRACK_HREF):
        print("Warning: ", "Audio Features track_href has duplicates")
    if has_duplicate_values(audio_features, AudioFeaturesTable.ANALYSIS_URL):
        print("Warning: ", "Audio Features analysis_url has duplicates")

    print("Duplicates analysis for audio features table was done successfully")

    print("Starting duplicates analysis for playlists table...")
    if has_duplicate_values(playlists, PlaylistTable.NAME):
        print("Warning: ", "Playlists name has duplicates")
        
    print("Duplicates analysis for playlists table was done successfully")




