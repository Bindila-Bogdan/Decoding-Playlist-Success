from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, isnull, explode
from pyspark import SparkContext, SparkFiles

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()
sc.addPyFile('/home/s3264440/project_group_18/models/tables.py')
SparkFiles.get('tables.py')
from models.tables import *

def get_null_values_count(df, column):
    return df.filter(isnull(col(column))).count()

def has_null_values(df, column, get_value=False):
    null_no = get_null_values_count(df, column)
    if null_no == 0:
        return False
    if get_value:
        return True, null_no
    return True

def get_empty_array_count(df, column):
    return df.filter(size(col(column)) == 0).count()


if __name__ == "__main__":
    playlists = spark.read.json("/user/s3264424/project_group_18/data/spotify_playlists/", multiLine=True)
    playlists = playlists.select(explode("playlists")).select("col.*")

    audio_features = spark.read.json("/user/s3264424/project_group_18/data/audio_features/").distinct()
    artists = spark.read.json("/user/s3264424/project_group_18/data/artists/").distinct()

    print("Starting null analysis for artists table...")
    if has_null_values(artists, ArtistTable.ARTIST_ID):
        print("Warning: ", "Artist id can be null")
    if has_null_values(artists, ArtistTable.NAME):
        print("Warning: ", "Artist name can be null")
    if has_null_values(artists, ArtistTable.URI):
        print("Warning: ", "Artist uri can be null")
    
    print("Empty array values for genre column: ", get_empty_array_count(artists, ArtistTable.GENRES)/artists.count()) # approx. 58%
    print("Null analysis for artists table was done successfully")

    print("Starting null analysis for audio features table...")
    if has_null_values(audio_features, AudioFeaturesTable.URI):
        print("Warning: ", "Audio Features uri can be null")
    if has_null_values(audio_features, AudioFeaturesTable.TRACK_HREF):
        print("Warning: ", "Audio Features track_href can be null")
    if has_null_values(audio_features, AudioFeaturesTable.ANALYSIS_URL):
        print("Warning: ", "Audio Features analysis_url can be null")

    print("Null analysis for audio features table was done successfully")

    print("Starting null analysis for playlists table...")
    description_null_values = has_null_values(playlists, PlaylistTable.DESCRIPTION, True)
    if description_null_values[0]:
        print("Warning: ", f"Playlists description can be null: {description_null_values[1]/playlists.count()} null values") # 98%
    if has_null_values(playlists, PlaylistTable.NAME):
        print("Warning: ", "Playlists name can be null")
        
    print("Null analysis for playlists table was done successfully")



