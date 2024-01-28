from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, explode
from pyspark import SparkContext, SparkFiles

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()
sc.addPyFile('/home/s3264440/project_group_18/models/tables.py')
SparkFiles.get('tables.py')
from models.tables import *

def drop_columns_from_df(df, columns, nested_column=None, is_nested=False):
    if is_nested:
        nested_columns = ', '.join([f'x.{column}' for column in columns])
        new_df = df.withColumn(nested_column, expr(f"transform({nested_column}, x -> struct({nested_columns}))"))
    else:
        new_df = df.drop(*columns)

    return new_df

if __name__ == "__main__":
    playlists = spark.read.json("/user/s3264424/project_group_18/data/spotify_playlists/", multiLine=True)
    audio_features = spark.read.json("/user/s3264424/project_group_18/data/audio_features/").distinct()

    artists = spark.read.json("/user/s3264424/project_group_18/data/artists/").distinct()
    artists.write.mode("append").parquet("/user/s3264440/project_group_18/data/artists")

    audio_features = drop_columns_from_df(audio_features, [AudioFeaturesTable.ANALYSIS_URL, AudioFeaturesTable.TRACK_HREF, AudioFeaturesTable.TYPE])
    audio_features.write.mode("append").parquet("/user/s3264440/project_group_18/data/audio_features") # store the clean data
    print("SUCCES: audio_features.parquet successufully written.")

    playlists = playlists.select(explode("playlists")).select("col.*") # drop the info column
    new_playlists = drop_columns_from_df(playlists, [PlaylistTable.DESCRIPTION, PlaylistTable.MODIFIED_AT])
    new_playlists = drop_columns_from_df(new_playlists, [PlaylistTable.ALBUM_NAME, PlaylistTable.ARTIST_URL, PlaylistTable.TRACK_NAME, PlaylistTable.TRACK_URI], PlaylistTable.TRACKS, True)
    new_playlists.write.mode("append").parquet("/user/s3264440/project_group_18/data/spotify_playlists") # store the clean data
    print("SUCCES: spotify_playlists.parquet successufully written.")