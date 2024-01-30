from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, size, countDistinct

sc = SparkContext(appName="Artist Analysis")

spark = SparkSession(sc).builder.getOrCreate()

artists_df = spark.read.json("/user/s3264440/project_group_18/data/artists/").distinct()

# Analysis 1: Most Popular Genres
exploded_genres_df = artists_df.select(explode("genres").alias("genre"))
genre_counts = exploded_genres_df.groupBy("genre").count().orderBy('count', ascending=False)

path_genre_counts = "/user/s3307913/FINAL/ARTISTS/GENRECOUNTS"
genre_counts.write.csv(path_genre_counts, header=True, mode='overwrite')

# Analysis 2: Genres Stats
artists_with_genre_count = artists_df.withColumn("num_genres", size("genres"))
genre_count_stats = artists_with_genre_count.describe("num_genres")

path_genre_stats = "/user/s3307913/FINAL/ARTISTS/GENRESTATS"
genre_count_stats.write.csv(path_genre_stats, header=True, mode='overwrite')

# Analysis 3: Number of Unique Genres
genre_diversity = artists_df.select(explode("genres").alias("genre")).agg(countDistinct("genre"))

path_unique_genre_counts = "/user/s3307913/FINAL/ARTISTS/UNIQUEGENRECOUNT"
genre_diversity.write.csv(path_unique_genre_counts, header=True, mode='overwrite')

# Analysis 4: Number of Genres Associated with Artists 
genres_per_artist = artists_df.withColumn("genre", explode("genres")).groupBy("id").agg(countDistinct("genre").alias("num_genres"))

path_genres_per_artists ="/user/s3307913/FINAL/ARTISTS/GENRESPERARTISTS"
genres_per_artist.write.csv(path_genres_per_artists, header=True, mode='overwrite')