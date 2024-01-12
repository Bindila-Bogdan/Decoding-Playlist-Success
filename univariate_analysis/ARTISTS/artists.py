from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, size

sc = SparkContext(appName="Artist Analysis")

spark = SparkSession(sc).builder.getOrCreate()

artists_df = spark.read.json("/user/s3264424/project_group_18/data/artists/").distinct()

# Analysis 1: Most Popular Genres
exploded_genres_df = artists_df.select(explode("genres").alias("genre"))
genre_counts = exploded_genres_df.groupBy("genre").count().orderBy('count', ascending=False)

path_genre_counts = "/user/s3307913/ARTISTS/GENRECOUNTS"
#genre_counts.toPandas().to_csv("genre_counts.csv", index=False)
genre_counts.write.csv(path_genre_counts, header=True, mode='overwrite')

# Analysis 2: Genres Associated with Artists
# Counting the number of genres per artist
artists_with_genre_count = artists_df.withColumn("num_genres", size("genres"))
genre_count_stats = artists_with_genre_count.describe("num_genres")

path_genre_stats = "/user/s3307913/ARTISTS/GENRESTATS"
genre_count_stats.write.csv(path_genre_stats, header=True, mode='overwrite')