# import SparkSession and functions for data preparation and modeling
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import (
    mean,
    col,
    lit,
    log,
    percentile_approx,
)


# instantiate the SparkSession class
spark = SparkSession.builder.getOrCreate()

# define parameters
SEED = 0
NUM_FOLDS = 3
OPTIMIZATION_METRIC = "r2"

REMOVE_OUTLIERS = True
REMOVE_SKEWENSS = True
PERFORM_UNDERSAMPLING = True

DATA_PATH = "/user/s3307891/all_aggregated_features"
MODEL_PATH = "/user/s3264424/project_group_18/random_forest_regressor/"
DATA_STORAGE_PATH = "/user/s3264424/project_group_18/data/regression_predictions/"


def measure_performance(train_validation_df, test_df, baseline_evaluator):
    if baseline_evaluator:
        prediction_col = "baseline_prediction"
        measurement_type = "baseline"
    else:
        prediction_col = "prediction"
        measurement_type = ""

    # compute rmse metric
    rmse_evaluator = RegressionEvaluator(labelCol="num_followers", predictionCol=prediction_col, metricName="rmse")
    print(f"Train-validation {measurement_type} rmse: {rmse_evaluator.evaluate(train_validation_df)}")
    print(f"Test {measurement_type} rmse: {rmse_evaluator.evaluate(test_df)}")

    # compute r2 metric
    r2_evaluator = RegressionEvaluator(labelCol="num_followers", predictionCol=prediction_col, metricName="r2")
    print(f"Train-validation {measurement_type} r2: {r2_evaluator.evaluate(train_validation_df)}")
    print(f"Test {measurement_type} r2: {r2_evaluator.evaluate(test_df)}")


# load aggregated features
agg_features_df = spark.read.csv(DATA_PATH, header="true", inferSchema="true").distinct()

# remove outliers that have a value largest than the defined percentile threshold
if REMOVE_OUTLIERS:
    outlier_treshold = agg_features_df.select(percentile_approx("num_followers", 0.999)).collect()[0][0]
    print(f"Outlier threshold: {outlier_treshold}")
    agg_features_df = agg_features_df.filter(col("num_followers") < outlier_treshold)

# undersample the playlists with one follower to match the number of the ones with two followers 
if PERFORM_UNDERSAMPLING:
    agg_features_one_follower_df = agg_features_df.filter(col("num_followers") == 1.0)

    playlist_no_one_follower = agg_features_df.filter(col("num_followers") == 1.0).count()
    playlist_no_two_followers = agg_features_df.filter(col("num_followers") == 2.0).count()

    fraction = playlist_no_two_followers / playlist_no_one_follower
    agg_features_sampled_df = agg_features_one_follower_df.sample(fraction=fraction)

    agg_features_df = agg_features_df.filter(col("num_followers") > 1.0)
    agg_features_df = agg_features_df.union(agg_features_sampled_df)

# remove skewness by applying the log function on the column that stores the number of followers
if REMOVE_SKEWENSS:
    agg_features_df = agg_features_df.withColumn("num_followers", log(col("num_followers")))

# split data in train-validation and test folds
train_validation_df, test_df = agg_features_df.randomSplit([0.8, 0.2], seed=SEED)

# compute the performance of the baseline model that predicts the mean number of followers from train-validation set
mean_num_followers = train_validation_df.select(mean("num_followers")).collect()[0][0]

train_validation_df = train_validation_df.withColumn("baseline_prediction", lit(mean_num_followers))
test_df = test_df.withColumn("baseline_prediction", lit(mean_num_followers))

measure_performance(train_validation_df, test_df, baseline_evaluator=True)

# assemble features in a vector per line
features = [col for col in agg_features_df.columns if col not in ["pid", "num_followers"]]
vector_assembler = VectorAssembler(inputCols=features, outputCol="features")

# scale features
standard_scaler = StandardScaler(inputCol="features", outputCol="scaled_features")

# instantiate Random Forest model
rf = RandomForestRegressor(
    featuresCol="scaled_features",
    labelCol="num_followers",
    seed=SEED,
)

# define a sequence of steps that preprocess and learn the data
pipeline = Pipeline(stages=[vector_assembler, standard_scaler, rf])

# define the space of hyperparameters for model optimization
"""
param_grid_rf = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, list(range(64, 129, 32)))
    .addGrid(rf.subsamplingRate, [0.5, 0.7, 1.0])
    .addGrid(rf.minInstancesPerNode, [1] +  list(range(5, 21, 5)))
    .addGrid(rf.featureSubsetStrategy, ["sqrt", "all"])
    .build()
)
"""
param_grid_rf = ParamGridBuilder().addGrid(rf.numTrees, [20]).build()

# define evaluator
evaluator = RegressionEvaluator(
    labelCol="num_followers", predictionCol="prediction", metricName=OPTIMIZATION_METRIC
)

# instantiate the cross-validation operation
cross_validation = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=param_grid_rf,
    evaluator=evaluator,
    numFolds=NUM_FOLDS,
    seed=SEED,
)

# preprocess data and train model
cv_model = cross_validation.fit(train_validation_df)

# get best model and the performance of each model
best_rf = cv_model.bestModel.stages[-1]
scores = cv_model.avgMetrics

# get the hyperparameters of the best model
best_rf_hyperparameters = {
    "subsampling rate": best_rf._java_obj.getSubsamplingRate(),
    "num trees": best_rf._java_obj.getNumTrees(),
    "min instances per node": best_rf._java_obj.getMinInstancesPerNode(),
    "feature subset strategy": best_rf._java_obj.getFeatureSubsetStrategy(),
}

# print the cross-validation score of each trained model
print(f"Cross-validation {OPTIMIZATION_METRIC} scores: {scores}")

# display info about the best model
print("\n***Info about the best model***")
print(f"{OPTIMIZATION_METRIC}: {min(scores)}")
print(f"Hyperparameters:\n{best_rf_hyperparameters}")
print(f"Feature importances:\n{dict(zip(features, best_rf.featureImportances))}")

# evaluate best model on train-validation and test data
train_validation_df = cv_model.transform(train_validation_df)
test_df = cv_model.transform(test_df)
measure_performance(train_validation_df, test_df, baseline_evaluator=False)

# store best model
cv_model.write().overwrite().save(MODEL_PATH)

# store predictions
columns_of_interest = ["pid", "num_followers", "prediction", "baseline_prediction"]

train_validation_df.select(*columns_of_interest).coalesce(1).write.mode(
    "overwrite"
).csv(DATA_STORAGE_PATH + "train_validation/")
test_df.select(*columns_of_interest).coalesce(1).write.mode("overwrite").csv(
    DATA_STORAGE_PATH + "test/"
)
