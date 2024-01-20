# import SparkSession and functions for data preparation and modeling
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# instantiate the SparkSession class
spark = SparkSession.builder.getOrCreate()

# define seed value for reproducible results and other parameters
SEED = 0
METRIC_NAME = "rmse"
NUM_FOLDS = 3
DATA_SPLIT_FRACTIONS = [0.8, 0.2]
MODEL_PATH = "/user/s3264424/project_group_18/random_forest_regressor/"


# load aggregated features
agg_features_df = spark.read.csv(
    "/user/s3307891/all_aggregated_features", header="true", inferSchema="true"
).distinct()

# split data in train-validation and test folds
train_validation_df, test_df = agg_features_df.randomSplit(DATA_SPLIT_FRACTIONS, seed=SEED)

# assemble features in a vector per line
features = [
    col for col in agg_features_df.columns if col not in ["pid", "num_followers"]
]
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
param_grid_rf = (
    ParamGridBuilder()
    .addGrid(rf.numTrees, list(range(64, 129, 16)))
    .addGrid(rf.subsamplingRate, [0.5, 0.9, 1.0])
    .addGrid(rf.minInstancesPerNode, [1] +  list(range(5, 21, 5)))
    .addGird(rf.featureSubsetStrategy, ["sqrt", "log2", "exhaustiveSearch"])
    .build()
)

# define evaluator
evaluator = RegressionEvaluator(
    labelCol="num_followers", predictionCol="prediction", metricName=METRIC_NAME
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

# evaluate best model on train-validation and test data
rmse_train_validation = evaluator.evaluate(cv_model.transform(train_validation_df))
rmse_test = evaluator.evaluate(cv_model.transform(test_df))

# print the cross-validation score of each trained model
print(f"Cross-validation {METRIC_NAME} scores: {scores}")

# display info about the best model
print("\n***Info about the best model***")
print(f"{METRIC_NAME}: {min(scores)}")
print(f"Hyperparameters:\n{best_rf_hyperparameters}")
print(f"Feature importances:\n{dict(zip(features, best_rf.featureImportances))}")
print(f"Train-validation {METRIC_NAME}: {rmse_train_validation}")
print(f"Test {METRIC_NAME}: {rmse_test}")

# store best model
cv_model.write().overwrite().save(MODEL_PATH)
