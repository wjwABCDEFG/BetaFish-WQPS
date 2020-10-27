# from pyspark.ml.linalg import Vector, Vectors
# from pyspark.sql import Row, functions
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# from pyspark.ml import Pipeline,PipelineModel
# from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer, HashingTF, Tokenizer
# from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel, BinaryLogisticRegressionSummary, LogisticRegression
# from pyspark import SparkConf, SparkContext
# from pyspark.sql.session import SparkSession
# from pyspark.ml.regression import DecisionTreeRegressor,DecisionTreeRegressionModel
# from pyspark.ml.evaluation import RegressionEvaluator
# import pandas as pd
# import numpy as np
# sc = SparkContext('local')
# spark = SparkSession(sc)
# def f(x):
#     rel = {}
#     rel['features'] = Vectors.dense(x[:15])
#     rel['label'] = x[15]
#     return rel
# def f_predict(x):
#     rel = {}
#     rel['features'] = Vectors.dense(x[:15])
#
#     return rel
#
# def train(training_set):
#     data_map = list(map(f, training_set))
#     data = pd.DataFrame(data_map)
#     df_data = spark.createDataFrame(data)
#     DecissionTree = DecisionTreeRegressor()
#     model = DecissionTree.fit(df_data)
#     return model
# def predict(model,x):
#     result = []
#     data_map = list(map(f_predict, x))
#     data = pd.DataFrame(data_map)
#     df_data = spark.createDataFrame(data)
#     pred_result = model.transform(df_data)
#     selected = pred_result.select( "prediction")
#     for row in selected.collect():
#         result.append(row)
#
#     return np.array(result).reshape((len(result)))
#
#
