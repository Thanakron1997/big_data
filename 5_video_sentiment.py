import sys
import pandas as pd  
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import Word2VecModel
from pyspark.ml.classification import MultilayerPerceptronClassificationModel
from pyspark.sql.functions import  isnull, when, count

def video_sentiment(model_name,file_name):
    '''get sentiment from video'''
    spark = SparkSession.builder \
        .appName("video_sentiment") \
        .config("spark.executor.memory", "10g") \
        .config("spark.driver.memory", "10g") \
        .config("spark.sql.files.maxPartitionBytes", "16g") \
        .config("spark.dynamicAllocation.enabled", "false") \
        .getOrCreate()

    df2 = pd.read_csv(file_name,low_memory=False)
    cleaned_df = spark.createDataFrame(df2)
    cleaned_df.show()
    cleaned_df = cleaned_df.withColumn('token', split(cleaned_df['clean'], ' '))
    cleaned_df.select([count(when(isnull(c), c)).alias(c) for c in cleaned_df.columns]).show()
    model = Word2VecModel.load("word2Vec_model")
    data_vec = model.transform(cleaned_df)
    data_vec.select([count(when(isnull(c), c)).alias(c) for c in data_vec.columns]).show()
    model2 = MultilayerPerceptronClassificationModel.load(model_name)
    predictions = model2.transform(data_vec)
    predictions.show()
    df_pandas = predictions.toPandas()
    df_pandas.to_csv("data_video_predict.csv",index=False)

if __name__ == "__main__": 

    if len(sys.argv) < 2:
        video_sentiment("mlp_layers100_200_3_ac_0.8505766897404547","video_clean.csv")
    else:
        model_name = sys.argv[1]
        file_name = sys.argv[2]
        video_sentiment(model_name,file_name)