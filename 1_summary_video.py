from video2text_mod.video2text_mod import get_summary
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf,col
from pyspark.sql.types import StringType
from pyspark.sql.types import StructType, StructField
import pandas as pd

# set video path
path =""

def tiktok_summary():
    spark = SparkSession.builder \
        .appName("video2text") \
        .config("spark.executor.instances", "2") \
        .config("spark.executor.cores", "28") \
        .config("spark.executor.memory", "128g") \
        .config("spark.driver.memory", "8g") \
        .config("spark.task.cpus", "14") \
        .getOrCreate()

    # Set maximum partition size to 64 GB
    spark.conf.set("spark.sql.files.maxPartitionBytes", "64g")

    data = pd.read_csv("data_set_tiktok_update_2.csv",low_memory=False)
    print(data.info())
    df= spark.createDataFrame(data)

    def videoTtext(video_path):
        full_path = path + str(video_path)
        result_object, result_describe = get_summary(full_path)
        return result_object, result_describe

    schema = StructType([
        StructField("object", StringType(), True),
        StructField("summary", StringType(), True)
    ])

    videoTtext_udf = udf(videoTtext, schema)
    df_result = df.withColumn("result", videoTtext_udf("file_name"))
    df_reCol = df_result.withColumn("summary", col("result.object")) \
                    .withColumn("object", col("result.summary"))
    df_reCol = df_reCol.drop('result')
    # Read CSV into a DataFrame
    # df_reCol.show(5, truncate=False)
    df_pandas = df_reCol.toPandas()
    df_pandas.to_csv("data_set_tiktok_update_2_resut.csv",index=False)

    if __name__ == "__main__":
        tiktok_summary()