from video2text_mod.video2text_mod import get_summary
# import pandas as pd
# from tqdm import tqdm
# import pandas as pd
# import multiprocessing
# video_path = "miso_ara_6797294685082619137.mp4"
# full_path = "/home/admin_1/working/bigdata/tiktok_video" + str(video_path)
#     # print(full_path)
# result_object, result_describe = get_summary(full_path)
# print(result_object)
# print(result_describe)
# def videoTtext(job):
#     result = {}
#     index_, df = job
#     video_path = df['file_name'][index_]
#     full_path = "/home/admin_1/working/bigdata/tiktok_video" + str(video_path)
#     # print(full_path)
#     result_object, result_describe = get_summary(full_path)
#     result['object'] = result_object
#     result['summary'] = result_describe
#     df_ = pd.DataFrame([result])
#     return df_

# def worker_function(job_queue, result_queue):
#     while True:
#         job = job_queue.get()
#         if job is None:
#             break
#         result = videoTtext(job)
#         # Put the result in the result queue
#         result_queue.put(result)

# df = pd.read_csv("test.csv",low_memory=False)
# num_processes = 1
# job_queue = multiprocessing.Queue()
# result_queue = multiprocessing.Queue()
# pool = multiprocessing.Pool(processes=num_processes, initializer=worker_function, initargs=(job_queue, result_queue)) 
# df_index = df.index.tolist() # Get the DataFrame index as a list
# jobs = [(sra_index, df) for sra_index in df_index] # Enqueue jobs (each job is a tuple of sra_index and df_sra)
# with tqdm(total=len(jobs), desc="Processing data: ", ncols=70) as pbar:
#     for job in jobs: # add job to queue the job will start but can add job
#         job_queue.put(job)
#     for _ in range(num_processes): # Add sentinel values to signal workers to exit (add last job with None for let process can exit)
#         job_queue.put(None)
#     results = [] # Collect results as they become available
#     for _ in range(len(jobs)): # get result by total job 
#         result = result_queue.get()
#         results.append(result)
#         pbar.update(1) # update process bar
# pool.close()
# pool.join()


# if len(results) > 0:
#     df_result = pd.concat(results, ignore_index=True) 
# else:
#     df_result = pd.DataFrame()
# df_result.to_csv("test_update.csv",index=False)


from pyspark.sql import SparkSession
from pyspark.sql.functions import udf,col
from pyspark.sql.types import StringType
from pyspark.sql.types import StructType, StructField
import pandas as pd
# from pyspark import SparkConf, SparkContext

# In Jupyter you have to stop the current context first
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
    full_path = "/home/admin_1/working/bigdata/tiktok_video/" + str(video_path)
    # print(full_path)
    result_object, result_describe = get_summary(full_path)
    return result_object, result_describe

schema = StructType([
    StructField("object", StringType(), True),
    StructField("summary", StringType(), True)
])

videoTtext_udf = udf(videoTtext, schema)
# Apply the UDF to create a new column
df_result = df.withColumn("result", videoTtext_udf("file_name"))
df_reCol = df_result.withColumn("summary", col("result.object")) \
                   .withColumn("object", col("result.summary"))
df_reCol = df_reCol.drop('result')
# Read CSV into a DataFrame
# df_reCol.show(5, truncate=False)
df_pandas = df_reCol.toPandas()
df_pandas.to_csv("data_set_tiktok_update_2_resut.csv",index=False)

# import pandas as pd

# def videoTtext(row):
#     video_path = row['file_name']
#     full_path = "/home/admin_1/working/bigdata/tiktok_video" + str(video_path)
#     # print(full_path)
#     result_object, result_describe = get_summary(full_path)
#     row['object'] = result_object
#     row['summary'] = result_describe
#     return row

# df = pd.read_csv("data_set_tiktok_update.csv",low_memory=False)
# df = df.apply(videoTtext,axis=1)
# df.to_csv("test_update.csv",index=False)

# print(df)
