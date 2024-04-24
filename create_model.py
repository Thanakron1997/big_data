from pyspark.sql import SparkSession
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import  isnull, when, count,split
from pyspark.ml.feature import  StringIndexer, Word2Vec
from pyspark.sql.types import StructType, StructField, StringType
import json 

# Parameters for Word2Vec model
vector_size = 100
window_size = 15
min_count = 1
seed = 777
MaxIter = 1
#
split_data = [0.7, 0.3]
# set layer in nn model
layers1 = [100,200,3]
layers2 = [100,200,200,3]
layers3 = [100,200,200,200,3]
mmIter = 1000

def create_nn_model():
    '''create three nn model with layers = [100,200,3], [100,200,200,3], [100,200,200,200,3]'''

    spark = SparkSession.builder \
        .appName("create_nn_model") \
        .config("spark.executor.memory", "10g") \
        .config("spark.driver.memory", "10g") \
        .config("spark.sql.files.maxPartitionBytes", "16g") \
        .config("spark.dynamicAllocation.enabled", "false") \
        .getOrCreate()

    custom_schema = StructType([
        StructField("clean",  StringType(), False),
        StructField("category", StringType(), False),
    ])

    cleaned_df = spark.read.schema(custom_schema).options(header='True', delimiter=",").csv('clean_data_2.csv')
    cleaned_df.printSchema()
    cleaned_df.show(2)
    cleaned_df = cleaned_df.withColumn('token', split(cleaned_df['clean'], ' '))

    # create vector by word2Vec
    word2Vec = Word2Vec(vectorSize=vector_size,  windowSize=window_size, minCount=min_count,seed=seed, inputCol="token", outputCol="model")
    word2Vec.setMaxIter(MaxIter)
    modelVec = word2Vec.fit(cleaned_df)
    modelVec.save("word2Vec_model")
    data_vec = modelVec.transform(cleaned_df)

    # set label to index label 
    model_df = data_vec.select(['model','category'])
    model_df.select([count(when(isnull(c), c)).alias(c) for c in model_df.columns]).show()
    indexer = StringIndexer(inputCol = 'category', outputCol = 'label',stringOrderType="frequencyDesc")
    indexer_model = indexer.fit(model_df)
    model_df = indexer_model.transform(model_df)
    indexer_model.save("indexer_model")

    # save vector data 
    try:
        model_df.write.save("all_vector_dataset.parquet")
    except:
        pass
    model_df = model_df.select(['model','label'])
    model_df.select([count(when(isnull(c), c)).alias(c) for c in model_df.columns]).show()

    # split data
    train, test = model_df.randomSplit(split_data, seed=seed)
    train.select([count(when(isnull(c), c)).alias(c) for c in train.columns]).show()

    # model 1
    acc_result = {}
    mlp = MultilayerPerceptronClassifier(featuresCol='model',labelCol='label',layers = layers1, seed = seed,maxIter=mmIter).fit(train)
    df_test = mlp.transform(test)
    evaluator = MulticlassClassificationEvaluator(labelCol = 'label', predictionCol = 'prediction', metricName = 'accuracy')
    mlpacc = evaluator.evaluate(df_test)
    model_name = "mlp_layers" + "_".join(str(layer) for layer in layers1) + "_ac_" +str(mlpacc)
    acc_result[model_name] = str(mlpacc)
    mlp.save(model_name)

    # model 2
    mlp = MultilayerPerceptronClassifier(featuresCol='model',labelCol='label',layers = layers2, seed = seed,maxIter=mmIter).fit(train)
    df_test = mlp.transform(test)
    evaluator = MulticlassClassificationEvaluator(labelCol = 'label', predictionCol = 'prediction', metricName = 'accuracy')
    mlpacc = evaluator.evaluate(df_test)
    model_name = "mlp_layers" + "_".join(str(layer) for layer in layers2) + "_ac_" +str(mlpacc)
    acc_result[model_name] = str(mlpacc)
    mlp.save(model_name)

    # model 3
    mlp = MultilayerPerceptronClassifier(featuresCol='model',labelCol='label',layers = layers3, seed = seed,maxIter=mmIter).fit(train)
    df_test = mlp.transform(test)
    evaluator = MulticlassClassificationEvaluator(labelCol = 'label', predictionCol = 'prediction', metricName = 'accuracy')
    mlpacc = evaluator.evaluate(df_test)
    model_name = "mlp_layers" + "_".join(str(layer) for layer in layers3) + "_ac_" +str(mlpacc)
    acc_result[model_name] = str(mlpacc)
    mlp.save(model_name)

    with open("Accuracy_model.json", "w") as outfile: 
        json.dump(acc_result, outfile)
    model_df.write.option("header",True).csv("result_file_x.csv")
    spark.stop()

if __name__ == "__main__":
    create_nn_model()