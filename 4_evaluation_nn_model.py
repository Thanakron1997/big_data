
from pyspark.sql import SparkSession
from pyspark.sql.types import  FloatType
from pyspark.sql.functions import *
from pyspark.ml.classification import MultilayerPerceptronClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import json
import glob

def evaluation_model():
    '''get confusionMatrix, hammingLoss, logLoss, precision, recall and f1 from model'''
    seed = 777
    split_data = [0.7, 0.3]
    spark = SparkSession.builder \
        .appName("evaluation-model") \
        .config("spark.executor.memory", "10g") \
        .config("spark.driver.memory", "10g") \
        .config("spark.sql.files.maxPartitionBytes", "16g") \
        .config("spark.dynamicAllocation.enabled", "false") \
        .getOrCreate()

    model_df = spark.read.parquet('all_vector_dataset.parquet')
    train, test = model_df.randomSplit(split_data, seed=seed)

    lst_model = glob.glob('mlp_layers*')
    for model_name in lst_model:
        data_result = {}
        model2 = MultilayerPerceptronClassificationModel.load(model_name)
        predictions = model2.transform(test)

        preds_and_labels = predictions.select(['prediction','label'])\
                                    .withColumn('label', col('label')\
                                    .cast(FloatType()))\
                                    .orderBy('prediction')

        metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))
        cm = metrics.confusionMatrix().toArray()
        layer = model2.getLayers()
        evaluator = MulticlassClassificationEvaluator(labelCol = 'label', predictionCol = 'prediction', metricName = 'hammingLoss')
        hammingLoss = evaluator.evaluate(predictions)
        evaluator_log = MulticlassClassificationEvaluator(labelCol = 'label', predictionCol = 'prediction', metricName = 'logLoss')
        logLoss = evaluator_log.evaluate(predictions)
        precision_positive =metrics.precision(0.0)
        recall_positive =metrics.recall(0.0)
        f1_positve = 2*((precision_positive*recall_positive))/(precision_positive+recall_positive)
        precision_negative =metrics.precision(1.0)
        recall_negative =metrics.recall(1.0)
        f1_negative = 2*((precision_negative*recall_negative))/(precision_negative+recall_negative)
        precision_normal =metrics.precision(2.0)
        recall_normal =metrics.recall(2.0)
        f1_normal = 2*((precision_normal*recall_normal))/(precision_normal+recall_normal)

        data_result['name'] = str(model_name)
        data_result['confusionMatrix'] = str(cm)
        data_result['layer'] = str(layer)
        data_result['hammingLoss'] = str(hammingLoss)
        data_result['logLoss'] = str(logLoss)
        data_result['precision_positive'] = str(precision_positive)
        data_result['recall_positive'] = str(recall_positive)
        data_result['f1_positve'] = str(f1_positve)
        data_result['precision_negative'] = str(precision_negative)
        data_result['recall_negative'] = str(recall_negative)
        data_result['f1_negative'] = str(f1_negative)
        data_result['precision_normal'] = str(precision_normal)
        data_result['recall_normal'] = str(recall_normal)
        data_result['f1_normal'] = str(f1_normal)
        
        file_json = "result" + str(model_name) + ".json"
        try:
            json.dump(data_result, open( file_json, 'w' ) )
        except:
            f = open(file_json,'w')
            f.write(str(data_result))
            f.close()
    spark.stop()

if __name__ == "__main__":
    evaluation_model()