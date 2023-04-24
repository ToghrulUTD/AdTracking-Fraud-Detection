from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import argparse

def main(args):
    spark = SparkSession.builder \
        .appName("Training script for TalkingData AdTracking Fraud Detection using PySpark") \
        .getOrCreate()

    # Read processed data
    processed_data = spark.read.parquet(args.processed_data_input)

    # Feature columns
    feature_columns = ['app', 'device', 'os', 'channel', "IP_first_click", "IP_time_since_last_click", "IP_click_count", "IP_download_ratio"]

    # Prepare features
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    processed_data = assembler.transform(processed_data)

    # Train-test split
    train_data, test_data = processed_data.randomSplit([0.7, 0.3], seed=42)

    # Train the model
    rf = RandomForestClassifier(featuresCol="features", labelCol="is_attributed", numTrees=100, seed=42)
    model = rf.fit(train_data)

    # Model evaluation
    predictions = model.transform(test_data)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="is_attributed", metricName="areaUnderROC")
    auc = evaluator.evaluate(predictions)

    print(f"AUC: {auc}")

    # Save the model
    model.write().overwrite().save(args.model_output_path)

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training script for TalkingData AdTracking Fraud Detection using PySpark')
    parser.add_argument('--processed_data_input', type=str, required=True, help='Input file path for processed data')
    parser.add_argument('--model_output_path', type=str, required=True, help='Output path for the trained model')

    args = parser.parse_args()
    main(args)
