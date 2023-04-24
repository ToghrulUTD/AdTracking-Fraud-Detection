from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, max, min, when, lag, mean, unix_timestamp
from pyspark.sql.window import Window
import argparse

def main(args):
    spark = SparkSession.builder \
        .appName("ETL script for TalkingData AdTracking Fraud Detection using PySpark") \
        .getOrCreate()

    # Read raw data
    raw_data = spark.read.csv(args.input_file, header=True, inferSchema=True)

    # Feature engineering
    window_ip = Window.partitionBy("ip").orderBy('click_time')

    data = raw_data.withColumn("IP_first_click", when(min("click_time").over(window_ip) == col("click_time"), 1).otherwise(0)) \
                .withColumn("IP_click_count", count("*").over(window_ip)) \
                .withColumn("IP_download_count", sum("is_attributed").over(window_ip)) \
                .withColumn("IP_download_ratio", col("IP_download_count") / col("IP_click_count")) \
                .withColumn("last_click_time", lag("click_time", 1).over(window_ip)) \
                .withColumn("IP_time_since_last_click", when(col("last_click_time").isNull(), -1).otherwise((unix_timestamp(col("click_time")) - unix_timestamp(col("last_click_time")))))

    # Calculate IP address statistics
    ip_stats = data.groupBy('ip').agg(
        count('*').alias('IP_click_count'),
        sum('is_attributed').alias('IP_download_count'),
        (sum('is_attributed') / count('*')).alias('IP_download_ratio'),
        max('click_time').alias('last_click_time')
    )

    # Save the processed data
    data.write.parquet(args.processed_data_output)

    # Save the IP address statistics
    ip_stats.write.parquet(args.ip_stats_output)

    spark.stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ETL script for TalkingData AdTracking Fraud Detection using PySpark')
    parser.add_argument('--input_file', type=str, required=True, help='Input file path for raw data')
    parser.add_argument('--processed_data_output', type=str, required=True, help='Output file path for processed data')
    parser.add_argument('--ip_stats_output', type=str, required=True, help='Output file path for IP address statistics')

    args = parser.parse_args()
    main(args)
