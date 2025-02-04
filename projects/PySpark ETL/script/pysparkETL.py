# Filename: pysparkETL.py
# Author: Nimalan Subramanian
# Created: 2025-01-29
# Description: Scalable ETL pipeline using PySpark to process large datasets. ETL process done on temperature dataset and saved to parquet/database

from pyspark.sql import SparkSession

#Initialize Spark Session
spark = SparkSession.builder \
    .appName("ETL Pipeline") \
    .getOrCreate()

#Extract
#load CSV into Spark Data Frame
file_path = "/Users/nimalansubramanian/Downloads/carbon-emissions/temperature.csv"
df = spark.read.csv(file_path, header = True, inferSchema = True)

#display schema and preview data
df.printSchema()
df.show(5)

#Transform
#fill missing values for country codes
df = df.fillna({"ISO2": "Unknown"})

#drop rows with null temperature values
temperature_columns = [col for col in df.columns if col.startswith('F')]
df = df.dropna(subset = temperature_columns, how = "all")

from pyspark.sql.functions import expr

#reshape temperature data to have 'Year' and 'Temperature' columns
df_pivot = df.selectExpr(
    "ObjectId", "Country", "ISO3",
    "stack(62, " + 
    ",".join([f"'F{1961 + i}', F{1961 + i}" for i in range(62)]) + 
    ") as (Year, Temperature)"
)

#convert 'Year' column to integer
df_pivot = df_pivot.withColumn("Year", expr("int(substring(Year, 2, 4))"))
df_pivot.show(5)

#Load
output_path = "/processed_temperature.parquet"
df_pivot.write.mode("overwrite").parquet(output_path)

#load saved parquet file
processed_df = spark.read.parquet(output_path)
processed_df.show(5)