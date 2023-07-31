#Getting Started with PySpark

# Installation
# You can install PySpark using:
# pip install pyspark

# Initialize PySpark
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Getting Started with PySpark") \
    .getOrCreate()

# Generating Some Data
from pyspark.sql import Row

data = [("John", "Doe", 30), ("Jane", "Doe", 25), ("Michael", "Smith", 40)]
columns = ["First Name", "Last Name", "Age"]

# Create DataFrame
df = spark.createDataFrame(data, columns)

df.show()