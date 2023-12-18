from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder \
    .appName("Local Spark Session") \
    .master("local[*]") \
    .getOrCreate()

# Sample data file path (Replace with your actual file path)
sample_data_path = "/Users/jonathanschlosser/Desktop/Code_Snippets/PySpark/customers-10000.csv"

# Read CSV data into a DataFrame
df = spark.read.csv(sample_data_path, header=True, inferSchema=True)

# Show the first few records (optional, for verification)
df.show()

# # Describe the DataFrame to get basic statistics for numeric columns
# statistics = df.describe()
# statistics.show()

# # Stop the Spark session when done
# spark.stop()
