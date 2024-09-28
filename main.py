from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, pow, current_date
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.functions import vector_to_array

spark = SparkSession.builder \
    .appName("Heart Disease Data Preprocessing") \
    .getOrCreate()
    

# 1. Data Loading
data_path = r"data\heart_disease_data.csv"  
df = spark.read.csv(data_path, header=True, inferSchema=True)

# 2. One-Hot Encoding
indexer = StringIndexer(inputCol="cp", outputCol="cp_index")
df = indexer.fit(df).transform(df)

# Apply One-Hot Encoding
encoder = OneHotEncoder(inputCols=["cp_index"], outputCols=["cp_ohe"])
df = encoder.fit(df).transform(df)

# Convert the one-hot encoded vector (cp_ohe) into separate columns
df = df.withColumn("cp_ohe_array", vector_to_array(col("cp_ohe")))
df = df.withColumn("cp_ohe_0", col("cp_ohe_array")[0]) \
       .withColumn("cp_ohe_1", col("cp_ohe_array")[1]) \
       .withColumn("cp_ohe_2", col("cp_ohe_array")[2]) \
       .withColumn("cp_ohe_3", col("cp_ohe_array")[3])

# 3. Feature Derivation
df = df.withColumn("powerOfTrestbps", pow(col("trestbps"), 2))

# 4. Data Filtering
filtered_df = df.filter((col("age") > 50) & (col("trestbps") > 140))

# 5. Quantization of Cholesterol Levels
filtered_df = filtered_df.withColumn(
    "cholesterol_level",
    when(col("chol") < 200, "Low")
    .when((col("chol") >= 200) & (col("chol") <= 239), "Medium")
    .otherwise("High")
)

# 6. Data Reduction
high_cholesterol_count = filtered_df.filter(col("cholesterol_level") == "High").count()
print(f"Number of patients with High cholesterol level: {high_cholesterol_count}")

# 7. Add 'Report Date' column with the current date
filtered_df = filtered_df.withColumn("Report Date", current_date())

# 8. Data Export
# Select the necessary columns for output, including split one-hot encoded columns
output_df = filtered_df.select(
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "target", "Report Date",
    "cp_ohe_0", "cp_ohe_1", "cp_ohe_2", "cp_ohe_3", "cholesterol_level", "powerOfTrestbps"
)

output_path = r"output\processed_heart_disease_data.csv"  
output_df.write.csv(output_path, header=True)

# Stop the Spark session
spark.stop()
