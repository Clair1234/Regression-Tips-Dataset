from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

spark=SparkSession.builder.appName('LinearRegression').getOrCreate()

file_location = "tips.csv"
file_type = "csv"

df = spark.read.csv(file_location, header=True, inferSchema=True)

### What data are we working with
df.printSchema()

### Convert categorical features
indexer = StringIndexer(
    inputCols=["sex", "smoker", "day", "time"], 
    outputCols=["sex_indexed", "smoker_indexed", "day_indexed", "time_indexed"])

df_indexed = indexer.fit(df).transform(df)
df_indexed.show(5)

### Create Independent feature vector
featureassembler = VectorAssembler(inputCols=
    ['tip', 'size', 'sex_indexed', 'smoker_indexed', 'day_indexed', 
    'time_indexed'], outputCol="Independent Features")

output = featureassembler.transform(df_indexed)

finalized_data = output.select("Independent Features", "total_bill")
finalized_data.show(5)

### Train Test Split
train_data, test_data = finalized_data.randomSplit([0.75, 0.25])
regressor = LinearRegression(featuresCol='Independent Features', labelCol='total_bill')
regressor = regressor.fit(train_data)

regressor.coefficients

### Predictions
pred_results = regressor.evaluate(test_data)

### Final comparaison
pred_results.predictions.show(5)

### Stop session
spark.stop()