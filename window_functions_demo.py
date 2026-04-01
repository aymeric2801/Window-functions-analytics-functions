import os
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import (
    row_number, rank, dense_rank, ntile,
    lag, lead,
    sum as spark_sum, avg, col,
    when, count
)
from datetime import datetime, timedelta
import random

spark = SparkSession.builder \
    .appName("AdvancedWindowFunctions") \
    .master("local[*]") \
    .config("spark.driver.host", "127.0.0.1") \
    .getOrCreate()

data = []

start_date = datetime(2022, 1, 1)

customers = list(range(1, 101))
categories = ["Electronics", "Clothing", "Books", "Home", "Sports", "Beauty"]
regions = ["North", "South", "West", "East"]
payment_methods = ["Card", "PayPal", "Wire"]
statuses = ["Completed", "Cancelled", "Pending"]

for i in range(5000):
    order_date = start_date + timedelta(days=random.randint(0, 730))
    amount = round(random.uniform(10, 2000), 2)

    data.append((
        i + 1,
        order_date,
        random.choice(customers),
        amount,
        random.choice(categories),
        random.choice(regions),
        random.choice(payment_methods),
        random.choice(statuses)
    ))

df = spark.createDataFrame(data, [
    "order_id", "order_date", "customer_id",
    "amount", "category", "region",
    "payment_method", "status"
])

df.orderBy("customer_id", "order_date").show(20, truncate=False)

window_rank = Window.partitionBy("category").orderBy(col("amount").desc())

df.select(
    "category", "amount",
    row_number().over(window_rank).alias("row_number"),
    rank().over(window_rank).alias("rank"),
    dense_rank().over(window_rank).alias("dense_rank")
).show(20)

window_top = Window.partitionBy("category").orderBy(col("amount").desc())

df.withColumn("rn", row_number().over(window_top)) \
  .filter(col("rn") <= 3) \
  .drop("rn") \
  .show()

window_ntile = Window.orderBy(col("amount").desc())

df.select(
    "customer_id", "amount",
    ntile(4).over(window_ntile).alias("quartile")
).show(20)

window_order = Window.partitionBy("customer_id").orderBy("order_date")

df.select(
    "customer_id", "order_date", "amount",
    lag("amount", 1).over(window_order).alias("previous_order"),
    lead("amount", 1).over(window_order).alias("next_order")
).show(20)

window_running = Window.partitionBy("customer_id") \
    .orderBy("order_date") \
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

df.select(
    "customer_id", "order_date", "amount",
    spark_sum("amount").over(window_running).alias("running_total")
).show(20)

daily = df.groupBy("order_date") \
          .agg(spark_sum("amount").alias("daily_total"))

window_moving = Window.orderBy("order_date").rowsBetween(-3, 3)

daily.select(
    "order_date", "daily_total",
    avg("daily_total").over(window_moving).alias("moving_avg_7day")
).show(20)

df_completed = df.withColumn(
    "amount_completed",
    when(col("status") == "Completed", col("amount")).otherwise(0)
)

df_completed.select(
    "customer_id", "order_date", "status", "amount",
    spark_sum("amount_completed").over(window_running).alias("running_completed_total")
).show(20)

window_category_total = Window.partitionBy("category")

df.select(
    "category", "amount",
    (col("amount") / spark_sum("amount").over(window_category_total) * 100)
    .alias("percent_of_category")
).show(20)

df.select(
    "customer_id", "order_date",
    count("*").over(window_running).alias("order_count_so_far")
).show(20)

spark.stop()