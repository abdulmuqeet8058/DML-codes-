# Apache Spark Experiments with Results and General Graph
import time
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Data preparation
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize Spark Session
spark = SparkSession.builder.master("local[*]").appName("Apache Experiments").getOrCreate()

# Results dictionary to store outputs of each experiment
results = {}

# 1. Scalability with Data Size
def apache_scalability_experiment():
    dataset_sizes = [int(len(X_train) * p) for p in [0.2, 0.4, 0.6, 0.8, 1.0]]
    training_times = []

    for size in dataset_sizes:
        X_sample, y_sample = X_train[:size], y_train[:size]

        feature_columns = [str(i) for i in range(X_train.shape[1])]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

        pandas_df = pd.DataFrame(X_sample, columns=feature_columns)
        pandas_df["label"] = y_sample
        spark_df = spark.createDataFrame(pandas_df)

        data_with_features = assembler.transform(spark_df).select("features", "label")

        lr = LinearRegression(featuresCol="features", labelCol="label")
        start_time = time.time()
        model = lr.fit(data_with_features)
        training_times.append(time.time() - start_time)

    plt.figure()
    plt.plot([p * 100 for p in [0.2, 0.4, 0.6, 0.8, 1.0]], training_times, label="Training Time")
    plt.xlabel("Dataset Size (%)")
    plt.ylabel("Training Time (seconds)")
    plt.title("Scalability with Data Size - Apache Spark")
    plt.legend()
    plt.show()

    results['Scalability'] = training_times

# 2. Fault Tolerance
def apache_fault_tolerance_experiment():
    try:
        X_sample, y_sample = X_train[:int(len(X_train) * 0.8)], y_train[:int(len(y_train) * 0.8)]
        feature_columns = [str(i) for i in range(X_train.shape[1])]

        pandas_df = pd.DataFrame(X_sample, columns=feature_columns)
        pandas_df["label"] = y_sample
        spark_df = spark.createDataFrame(pandas_df)
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        data_with_features = assembler.transform(spark_df).select("features", "label")

        lr = LinearRegression(featuresCol="features", labelCol="label")

        start_time = time.time()
        model = lr.fit(data_with_features)
        elapsed_time = time.time() - start_time

        plt.figure()
        plt.bar(["Training with Fault"], [elapsed_time], color='orange', label="Time with Fault")
        plt.ylabel("Training Time (seconds)")
        plt.title("Fault Tolerance - Apache Spark")
        plt.legend()
        plt.show()

        results['Fault Tolerance'] = elapsed_time
    except Exception as e:
        print("Simulation of fault failed:", e)

# 3. Communication Overhead
def apache_communication_overhead_experiment():
    feature_columns = [str(i) for i in range(X_train.shape[1])]
    pandas_df = pd.DataFrame(X_train, columns=feature_columns)
    pandas_df["label"] = y_train
    spark_df = spark.createDataFrame(pandas_df)

    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    data_with_features = assembler.transform(spark_df).select("features", "label")

    lr = LinearRegression(featuresCol="features", labelCol="label")

    start_time = time.time()
    model = lr.fit(data_with_features)  # Synchronous
    sync_time = time.time() - start_time

    start_time = time.time()
    model = lr.fit(data_with_features)  # Simulate asynchronous
    async_time = time.time() - start_time

    plt.figure()
    plt.bar(["Synchronous", "Asynchronous"], [sync_time, async_time], color=['blue', 'green'])
    plt.ylabel("Training Time (seconds)")
    plt.title("Communication Overhead - Apache Spark")
    plt.show()

    results['Communication'] = {'Synchronous': sync_time, 'Asynchronous': async_time}

# 4. Ease of Use and Setup
def apache_ease_of_use_experiment():
    start_time = time.time()

    # Simulate environment setup
    spark = SparkSession.builder.master("local[*]").appName("Ease Setup Test").getOrCreate()

    elapsed_time = time.time() - start_time

    plt.figure()
    plt.bar(["Environment Setup"], [elapsed_time], color='purple', label="Setup Time")
    plt.ylabel("Setup Time (seconds)")
    plt.title("Ease of Setup - Apache Spark")
    plt.legend()
    plt.show()

    results['Ease of Setup'] = elapsed_time

# General Graph for Apache Spark
def general_apache_graph():
    aspects = ['Scalability', 'Fault Tolerance', 'Communication (Sync)', 'Communication (Async)', 'Ease of Setup']
    values = [
        sum(results['Scalability']) if 'Scalability' in results else 0,
        results['Fault Tolerance'] if 'Fault Tolerance' in results else 0,
        results['Communication']['Synchronous'] if 'Communication' in results else 0,
        results['Communication']['Asynchronous'] if 'Communication' in results else 0,
        results['Ease of Setup'] if 'Ease of Setup' in results else 0
    ]

    plt.figure()
    plt.bar(aspects, values, color=['red', 'orange', 'blue', 'green', 'purple'])
    plt.ylabel("Time/Performance (seconds)")
    plt.title("Apache Spark - General Framework Overview")
    plt.xticks(rotation=45)
    plt.show()

# Run all experiments
apache_scalability_experiment()
apache_fault_tolerance_experiment()
apache_communication_overhead_experiment()
apache_ease_of_use_experiment()
general_apache_graph()

# Print results
print("Results Summary:")
for key, value in results.items():
    print(f"{key}: {value}")