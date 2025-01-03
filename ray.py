# Ray Experiments with Results and General Graph
import ray
import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

# Initialize Ray
ray.shutdown()
ray.init(ignore_reinit_error=True)

# Data preparation
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Results dictionary to store outputs of each experiment
results = {}

# 1. Scalability with Data Size
def ray_scalability_experiment():
    dataset_sizes = [int(len(X_train) * p) for p in [0.2, 0.4, 0.6, 0.8, 1.0]]
    training_times = []

    @ray.remote
    def train_model(X_sample, y_sample):
        model = LinearRegression()
        model.fit(X_sample, y_sample)

    for size in dataset_sizes:
        X_sample, y_sample = X_train[:size], y_train[:size]

        start_time = time.time()
        ray.get(train_model.remote(X_sample, y_sample))
        training_times.append(time.time() - start_time)

    plt.figure()
    plt.plot([p * 100 for p in [0.2, 0.4, 0.6, 0.8, 1.0]], training_times, label="Training Time")
    plt.xlabel("Dataset Size (%)")
    plt.ylabel("Training Time (seconds)")
    plt.title("Scalability with Data Size - Ray")
    plt.legend()
    plt.show()

    results['Scalability'] = training_times

# 2. Fault Tolerance
def ray_fault_tolerance_experiment():
    @ray.remote
    def train_model_with_fault(X_sample, y_sample):
        model = LinearRegression()
        model.fit(X_sample, y_sample)

    try:
        X_sample, y_sample = X_train[:int(len(X_train) * 0.8)], y_train[:int(len(y_train) * 0.8)]

        # First part of training
        print("Starting initial training...")
        start_time = time.time()
        ray.get(train_model_with_fault.remote(X_sample, y_sample))
        training_time_part1 = time.time() - start_time

        # Simulate a fault and retry training
        print("Simulating fault and restarting training...")
        start_time = time.time()
        ray.get(train_model_with_fault.remote(X_sample, y_sample))
        training_time_part2 = time.time() - start_time

        total_time = training_time_part1 + training_time_part2

        plt.figure()
        plt.bar(["Training with Fault"], [total_time], color='orange', label="Total Time")
        plt.ylabel("Training Time (seconds)")
        plt.title("Fault Tolerance - Ray")
        plt.legend()
        plt.show()

        results['Fault Tolerance'] = total_time
    except Exception as e:
        print("Simulation of fault failed:", e)

# 3. Communication Overhead
def ray_communication_overhead_experiment():
    @ray.remote
    def synchronous_training(X_sample, y_sample):
        model = LinearRegression()
        model.fit(X_sample, y_sample)

    @ray.remote
    def asynchronous_training(X_sample, y_sample):
        model = LinearRegression()
        model.fit(X_sample, y_sample)

    X_sample, y_sample = X_train[:int(len(X_train) * 0.8)], y_train[:int(len(y_train) * 0.8)]

    # Measure synchronous time
    start_time = time.time()
    ray.get(synchronous_training.remote(X_sample, y_sample))
    sync_time = time.time() - start_time

    # Measure asynchronous time (parallel execution)
    start_time = time.time()
    futures = [asynchronous_training.remote(X_sample, y_sample) for _ in range(4)]
    ray.get(futures)
    async_time = time.time() - start_time

    plt.figure()
    plt.bar(["Synchronous", "Asynchronous"], [sync_time, async_time], color=['blue', 'green'])
    plt.ylabel("Training Time (seconds)")
    plt.title("Communication Overhead - Ray")
    plt.show()

    results['Communication'] = {'Synchronous': sync_time, 'Asynchronous': async_time}

# 4. Ease of Use and Setup
def ray_ease_of_use_experiment():
    start_time = time.time()

    # Simulate Ray environment setup
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    elapsed_time = time.time() - start_time

    plt.figure()
    plt.bar(["Environment Setup"], [elapsed_time], color='purple', label="Setup Time")
    plt.ylabel("Setup Time (seconds)")
    plt.title("Ease of Setup - Ray")
    plt.legend()
    plt.show()

    results['Ease of Setup'] = elapsed_time

# General Graph for Ray
def general_ray_graph():
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
    plt.title("Ray - General Framework Overview")
    plt.xticks(rotation=45)
    plt.show()

# Run all experiments
ray_scalability_experiment()
ray_fault_tolerance_experiment()
ray_communication_overhead_experiment()
ray_ease_of_use_experiment()
general_ray_graph()

# Print results
print("Results Summary:")
for key, value in results.items():
    print(f"{key}: {value}")
