# TensorFlow Experiments with Correct Fault Tolerance and General Graph
import time
import matplotlib.pyplot as plt
import tensorflow as tf
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

# Results dictionary to store outputs of each experiment
results = {}

# 1. Scalability with Data Size
def tensorflow_scalability_experiment():
    dataset_sizes = [int(len(X_train) * p) for p in [0.2, 0.4, 0.6, 0.8, 1.0]]
    training_times = []

    for size in dataset_sizes:
        X_sample, y_sample = X_train[:size], y_train[:size]

        # Build and compile the model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")

        # Measure training time
        start_time = time.time()
        model.fit(X_sample, y_sample, epochs=5, batch_size=32, verbose=0)
        training_times.append(time.time() - start_time)

    plt.figure()
    plt.plot([p * 100 for p in [0.2, 0.4, 0.6, 0.8, 1.0]], training_times, label="Training Time")
    plt.xlabel("Dataset Size (%)")
    plt.ylabel("Training Time (seconds)")
    plt.title("Scalability with Data Size - TensorFlow")
    plt.legend()
    plt.show()

    results['Scalability'] = training_times

# 2. Fault Tolerance
# 2. Fault Tolerance
def tensorflow_fault_tolerance_experiment():
    try:
        # Subset of data for simulation
        X_sample, y_sample = X_train[:int(len(X_train) * 0.8)], y_train[:int(len(y_train) * 0.8)]

        # Model creation function to ensure consistent model rebuild
        def create_model():
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(X_train.shape[1],)),
                tf.keras.layers.Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse")
            return model

        # Create and compile the model
        model = create_model()

        checkpoint_path = "/content/fault_tolerance_checkpoint.weights.h5"  # Ensure file ends with .weights.h5
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

        # Initial training (simulate fault midway)
        print("Starting initial training...")
        start_time = time.time()
        model.fit(X_sample, y_sample, epochs=2, batch_size=32, callbacks=[checkpoint], verbose=1)
        training_time_part1 = time.time() - start_time

        # Simulate fault (halt and reinitialize model)
        print("Simulating fault: Resetting model...")
        model = create_model()  # Reinitialize the model

        # Load weights from the checkpoint
        print("Loading weights from checkpoint...")
        model.load_weights(checkpoint_path)

        # Resume training after recovering weights
        start_time = time.time()
        model.fit(X_sample, y_sample, epochs=3, batch_size=32, verbose=1)
        training_time_part2 = time.time() - start_time

        # Total training time
        total_time = training_time_part1 + training_time_part2

        # Visualization
        plt.figure()
        plt.bar(["Training with Fault"], [total_time], color='orange', label="Total Time")
        plt.ylabel("Training Time (seconds)")
        plt.title("Fault Tolerance - TensorFlow")
        plt.legend()
        plt.show()

        results['Fault Tolerance'] = total_time
    except Exception as e:
        print("Simulation of fault failed:", e)


# 3. Communication Overhead
def tensorflow_communication_overhead_experiment():
    X_sample, y_sample = X_train[:int(len(X_train) * 0.8)], y_train[:int(len(y_train) * 0.8)]

    # Synchronous training
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    start_time = time.time()
    model.fit(X_sample, y_sample, epochs=5, batch_size=32, verbose=0)
    sync_time = time.time() - start_time

    # Simulated asynchronous training
    start_time = time.time()
    model.fit(X_sample, y_sample, epochs=5, batch_size=64, verbose=0)  # Simulated with larger batch size
    async_time = time.time() - start_time

    plt.figure()
    plt.bar(["Synchronous", "Asynchronous"], [sync_time, async_time], color=['blue', 'green'])
    plt.ylabel("Training Time (seconds)")
    plt.title("Communication Overhead - TensorFlow")
    plt.show()

    results['Communication'] = {'Synchronous': sync_time, 'Asynchronous': async_time}

# 4. Ease of Use and Setup
def tensorflow_ease_of_use_experiment():
    start_time = time.time()

    # Simulate environment setup (loading TensorFlow and model definition)
    tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1)
    ])

    elapsed_time = time.time() - start_time

    plt.figure()
    plt.bar(["Environment Setup"], [elapsed_time], color='purple', label="Setup Time")
    plt.ylabel("Setup Time (seconds)")
    plt.title("Ease of Setup - TensorFlow")
    plt.legend()
    plt.show()

    results['Ease of Setup'] = elapsed_time

# General Graph for TensorFlow
def general_tensorflow_graph():
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
    plt.title("TensorFlow - General Framework Overview")
    plt.xticks(rotation=45)
    plt.show()

# Run all experiments
tensorflow_scalability_experiment()
tensorflow_fault_tolerance_experiment()
tensorflow_communication_overhead_experiment()
tensorflow_ease_of_use_experiment()
general_tensorflow_graph()

# Print results
print("Results Summary:")
for key, value in results.items():
    print(f"{key}: {value}")
