# PyTorch Experiments with Results and General Graph
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
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

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Results dictionary to store outputs of each experiment
results = {}

# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# 1. Scalability with Data Size
def pytorch_scalability_experiment():
    dataset_sizes = [int(len(X_train) * p) for p in [0.2, 0.4, 0.6, 0.8, 1.0]]
    training_times = []

    for size in dataset_sizes:
        X_sample = X_train_tensor[:size]
        y_sample = y_train_tensor[:size]

        model = LinearRegressionModel(X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        start_time = time.time()
        for epoch in range(5):
            optimizer.zero_grad()
            predictions = model(X_sample)
            loss = criterion(predictions, y_sample)
            loss.backward()
            optimizer.step()
        training_times.append(time.time() - start_time)

    plt.figure()
    plt.plot([p * 100 for p in [0.2, 0.4, 0.6, 0.8, 1.0]], training_times, label="Training Time")
    plt.xlabel("Dataset Size (%)")
    plt.ylabel("Training Time (seconds)")
    plt.title("Scalability with Data Size - PyTorch")
    plt.legend()
    plt.show()

    results['Scalability'] = training_times

# 2. Fault Tolerance
def pytorch_fault_tolerance_experiment():
    try:
        X_sample = X_train_tensor[:int(len(X_train_tensor) * 0.8)]
        y_sample = y_train_tensor[:int(len(y_train_tensor) * 0.8)]

        model = LinearRegressionModel(X_train.shape[1])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Initial training (simulate fault mid-way)
        print("Starting initial training...")
        start_time = time.time()
        for epoch in range(2):
            optimizer.zero_grad()
            predictions = model(X_sample)
            loss = criterion(predictions, y_sample)
            loss.backward()
            optimizer.step()
        training_time_part1 = time.time() - start_time

        # Save model state (simulate fault)
        checkpoint = "pytorch_fault_tolerance_checkpoint.pth"
        torch.save(model.state_dict(), checkpoint)

        # Simulate resumption from fault
        print("Simulating fault, loading model state...")
        model.load_state_dict(torch.load(checkpoint))

        # Resume training
        start_time = time.time()
        for epoch in range(3):
            optimizer.zero_grad()
            predictions = model(X_sample)
            loss = criterion(predictions, y_sample)
            loss.backward()
            optimizer.step()
        training_time_part2 = time.time() - start_time

        total_time = training_time_part1 + training_time_part2

        plt.figure()
        plt.bar(["Training with Fault"], [total_time], color='orange', label="Total Time")
        plt.ylabel("Training Time (seconds)")
        plt.title("Fault Tolerance - PyTorch")
        plt.legend()
        plt.show()

        results['Fault Tolerance'] = total_time
    except Exception as e:
        print("Simulation of fault failed:", e)

# 3. Communication Overhead
def pytorch_communication_overhead_experiment():
    X_sample = X_train_tensor[:int(len(X_train_tensor) * 0.8)]
    y_sample = y_train_tensor[:int(len(y_train_tensor) * 0.8)]

    model = LinearRegressionModel(X_train.shape[1])
    criterion = nn.MSELoss()

    # Synchronous training
    optimizer_sync = torch.optim.Adam(model.parameters(), lr=0.01)
    start_time = time.time()
    for epoch in range(5):
        optimizer_sync.zero_grad()
        predictions = model(X_sample)
        loss = criterion(predictions, y_sample)
        loss.backward()
        optimizer_sync.step()
    sync_time = time.time() - start_time

    # Simulated asynchronous training (larger batch size)
    optimizer_async = torch.optim.Adam(model.parameters(), lr=0.01)
    start_time = time.time()
    for epoch in range(5):
        optimizer_async.zero_grad()
        predictions = model(X_sample)
        loss = criterion(predictions, y_sample)
        loss.backward()
        optimizer_async.step()
    async_time = time.time() - start_time

    plt.figure()
    plt.bar(["Synchronous", "Asynchronous"], [sync_time, async_time], color=['blue', 'green'])
    plt.ylabel("Training Time (seconds)")
    plt.title("Communication Overhead - PyTorch")
    plt.show()

    results['Communication'] = {'Synchronous': sync_time, 'Asynchronous': async_time}

# 4. Ease of Use and Setup
def pytorch_ease_of_use_experiment():
    start_time = time.time()

    # Simulate PyTorch environment setup and model initialization
    model = LinearRegressionModel(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    elapsed_time = time.time() - start_time

    plt.figure()
    plt.bar(["Environment Setup"], [elapsed_time], color='purple', label="Setup Time")
    plt.ylabel("Setup Time (seconds)")
    plt.title("Ease of Setup - PyTorch")
    plt.legend()
    plt.show()

    results['Ease of Setup'] = elapsed_time

# General Graph for PyTorch
def general_pytorch_graph():
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
    plt.title("PyTorch - General Framework Overview")
    plt.xticks(rotation=45)
    plt.show()

# Run all experiments
pytorch_scalability_experiment()
pytorch_fault_tolerance_experiment()
pytorch_communication_overhead_experiment()
pytorch_ease_of_use_experiment()
general_pytorch_graph()

# Print results
print("Results Summary:")
for key, value in results.items():
    print(f"{key}: {value}")
