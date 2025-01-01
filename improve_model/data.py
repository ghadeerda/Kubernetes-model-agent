import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def preprocess_and_merge(node_file, app_file, output_file):
    """
    Preprocess and merge node and app data, keeping only the attack column from the node data.
    """
    # Load node and app data
    node_data = pd.read_csv(node_file)
    app_data = pd.read_csv(app_file)

    # Convert `time` columns to datetime for easier merging
    node_data["time"] = pd.to_datetime(node_data["time"])
    app_data["time"] = pd.to_datetime(app_data["time"])

    # Remove the `attack` column from app_data to avoid duplication
    app_data_cleaned = app_data.drop(columns=["attack", "open_fds"], errors="ignore")

    # Merge based on closest time (nearest timestamp)
    merged_data = pd.merge_asof(
        node_data.sort_values("time"),
        app_data_cleaned.sort_values("time"),
        on="time",
        direction="nearest"
    )

    # Drop unnecessary columns and save to disk
    final_data = merged_data.drop(columns=["id_x", "id_y", "time"], errors="ignore")
    final_data.to_csv(output_file, index=False)
    print(f"Preprocessed and saved merged data to {output_file}")

def train_and_evaluate(data_file):
    """
    Train a classifier on the preprocessed data and evaluate its performance.
    """
    # Load the data
    data = pd.read_csv(data_file)

    # Drop unnecessary columns
    data = data.drop(columns=["id"], errors="ignore")

    # Separate features and target
    X = data.drop("attack", axis=1)
    y = data["attack"]

    # Normalize the features
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {data_file}:", accuracy)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# List of framework-specific input files and output files
frameworks = {
    "django": ("./data/app_django_eks_dev_us.csv", "./data/node_django_eks_dev_us.csv", "./data/final_data_django.csv"),
    "fastapi": ("./data/app_fastapi_eks_dev_us.csv", "./data/node_fastapi_eks_dev_us.csv", "./data/final_data_fastapi.csv"),
    "flask": ("./data/app_flask_eks_dev_us.csv", "./data/node_flask_eks_dev_us.csv", "./data/final_data_flask.csv"),
    "go": ("./data/app_go_eks_dev_us.csv", "./data/node_go_eks_dev_us.csv", "./data/final_data_go.csv"),
    "nodejs": ("./data/app_nodejs_eks_dev_us.csv", "./data/node_nodejs_eks_dev_us.csv", "./data/final_data_nodejs.csv")
}

# Process, merge, and train for each framework
for framework, (node_file, app_file, output_file) in frameworks.items():
    print(f"Processing data for {framework}...")
    preprocess_and_merge(node_file, app_file, output_file)
    train_and_evaluate(output_file)


## all
def combine_datasets(output_combined_file, *file_paths):
    """
    Combine multiple datasets into one final dataset.
    """
    # Load and concatenate all datasets
    combined_data = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)

    # Save the combined dataset to disk
    combined_data.to_csv(output_combined_file, index=False)
    print(f"Combined dataset saved to {output_combined_file}")

# File paths for final datasets
final_datasets = [
    "./data/final_data_django.csv",
    "./data/final_data_fastapi.csv",
    "./data/final_data_flask.csv",
    "./data/final_data_go.csv",
    "./data/final_data_nodejs.csv"
]

# Combine and save the final dataset
output_combined_file = "./data/final_data_combined.csv"
combine_datasets(output_combined_file, *final_datasets)
# Train and evaluate the combined dataset
train_and_evaluate(output_combined_file)