

# MLflow Decision Tree Classifier — Iris Dataset

This project demonstrates how to use **MLflow** for experiment tracking, model logging, and artifact management using the classic **Iris dataset** and a **Decision Tree Classifier**.

The pipeline covers:

* Training and evaluation of a Decision Tree model.
* Logging of parameters, metrics, artifacts, and models to an MLflow tracking server.
* Saving visualizations (confusion matrix) for analysis.

---

## 📂 Project Structure

```
.
├── main.py                  # MLflow experiment tracking code
├── requirements.txt         # Python dependencies
├── confusion_matrix.png     # Logged confusion matrix artifact (generated)
└── README.md                # Project documentation
```

---

## ⚙️ Prerequisites

1. **Python 3.8+**
2. **MLflow Tracking Server** running locally or remotely.

Example (local tracking server):

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

**requirements.txt**

```
mlflow
scikit-learn
matplotlib
seaborn
```

---

## 🚀 Running the Project

1. **Start MLflow Tracking Server**

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

2. **Run the Script**

```bash
python main.py
```

3. **View the Experiment in MLflow UI**
   Open your browser and go to:

```
http://localhost:5000
```

Navigate to the experiment **"iris-dt"** to view:

* Parameters (`max_depth`)
* Metrics (`accuracy`)
* Artifacts (`confusion_matrix.png`, source code)
* Registered model

---

## 📊 What This Script Does

* Loads the **Iris dataset**.
* Splits data into train/test sets.
* Trains a Decision Tree model with `max_depth=4`.
* Evaluates the model and computes accuracy.
* Generates a confusion matrix plot.
* Logs the following to MLflow:

  * **Parameters**: `max_depth`
  * **Metrics**: `accuracy`
  * **Artifacts**: confusion matrix image, source code
  * **Model**: serialized Decision Tree model
  * **Tags**: model type, author

---

## 🖼️ Example Confusion Matrix Output

*(Automatically logged to MLflow)*

```
         Predicted
Actual   setosa  versicolor  virginica
setosa      10       0         0
versicolor   0       9         1
virginica    0       1         9
```

---

## 📌 Notes

* The tracking URI is set to `http://localhost:5000`. Update `mlflow.set_tracking_uri()` if using a remote server.
* MLflow will automatically create the experiment **"iris-dt"** if it doesn't exist.
* You can register the logged model in the MLflow Model Registry for deployment.

---

## 🧑‍💻 Author

**Dhyanendra** — Software Engineer, Data Science Enthusiast

