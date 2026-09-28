# 🎓 Student Behaviour Analysis

A machine-learning-based web application for analyzing and understanding student behavioural patterns using **K-Means Clustering** and **Principal Component Analysis (PCA)**.

The project processes student data, applies preprocessing and dimensionality reduction, and uses clustering to identify groups of students with similar behavioural characteristics.

## 🚀 Features

* 📊 Student behaviour data analysis
* 🤖 K-Means clustering
* 📉 PCA-based dimensionality reduction
* ⚙️ Data preprocessing and feature scaling
* 📁 CSV-based student dataset
* 🌐 Web interface using Flask
* 🔍 Student-level behaviour analysis
* 💾 Pre-trained machine-learning models

## 🧠 Machine Learning Approach

The project uses the following machine-learning workflow:

```text
Student Dataset
      ↓
Data Preprocessing
      ↓
Feature Scaling
      ↓
PCA
      ↓
K-Means Clustering
      ↓
Student Behaviour Groups
      ↓
Web Application
```

### K-Means Clustering

K-Means is used to group students according to similarities in their behavioural data.

Students with similar characteristics are assigned to the same cluster.

### PCA

Principal Component Analysis (PCA) is used to reduce the dimensionality of the dataset while retaining important information.

This can make behavioural patterns easier to analyze and visualize.

## 🛠️ Technologies Used

* **Python**
* **Flask**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **HTML/CSS**
* **Machine Learning**
* **Git & GitHub**

## 📂 Project Structure

```text
student-behaviour-analysis/
│
├── app.py
├── students.py
├── train_model.py
├── students.csv
│
├── kmeans_model.pkl
├── pca_model.pkl
├── scaler.pkl
│
├── requirements.txt
├── README.md
│
└── templates/
    └── ...
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/sandeep-2520/student-behaviour-analysis.git
```

### 2. Enter the project directory

```bash
cd student-behaviour-analysis
```

### 3. Create a virtual environment

```bash
python3 -m venv venv
```

### 4. Activate the virtual environment

macOS/Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Running the Application

Start the Flask application:

```bash
python app.py
```

The application will start on the local Flask server.

Open the URL displayed in your terminal in a web browser.

## 📊 Dataset

The project uses `students.csv` as the student dataset.

The dataset is processed before being passed to the machine-learning pipeline.

## 📦 Pre-trained Models

The repository contains the trained model files:

```text
kmeans_model.pkl
pca_model.pkl
scaler.pkl
```

These files allow the application to use the trained preprocessing and machine-learning components.

## 🔬 Project Objective

The main objective of this project is to demonstrate how machine learning can be applied to student behavioural data to discover meaningful patterns and groups.

The analysis can help provide a structured view of behavioural characteristics within a student dataset.

## 🔮 Future Improvements

Possible future improvements include:

* Interactive data visualizations
* Student performance prediction
* More behavioural features
* Automated reports
* Dashboard with cluster statistics
* Database integration
* Authentication and user management
* Deployment to a cloud platform
* Additional machine-learning algorithms

## 👨‍💻 Author

**Uppari Sandeep**

GitHub:
https://github.com/sandeep-2520

## 📄 License

This project is intended for educational and demonstration purposes.

