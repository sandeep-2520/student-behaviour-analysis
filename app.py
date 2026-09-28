
from flask import Flask, render_template, request
import mysql.connector
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained ML models
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")


# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

def get_data():

    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="student_behavior"
    )

    df = pd.read_sql("SELECT * FROM students", db)

    db.close()

    return df


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

@app.route("/")
def dashboard():

    df = get_data()

    features = [
        "attendance",
        "study_hours",
        "assignment_score",
        "quiz_score",
        "participation",
        "lms_activity",
        "late_submissions",
        "previous_performance",
        "screen_time"
    ]

    # Scale data
    X = df[features]

    X_scaled = scaler.transform(X)

    # Predict clusters
    df["cluster"] = kmeans.predict(X_scaled)

    # Cluster names
    cluster_names = {
        0: "Low Engagement",
        1: "Moderate Engagement",
        2: "High Engagement"
    }

    df["behaviour"] = df["cluster"].map(cluster_names)

    # Dashboard statistics
    total_students = len(df)

    high_engagement = len(
        df[df["cluster"] == 2]
    )

    moderate_engagement = len(
        df[df["cluster"] == 1]
    )

    low_engagement = len(
        df[df["cluster"] == 0]
    )

    return render_template(
        "dashboard.html",
        students=df.to_dict("records"),
        total_students=total_students,
        high_engagement=high_engagement,
        moderate_engagement=moderate_engagement,
        low_engagement=low_engagement
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

@app.route("/predict", methods=["GET", "POST"])
def predict():

    # Open prediction page
    if request.method == "GET":

        return render_template("prediction.html")


    # --------------------------------------------------
    # FEATURES
    # --------------------------------------------------

    features = [
        "attendance",
        "study_hours",
        "assignment_score",
        "quiz_score",
        "participation",
        "lms_activity",
        "late_submissions",
        "previous_performance",
        "screen_time"
    ]


    # --------------------------------------------------
    # GET USER INPUT
    # --------------------------------------------------

    try:

        data = [
            float(request.form["attendance"]),
            float(request.form["study_hours"]),
            float(request.form["assignment_score"]),
            float(request.form["quiz_score"]),
            float(request.form["participation"]),
            float(request.form["lms_activity"]),
            float(request.form["late_submissions"]),
            float(request.form["previous_performance"]),
            float(request.form["screen_time"])
        ]

    except (KeyError, ValueError):

        return "Please enter valid values in all fields.", 400


    # --------------------------------------------------
    # PREPARE DATA FOR MACHINE LEARNING
    # --------------------------------------------------

    X = pd.DataFrame(
        [data],
        columns=features
    )

    # Scale input
    X_scaled = scaler.transform(X)


    # --------------------------------------------------
    # PREDICT CLUSTER
    # --------------------------------------------------

    cluster = int(
        kmeans.predict(X_scaled)[0]
    )


    # Cluster names
    cluster_names = {
        0: "Low Engagement",
        1: "Moderate Engagement",
        2: "High Engagement"
    }

    behaviour = cluster_names.get(
        cluster,
        "Unknown"
    )


    # --------------------------------------------------
    # SAVE NEW STUDENT TO DATABASE
    # --------------------------------------------------

    try:

        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="student_behavior"
        )

        cursor = db.cursor()


        insert_query = """
        INSERT INTO students
        (
            attendance,
            study_hours,
            assignment_score,
            quiz_score,
            participation,
            lms_activity,
            late_submissions,
            previous_performance,
            screen_time
        )
        VALUES
        (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """


        cursor.execute(
            insert_query,
            tuple(data)
        )


        db.commit()


        cursor.close()
        db.close()


    except mysql.connector.Error as error:

        return f"Database error: {error}", 500


    # --------------------------------------------------
    # SHOW RESULT
    # --------------------------------------------------

    return render_template(
        "prediction.html",
        behaviour=behaviour,
        cluster=cluster
    )


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    app.run(debug=True)

