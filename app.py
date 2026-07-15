from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("models/loan_model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict")
def predict_page():
    return render_template("predict.html")


@app.route("/submit", methods=["POST"])
def submit():

    try:

        gender = int(request.form["gender"])
        married = int(request.form["married"])
        dependents = int(request.form["dependents"])
        education = int(request.form["education"])
        self_employed = int(request.form["self_employed"])

        applicant_income = float(request.form["applicant_income"])
        coapplicant_income = float(request.form["coapplicant_income"])

        loan_amount = float(request.form["loan_amount"])
        loan_term = float(request.form["loan_term"])

        credit_history = float(request.form["credit_history"])
        property_area = int(request.form["property_area"])

        # Engineered Features

        total_income = applicant_income + coapplicant_income

        loan_income_ratio = loan_amount / total_income if total_income != 0 else 0

        emi_estimate = (loan_amount * 1000) / loan_term if loan_term != 0 else 0

        if total_income <= 3000:
            income_category = 0
        elif total_income <= 6000:
            income_category = 1
        elif total_income <= 9000:
            income_category = 2
        else:
            income_category = 3

        features = [[
            gender,
            married,
            dependents,
            education,
            self_employed,
            applicant_income,
            coapplicant_income,
            loan_amount,
            loan_term,
            credit_history,
            property_area,
            total_income,
            loan_income_ratio,
            emi_estimate,
            income_category
        ]]

        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "Loan Approved"
            color = "green"
        else:
            result = "Loan Rejected"
            color = "red"

        return render_template(
            "submit.html",
            prediction=result,
            color=color,
            total_income=round(total_income,2),
            emi=round(emi_estimate,2)
        )

    except Exception as e:
        return f"Error : {e}"


if __name__ == "__main__":
    app.run(debug=True)