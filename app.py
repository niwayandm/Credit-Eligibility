from flask import Flask, render_template, request, redirect, make_response
import joblib
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

filename = "model/rf.pkl"
model = joblib.load(filename)

@app.route("/")
def index():
    return render_template("index.html") 

@app.route('/predict', methods=['GET', 'POST'])
def home():
    df = pd.read_csv('dataset/data_clean.csv')
    df=df.drop('Loan_Status',axis=1)
    
    if request.method == "POST":
        name = request.form["name"]
        gender = request.form["gender"]
        married = request.form["married"]
        dependents = request.form["dependents"]
        education = request.form["education"]
        self_employed = request.form["self_employed"]
        applicant_income = int(request.form["applicant_income"])
        co_applicant_income = int(request.form["co_applicant_income"])
        loan_amount = int(request.form["loan_amount"])
        loan_amount_term = float(request.form["loan_amount_term"])
        credit_history = float(request.form["credit_history"])
        property_area = request.form["property_area"]

        if(property_area=='Urban'):
            df = df.append({'Gender': gender,'Married': married, 'Dependents': dependents,
            'Education': education,'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,'CoapplicantIncome': co_applicant_income,'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term, 'Credit_History': credit_history, 
            'Rural_Area':0, 'Urban_Area':1, 'Semiurban_Area':0}, ignore_index=True)
        if(property_area=='Rural'):
            df = df.append({'Gender': gender,'Married': married, 'Dependents': dependents,
            'Education': education,'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,'CoapplicantIncome': co_applicant_income,'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term, 'Credit_History': credit_history, 
            'Rural_Area':1, 'Urban_Area':0, 'Semiurban_Area':0}, ignore_index=True)
        if(property_area=='Semiurban'):
            df = df.append({'Gender': gender,'Married': married, 'Dependents': dependents,
            'Education': education,'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,'CoapplicantIncome': co_applicant_income,'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term, 'Credit_History': credit_history, 
            'Rural_Area':0, 'Urban_Area':0, 'Semiurban_Area':1}, ignore_index=True)
    
    interest = loan_amount * 2/100
    max_loan = interest + loan_amount
    loan_term_months = int(loan_amount_term/30)
    loan_per_month = round(max_loan/loan_term_months,3)

    pred = int(model.predict(df.iloc[[-1]]))
    return render_template('result.html', pred=pred, name=name, max_loan=max_loan,
        loan_term_months=loan_term_months, loan_per_month=loan_per_month)

if __name__ == "__main__":
    app.debug=True
    app.run()