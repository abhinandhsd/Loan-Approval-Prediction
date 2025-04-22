

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model=pickle.load(open('savedmodel.pkl','rb'))

@app.route('/')
def home():
    result=''
    return render_template('abcd.html',**locals())

@app.route('/predict', methods=['POST','GET'])
def predict():
    Dependents=float(request.form['Dependents'])	
    Education=float(request.form['Education'])	
    ApplicantIncome=float(request.form['ApplicantIncome'])	
    CoapplicantIncome=float(request.form['CoapplicantIncome'])
    LoanAmount=float(request.form['LoanAmount'])	
    Credit_History=float(request.form['Credit_History'])	
    Property_Area=float(request.form['Property_Area'])	
    Male=float(request.form['Gender'])	
    Employed_Yes=float(request.form['Employed_Yes'])	
    Married_Yes=float(request.form['Married_Yes'])
    result= np.array([[Dependents, Education, ApplicantIncome, CoapplicantIncome, 
                                LoanAmount, Credit_History, Property_Area, Male, Employed_Yes, Married_Yes]])
    prediction = model.predict(result)[0]

    return f"The prediction result is: {prediction}"

if __name__ == '__main__':
    app.run(debug=True)

