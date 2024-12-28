from flask import Flask, request
import joblib
import numpy as np
from flask_pymongo import PyMongo
import requests

app = Flask(__name__)
app.secret_key = '935a55bf-295f-476f-b026-8c362867fa2b'

# Journal Setup from MongoDB

app.config["MONGO_URI"] = "mongodb+srv://prameyamohanty10:PrAmEyAmOhAnTy%40100808@mydatabase.gyir6sh.mongodb.net/FemCare?retryWrites=true&w=majority&appName=MyDatabase/FemCare"
mongo = PyMongo(app)

# ========================================================================================

# Cervical Cancer Setup

model = joblib.load(r"models/cervical.joblib")
features = [["Age", "Number of sexual partners", "First sexual intercourse", "Num of pregnancies", "Smokes (years)", "Smokes (packs/year)", "Hormonal Contraceptives (years)", "IUD (years)", "STDs: Number of diagnosis"], [
    "condylomatosis", "cervical condylomatosis", "vaginal condylomatosis", "vulvo-perineal condylomatosis", "syphilis", "pelvic inflammatory disease", "genital herpes", "molluscum contagiosum", "AIDS", "HIV", "Hepatitis B", "HPV"], ["Cancer", "CIN", "hpv"]]

mainsfeatures = ["Age", "Number of sexual partners", "First sexual intercourse", "Num of pregnancies", "Smokes (years)", "Smokes (packs/year)", "Hormonal Contraceptives (years)", "IUD (years)", "STDs", "STDs (number)", "condylomatosis", "cervical condylomatosis",
                 "vaginal condylomatosis", "vulvo-perineal condylomatosis", "syphilis", "pelvic inflammatory disease", "genital herpes", "molluscum contagiosum", "AIDS", "HIV", "Hepatitis B", "HPV", "STDs: Number of diagnosis", "Cancer", "CIN", "HPV"]

# ========================================================================================

# PCOS Setup


def predict_pcos(sample):
    loaded_model = joblib.load('models/pcos.joblib')
    loaded_scaler = joblib.load('models/pcos_scaler.joblib')
    sample = np.array(sample).reshape(1, -1)
    sample = loaded_scaler.transform(sample)
    prediction = loaded_model.predict(sample)
    return "PCOS Detected" if prediction[0] == 1 else "No PCOS"

# ========================================================================================


@app.route("/cervical-cancer", methods=["GET", "POST"])
def cervical_cancer():
    values = []
    if request.method == "POST":
        for item in mainsfeatures:
            resp = request.form.get(item)
            if resp != None:
                values.append(resp)
        sum = 0
        for item in features[1]:
            sum += (int(request.form.get(item)))
        if sum > 0:
            values.insert(8, "1")
            values.insert(9, str(sum))
        else:
            values.insert(8, "0")
            values.insert(9, str(sum))
        vals = [float(value) for value in values]
        prediction = model.predict(np.array([vals]))

        return str(prediction[0]*25) + "%"


@app.route("/pcos", methods=["POST"])
def pcos():
    sample = list(request.form.to_dict().values())
    pcospred = predict_pcos(sample)
    return pcospred


@app.route("/article", methods=["POST"])
def fetchArticle():
    slug = request.args['slug']
    article = mongo.db.articles.find_one({"slug": slug})
    article.pop("_id")
    return article
    # return dumps(article), 200, {'Content-Type': 'application/json'}


@app.route("/get", methods=["POST"])
def get():
    reqUrl = "https://femcare-chatbot.vercel.app/get"
    msg = request.form["msg"]

    headersList = {
        "Accept": "*/*",
        "User-Agent": "Thunder Client (https://www.thunderclient.com)",
        "Content-Type": "multipart/form-data; boundary=kljmyvW1ndjXaOEAg4vPm6RBUqO6MC5A"
    }

    payload = f"--kljmyvW1ndjXaOEAg4vPm6RBUqO6MC5A\r\nContent-Disposition: form-data; name=\"msg\"\r\n\r\n{msg}\r\n--kljmyvW1ndjXaOEAg4vPm6RBUqO6MC5A--\r\n"

    response = requests.request(
        "POST", reqUrl, data=payload,  headers=headersList)
    return response.text


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port="5002")
