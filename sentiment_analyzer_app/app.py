from flask import Flask, request, render_template
import re
import joblib
from helper_functions import clean_text, lemmatize_text


############################################################################################

random_forest_model = joblib.load('models/Random_forest_model.pkl')
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

def predict_sentiment(input_text):
    # Preprocess the input text
    preprocess_text = clean_text(input_text)

    preprocessed_text = lemmatize_text(preprocess_text)

    features = tfidf_vectorizer.transform([preprocessed_text])
    
    prediction = random_forest_model.predict(features)[0]
    
    if prediction == 1:
        return "Positive"
    else:
        return "Negative"

############################################################################################

app = Flask(__name__)


@app.route("/")
def home_page():
    return render_template("home.html")


@app.route("/predict", methods=["POST", "GET"])
def predicted():
    show = ""
    if request.method == "POST":
        testString = request.form.get("Test-String")
        show = predict_sentiment(testString)
    return render_template("predict.html", show=show)


if __name__ == "__main__":
    app.run(debug=True)
