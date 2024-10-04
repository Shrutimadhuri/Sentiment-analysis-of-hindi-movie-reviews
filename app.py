from flask import Flask, render_template, request
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib

app = Flask(__name__)

# Load the model and label encoder
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3)
model.load_state_dict(torch.load('final_bert_model.pth'))
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
label_encoder = joblib.load('bert_label_encoder.joblib')

# Define sentiment labels
sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']

    # Tokenize the input review
    inputs = tokenizer.encode_plus(
        review,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Make prediction
    with torch.no_grad():
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        _, predicted = torch.max(outputs.logits, dim=1)
    
    # Get the sentiment label
    sentiment = sentiment_labels[predicted.item()]

    return render_template('index.html', prediction=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
