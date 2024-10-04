import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load pre-trained models and tokenizers from Hugging Face
muril_model = AutoModelForSequenceClassification.from_pretrained('google/muril-base-cased')
muril_tokenizer = AutoTokenizer.from_pretrained('google/muril-base-cased')

# Load a different model for sentiment analysis
indicbert_model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
indicbert_tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
muril_model.to(device)
indicbert_model.to(device)

# Function for sentiment prediction
def predict_sentiment(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
    return predictions.item()

# Streamlit App
st.title('Hindi Review Sentiment Analysis')
st.write('Enter a review in Hindi, and the sentiment will be predicted using two models: MuRIL and a multilingual BERT model.')

# User Input
review_text = st.text_area('Enter Hindi Review', '')

if st.button('Analyze Sentiment'):
    if review_text:
        # Perform Sentiment Analysis using MuRIL
        muril_prediction = predict_sentiment(muril_model, muril_tokenizer, review_text)
        # Perform Sentiment Analysis using IndicBERT
        indicbert_prediction = predict_sentiment(indicbert_model, indicbert_tokenizer, review_text)

        # Determine sentiment based on predictions
        muril_sentiment = "Positive" if muril_prediction == 1 else "Negative"
        indicbert_sentiment = "Positive" if indicbert_prediction >= 3 else "Negative"  # Adjust threshold for IndicBERT

        # Display Results
        st.write(f"**MuRIL Model Prediction:** {muril_sentiment}")
        st.write(f"**IndicBERT Model Prediction:** {indicbert_sentiment}")
    else:
        st.write("Please enter a review to analyze.")
