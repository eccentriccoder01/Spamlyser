import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Title and description
st.title("📱 Spamlyser: SMS Spam Classifier")
st.markdown("Classify SMS messages using different transformer backbones trained on the `sms_spam` dataset.")

# Updated Hugging Face model repo locations
MODEL_OPTIONS = {
    "DistilBERT": "mreccentric/distilbert-base-uncased-spamlyser",
    "BERT": "mreccentric/bert-base-uncased-spamlyser",
    "RoBERTa": "mreccentric/roberta-base-spamlyser",
    "ALBERT": "mreccentric/albert-base-v2-spamlyser"
}

# Sidebar for model selection
selected_model_name = st.sidebar.selectbox("Choose a model", list(MODEL_OPTIONS.keys()))
model_id = MODEL_OPTIONS[selected_model_name]

# Load model and tokenizer (cached to prevent reloading on every run)
@st.cache_resource
def load_pipeline(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

classifier = load_pipeline(model_id)

# User input
user_sms = st.text_area("Enter SMS message to classify", height=150)

# Classify button
if st.button("Classify"):
    if user_sms.strip():
        result = classifier(user_sms)[0]
        label = result['label']
        score = result['score']
        st.write(f"**Prediction:** `{label}` with confidence `{score:.2f}`")
    else:
        st.warning("Please enter an SMS message.")
