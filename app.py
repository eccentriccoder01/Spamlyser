import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Assuming these files exist in the same directory
from export_feature import export_results_button 
# from ensemble_classifier_method import EnsembleSpamClassifier, ModelPerformanceTracker, PredictionResult
# Dummy classes if you don't have the actual files for testing
try:
    from ensemble_classifier_method import EnsembleSpamClassifier, ModelPerformanceTracker, PredictionResult
except ImportError:
    st.warning("`ensemble_classifier_method.py` not found. Using dummy classes. Please provide the actual file for full functionality.")
    class PredictionResult:
        def __init__(self, label, score, spam_probability=None):
            self.label = label
            self.score = score
            self.spam_probability = spam_probability
    class ModelPerformanceTracker:
        def __init__(self):
            self.stats = {}
        def update_performance(self, model_name, correct): pass
        def get_all_stats(self): return {}
        def save_to_file(self, filename): pass
    class EnsembleSpamClassifier:
        def __init__(self, performance_tracker):
            self.performance_tracker = performance_tracker
            self.default_weights = {"DistilBERT": 0.25, "BERT": 0.25, "RoBERTa": 0.25, "ALBERT": 0.25}
            self.model_weights = self.default_weights.copy()
        def update_model_weights(self, weights): self.model_weights.update(weights)
        def get_model_weights(self): return self.model_weights
        def get_ensemble_prediction(self, predictions, method):
            # Dummy implementation for ensemble prediction
            if not predictions:
                return {'label': 'UNKNOWN', 'confidence': 0.0, 'spam_probability': 0.0, 'method': method, 'details': 'No model predictions'}
            
            # Simple majority voting for dummy
            spam_votes = sum(1 for p in predictions.values() if p['label'] == 'SPAM')
            ham_votes = sum(1 for p in predictions.values() if p['label'] == 'HAM')
            
            if spam_votes > ham_votes:
                label = 'SPAM'
                score = sum(p['score'] for p in predictions.values() if p['label'] == 'SPAM') / spam_votes if spam_votes else 0
                spam_prob = score # Simplified
            elif ham_votes > spam_votes:
                label = 'HAM'
                score = sum(p['score'] for p in predictions.values() if p['label'] == 'HAM') / ham_votes if ham_votes else 0
                spam_prob = 1 - score # Simplified
            else: # Tie or no clear majority, default to HAM for safety or SPAM for caution
                label = 'HAM' 
                score = 0.5
                spam_prob = 0.5
            return {'label': label, 'confidence': score, 'spam_probability': spam_prob, 'method': method, 'details': f'Dummy {method} applied'}
        
        def get_all_predictions(self, predictions):
            # Dummy method to return results for all ensemble methods
            dummy_results = {}
            for method_key in ["majority_voting", "weighted_average", "confidence_weighted", "adaptive_threshold", "meta_ensemble"]:
                dummy_results[method_key] = self.get_ensemble_prediction(predictions, method_key)
            return dummy_results
        
# Core Python imports
import time
import re
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from io import StringIO
import torch
from collections import defaultdict # Added for easier analytics data aggregation

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Spamlyser Pro - Ensemble Edition",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 100%);
    }
    
    .metric-container {
        background: linear-gradient(145deg, #1e1e1e, #2a2a2a);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #333;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin: 10px 0;
    }
    
    .prediction-card {
        background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
        padding: 25px;
        border-radius: 20px;
        border: 1px solid #404040;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        text-align: center;
        margin: 20px 0;
    }
    
    .ensemble-card {
        background: linear-gradient(145deg, #1a1a2a, #2d2d3d);
        padding: 20px;
        border-radius: 15px;
        border: 2px solid #6366f1;
        margin: 15px 0;
    }
    
    .spam-alert {
        background: linear-gradient(145deg, #2a1a1a, #3d2626);
        border: 2px solid #ff4444;
        color: #ff6b6b;
    }
    
    .ham-safe {
        background: linear-gradient(145deg, #1a2a1a, #263d26);
        border: 2px solid #44ff44;
        color: #6bff6b;
    }
    
    .analysis-header {
        background: linear-gradient(90deg, #333, #555);
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 4px solid #00d4aa;
    }
    
    .feature-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    
    .model-info {
        background: linear-gradient(145deg, #252525, #3a3a3a);
        padding: 15px;
        border-radius: 12px;
        border-left: 4px solid #00d4aa;
        margin: 15px 0;
    }
    
    .ensemble-method {
        background: linear-gradient(145deg, #252545, #3a3a5a);
        padding: 12px;
        border-radius: 10px;
        border-left: 4px solid #6366f1;
        margin: 8px 0;
    }
    
    .method-comparison {
        background: rgba(255, 255, 255, 0.03);
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)


# --- Load Sample Messages (with fallback) ---
try:
    sample_df = pd.read_csv("sample_data.csv")
except FileNotFoundError:
    st.warning("`sample_data.csv` not found. Creating a dummy DataFrame for sample messages.")
    sample_df = pd.DataFrame({
        'message': [
            "WINNER! You have been selected for a £1000 prize. Call now!",
            "Hi Mom, just letting you know I'm home safe.",
            "Free entry to our exclusive lottery! Text WIN to 87879.",
            "Meeting at 3 PM, don't be late.",
            "Urgent: Your bank account has been compromised. Verify at https://bit.ly/malicious",
            "Hey, how are you doing today?",
            "Congratulations! You've won a new iPhone! Claim your prize here: http://tinyurl.com/prize",
            "Just confirming our appointment for tomorrow at 10 AM.",
            "Your subscription is expiring. Renew now to avoid service interruption."
        ]
    })

# --- Session State Initialization ---
if 'classification_history' not in st.session_state:
    st.session_state.classification_history = []
if 'model_stats' not in st.session_state:
    st.session_state.model_stats = {model: {'spam': 0, 'ham': 0, 'total': 0} for model in ["DistilBERT", "BERT", "RoBERTa", "ALBERT"]}
if 'ensemble_tracker' not in st.session_state:
    st.session_state.ensemble_tracker = ModelPerformanceTracker()
if 'ensemble_classifier' not in st.session_state:
    st.session_state.ensemble_classifier = EnsembleSpamClassifier(performance_tracker=st.session_state.ensemble_tracker)
if 'ensemble_history' not in st.session_state:
    st.session_state.ensemble_history = []
if 'loaded_models' not in st.session_state:
    st.session_state.loaded_models = {model_name: None for model_name in ["DistilBERT", "BERT", "RoBERTa", "ALBERT"]}


# --- Model Configurations ---
MODEL_OPTIONS = {
    "DistilBERT": {
        "id": "mreccentric/distilbert-base-uncased-spamlyser",
        "description": "Lightweight & Fast",
        "icon": "⚡",
        "color": "#ff6b6b"
    },
    "BERT": {
        "id": "mreccentric/bert-base-uncased-spamlyser",
        "description": "Balanced Performance",
        "icon": "🎯",
        "color": "#4ecdc4"
    },
    "RoBERTa": {
        "id": "mreccentric/roberta-base-spamlyser",
        "description": "Robust & Accurate",
        "icon": "🚀",
        "color": "#45b7d1"
    },
    "ALBERT": {
        "id": "mreccentric/albert-base-v2-spamlyser",
        "description": "Parameter Efficient",
        "icon": "🧠",
        "color": "#96ceb4"
    }
}

ENSEMBLE_METHODS = {
    "majority_voting": {
        "name": "Majority Voting",
        "description": "Each model votes, majority wins",
        "icon": "🗳️",
        "color": "#ff6b6b"
    },
    "weighted_average": {
        "name": "Weighted Average",
        "description": "Combines probabilities with model weights",
        "icon": "⚖️",
        "color": "#4ecdc4"
    },
    "confidence_weighted": {
        "name": "Confidence Weighted",
        "description": "Weights votes by model confidence",
        "icon": "🎯",
        "color": "#45b7d1"
    },
    "adaptive_threshold": {
        "name": "Adaptive Threshold",
        "description": "Adjusts threshold based on agreement",
        "icon": "🔧",
        "color": "#96ceb4"
    },
    "meta_ensemble": {
        "name": "Meta Ensemble",
        "description": "Combines all methods, picks best",
        "icon": "🧠",
        "color": "#a855f7"
    }
}

# --- Header ---
st.markdown("""
<div style="text-align: center; padding: 20px 0; background: linear-gradient(90deg, #1a1a1a, #2d2d2d); border-radius: 15px; margin-bottom: 30px; border: 1px solid #404040;">
    <h1 style="color: #00d4aa; font-size: 3rem; margin: 0; text-shadow: 0 0 20px rgba(0, 212, 170, 0.3);">
        🛡️ Spamlyser Pro - Ensemble Edition
    </h1>
    <p style="color: #888; font-size: 1.2rem; margin: 10px 0 0 0;">
        Advanced Multi-Model SMS Threat Detection & Analysis Platform
    </p>
</div>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(145deg, #1e1e1e, #2a2a2a); border-radius: 15px; margin-bottom: 20px;">
        <h3 style="color: #00d4aa; margin: 0;">Analysis Mode</h3>
    </div>
    """, unsafe_allow_html=True)
    
    analysis_mode = st.radio(
        "Choose Analysis Mode",
        ["Single Model", "Ensemble Analysis"],
        help="Single Model: Use one model at a time\nEnsemble: Use all models together"
    )
    
    if analysis_mode == "Single Model":
        selected_model_name = st.selectbox(
            "Choose AI Model",
            list(MODEL_OPTIONS.keys()),
            format_func=lambda x: f"{MODEL_OPTIONS[x]['icon']} {x} - {MODEL_OPTIONS[x]['description']}"
        )
        
        model_info = MODEL_OPTIONS[selected_model_name]
        st.markdown(f"""
        <div class="model-info">
            <h4 style="color: {model_info['color']}; margin: 0 0 10px 0;">
                {model_info['icon']} {selected_model_name}
            </h4>
            <p style="color: #ccc; margin: 0; font-size: 0.9rem;">
                {model_info['description']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    else: # Ensemble Analysis Mode
        st.markdown("### 🎯 Ensemble Configuration")
        selected_ensemble_method = st.selectbox(
            "Choose Ensemble Method",
            list(ENSEMBLE_METHODS.keys()),
            format_func=lambda x: f"{ENSEMBLE_METHODS[x]['icon']} {ENSEMBLE_METHODS[x]['name']}"
        )
        method_info = ENSEMBLE_METHODS[selected_ensemble_method]
        st.markdown(f"""
        <div class="model-info">
            <h4 style="color: {method_info['color']}; margin: 0 0 10px 0;">
                {method_info['icon']} {method_info['name']}
            </h4>
            <p style="color: #ccc; margin: 0; font-size: 0.9rem;">
                {method_info['description']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        if selected_ensemble_method == "weighted_average":
            st.markdown("#### ⚖️ Model Weights")
            weights = {}
            for model_name in MODEL_OPTIONS.keys():
                default_weight = st.session_state.ensemble_classifier.model_weights.get(model_name, 0.25)
                weights[model_name] = st.slider(
                    f"{MODEL_OPTIONS[model_name]['icon']} {model_name}",
                    0.0, 1.0, default_weight, 0.05
                )
            if st.button("Update Weights"):
                st.session_state.ensemble_classifier.update_model_weights(weights)
                st.success("Weights updated!")
        if selected_ensemble_method == "adaptive_threshold":
            st.markdown("#### 🎛️ Threshold Settings")
            base_threshold = st.slider("Base Threshold", 0.1, 0.9, 0.5, 0.05) # This variable is not used further in the provided code logic.

    st.markdown("---")
    
    # Sidebar Overall Stats
    st.markdown("### 📊 Overall Statistics")
    total_single_predictions = sum(st.session_state.model_stats[model]['total'] for model in MODEL_OPTIONS)
    total_ensemble_predictions = len(st.session_state.ensemble_history)
    total_predictions_overall = total_single_predictions + total_ensemble_predictions

    st.markdown(f"""
    <div class="metric-container">
        <p style="color: #00d4aa; font-size: 1.2rem; margin-bottom: 5px;">Total Predictions</p>
        <h3 style="color: #eee; margin-top: 0;">{total_predictions_overall}</h3>
    </div>
    """, unsafe_allow_html=True)

    overall_spam_count = sum(st.session_state.model_stats[model]['spam'] for model in MODEL_OPTIONS) + \
                         sum(1 for entry in st.session_state.ensemble_history if entry['prediction'] == 'SPAM')
    overall_ham_count = sum(st.session_state.model_stats[model]['ham'] for model in MODEL_OPTIONS) + \
                        sum(1 for entry in st.session_state.ensemble_history if entry['prediction'] == 'HAM')
    
    col_spam, col_ham = st.columns(2)
    with col_spam:
        st.markdown(f"""
        <div class="metric-container spam-alert" style="padding: 15px;">
            <p style="color: #ff6b6b; font-size: 1rem; margin-bottom: 5px;">Spam Count</p>
            <h4 style="color: #ff6b6b; margin-top: 0;">{overall_spam_count}</h4>
        </div>
        """, unsafe_allow_html=True)
    with col_ham:
        st.markdown(f"""
        <div class="metric-container ham-safe" style="padding: 15px;">
            <p style="color: #6bff6b; font-size: 1rem; margin-bottom: 5px;">Ham Count</p>
            <h4 style="color: #6bff6b; margin-top: 0;">{overall_ham_count}</h4>
        </div>
        """, unsafe_allow_html=True)


# --- Model Loading Helpers ---
@st.cache_resource
def load_tokenizer(model_id):
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        st.error(f"❌ Error loading tokenizer for {model_id}: {str(e)}")
        return None

@st.cache_resource
def load_model(model_id):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    except Exception as e:
        st.error(f"❌ Error loading model {model_id}: {str(e)}")
        return None

@st.cache_resource
def _load_model_cached(model_id):
    try:
        tokenizer = load_tokenizer(model_id)
        model = load_model(model_id)
        if tokenizer is None or model is None:
            return None
        pipe = pipeline(
            "text-classification", 
            model=model, 
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
        return pipe
    except Exception as e:
        st.error(f"❌ Error creating pipeline for {model_id}: {str(e)}")
        return None

def load_model_if_needed(model_name, _progress_callback=None):
    if st.session_state.loaded_models[model_name] is None:
        model_id = MODEL_OPTIONS[model_name]["id"]
        status_container = st.empty()
        def update_status(message):
            if status_container:
                status_container.info(message)
            if _progress_callback:
                _progress_callback(message)
        try:
            update_status(f"Starting to load {model_name}...")
            update_status(f"🔄 Loading tokenizer for {model_name}...")
            update_status(f"🤖 Loading {model_name} model... (This may take a few minutes)")
            model = _load_model_cached(model_id)
            if model is not None:
                update_status(f"✅ Successfully loaded {model_name}")
                st.session_state.loaded_models[model_name] = model
            else:
                update_status(f"❌ Failed to load {model_name}")
                return None
            time.sleep(1)
        except Exception as e:
            update_status(f"❌ Error loading {model_name}: {str(e)}")
            return None
        finally:
            time.sleep(1)
            status_container.empty()
    return st.session_state.loaded_models[model_name]

def get_loaded_models():
    models = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_models = len(MODEL_OPTIONS)
    def update_progress(progress, message=""):
        progress_bar.progress(progress)
        if message:
            status_text.info(message)
    for i, (name, model_info) in enumerate(MODEL_OPTIONS.items()):
        update_progress(
            (i / total_models) * 0.9,
            f"Loading {name} model ({i+1}/{total_models})..."
        )
        models[name] = load_model_if_needed(
            name, 
            _progress_callback=lambda msg: update_progress(
                (i / total_models) * 0.9, 
                f"{name}: {msg}"
            )
        )
    update_progress(1.0, "✅ All models loaded successfully!")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    return models

load_all_models = get_loaded_models

# --- Helper Functions ---
def analyse_message_features(message):
    features = {
        'length': len(message),
        'word_count': len(message.split()),
        'uppercase_ratio': sum(1 for c in message if c.isupper()) / len(message) if message else 0,
        'digit_ratio': sum(1 for c in message if c.isdigit()) / len(message) if message else 0,
        'special_chars': len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', message)),
        'urls': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', message)),
        'phone_numbers': len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', message)),
        'exclamation_marks': message.count('!'),
        'question_marks': message.count('?')
    }
    return features

def get_risk_indicators(message, prediction):
    indicators = []
    spam_keywords = ['free', 'win', 'winner', 'congratulations', 'urgent', 'limited', 'offer', 'click', 'call now']
    found_keywords = [word for word in spam_keywords if word.lower() in message.lower()]
    if found_keywords:
        indicators.append(f"⚠️ Spam keywords detected: {', '.join(found_keywords)}")
    if len(message) > 0:
        uppercase_ratio = sum(1 for c in message if c.isupper()) / len(message)
        if uppercase_ratio > 0.3:
            indicators.append("🔴 Excessive uppercase usage")
    if message.count('!') > 2:
        indicators.append("❗ Multiple exclamation marks")
    if re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', message):
        indicators.append("📞 Phone number detected")
    if re.search(r'http[s]?://', message):
        indicators.append("🔗 URL detected")
    return indicators

def get_ensemble_predictions(message, models):
    predictions = {}
    for model_name, model in models.items():
        if model:
            try:
                result = model(message)[0]
                predictions[model_name] = {
                    'label': result['label'].upper(),
                    'score': result['score']
                }
            except Exception as e:
                st.warning(f"Error with {model_name}: {str(e)}")
                continue
    return predictions

# --- Main Interface ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
    <div class="analysis-header">
        <h3 style="color: #00d4aa; margin: 0;">🔍 {analysis_mode} Analysis</h3>
    </div>
    """, unsafe_allow_html=True)

    # Message input with sample selector
    sample_messages = [""] + sample_df["message"].tolist()
    selected_message = st.selectbox("Choose a sample message (or type your own below): ", sample_messages, key="sample_selector")

    # Set initial value of text_area based on sample_selector or previous user input
    user_sms_initial_value = selected_message if selected_message else st.session_state.get('user_sms_input_value', "")
    user_sms = st.text_area(
        "Enter SMS message to analyse",
        value=user_sms_initial_value,
        height=120,
        placeholder="Type or paste your SMS message here...",
        help="Enter the SMS message you want to classify as spam or ham (legitimate)",
        key="user_sms_input"
    )
    # Store current text_area value in session state for persistence
    st.session_state.user_sms_input_value = user_sms
    
    # Analysis controls
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        analyse_btn = st.button("🔍 Analyse Message", type="primary", use_container_width=True)
    with col_b:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    if clear_btn:
        st.session_state.user_sms_input_value = "" # Clear text area content
        if "sample_selector" in st.session_state:
            st.session_state.pop("sample_selector") # Reset sample selection
        st.rerun() # Rerun to update the UI with cleared values

if analyse_btn and user_sms.strip():
    if analysis_mode == "Single Model":
        classifier = load_model_if_needed(selected_model_name)
        if classifier is not None:
            with st.spinner(f"🤖 Analyzing with {selected_model_name}..."):
                time.sleep(0.5)
                result = classifier(user_sms)[0]
                label = result['label'].upper()
                confidence = result['score']
                st.session_state.model_stats[selected_model_name][label.lower()] += 1
                st.session_state.model_stats[selected_model_name]['total'] += 1
                st.session_state.classification_history.append({
                    'timestamp': datetime.now(),
                    'message': user_sms[:100] + "..." if len(user_sms) > 100 else user_sms, # Increased snippet length
                    'prediction': label,
                    'confidence': confidence,
                    'model': selected_model_name
                })
                features = analyse_message_features(user_sms)
                risk_indicators = get_risk_indicators(user_sms, label)
                st.markdown("### 🎯 Classification Results")
                card_class = "spam-alert" if label == "SPAM" else "ham-safe"
                icon = "🚨" if label == "SPAM" else "✅"
                st.markdown(f"""
                <div class="prediction-card {card_class}">
                    <h2 style="margin: 0 0 15px 0;">{icon} {label}</h2>
                    <h3 style="margin: 0;">Confidence: {confidence:.2%}</h3>
                    <p style="margin: 15px 0 0 0; opacity: 0.8;">
                        Model: {selected_model_name} | Analysed: {datetime.now().strftime('%H:%M:%S')}
                    </p>
                </div>
                """, unsafe_allow_html=True)
    else: # Ensemble Analysis
        with st.spinner("🤖 Loading all models for ensemble analysis..."):
            models = {}
            for model_name in MODEL_OPTIONS:
                models[model_name] = load_model_if_needed(model_name)
        if any(models.values()):
            with st.spinner("🔍 Running ensemble analysis..."):
                predictions = get_ensemble_predictions(user_sms, models)
                if predictions:
                    ensemble_result = st.session_state.ensemble_classifier.get_ensemble_prediction(
                        predictions, selected_ensemble_method
                    )
                    st.session_state.ensemble_history.append({
                        'timestamp': datetime.now(),
                        'message': user_sms[:100] + "..." if len(user_sms) > 100 else user_sms, # Increased snippet length
                        'prediction': ensemble_result['label'],
                        'confidence': ensemble_result['confidence'],
                        'method': selected_ensemble_method,
                        'spam_probability': ensemble_result['spam_probability']
                    })
                    features = analyse_message_features(user_sms)
                    risk_indicators = get_risk_indicators(user_sms, ensemble_result['label'])
                    st.markdown("### 🎯 Ensemble Classification Results")
                    card_class = "spam-alert" if ensemble_result['label'] == "SPAM" else "ham-safe"
                    icon = "🚨" if ensemble_result['label'] == "SPAM" else "✅"
                    st.markdown(f"""
                    <div class="prediction-card {card_class} ensemble-card">
                        <h2 style="margin: 0 0 15px 0;">{icon} {ensemble_result['label']}</h2>
                        <h3 style="margin: 0;">Confidence: {ensemble_result['confidence']:.2%}</h3>
                        <h4 style="margin: 10px 0;">Spam Probability: {ensemble_result['spam_probability']:.2%}</h4>
                        <p style="margin: 15px 0 0 0; opacity: 0.8;">
                            Method: {ENSEMBLE_METHODS[selected_ensemble_method]['name']} | 
                            Analysed: {datetime.now().strftime('%H:%M:%S')}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("#### 🤖 Individual Model Predictions")
                    cols = st.columns(len(predictions))
                    for i, (model_name, pred) in enumerate(predictions.items()):
                        with cols[i]:
                            color = "#ff6b6b" if pred['label'] == "SPAM" else "#4ecdc4"
                            st.markdown(f"""
                            <div class="method-comparison">
                                <h5 style="color: {MODEL_OPTIONS[model_name]['color']}; margin: 0;">
                                    {MODEL_OPTIONS[model_name]['icon']} {model_name}
                                </h5>
                                <p style="color: {color}; margin: 5px 0; font-weight: bold;">
                                    {pred['label']}
                                </p>
                                <p style="margin: 0; font-size: 0.9rem;">
                                    {pred['score']:.2%}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                    st.markdown("#### 📊 Ensemble Method Details")
                    st.markdown(f"**Method:** {ensemble_result['method']}")
                    st.markdown(f"**Details:** {ensemble_result['details']}")
                    if 'model_contributions' in ensemble_result:
                        st.markdown("##### Model Contributions:")
                        for contrib in ensemble_result['model_contributions']:
                            st.write(f"- {contrib['model']}: Weight {contrib['weight']:.3f}, "
                                   f"Contribution: {contrib['contribution']:.3f}")
                    if st.checkbox("🔍 Show All Ensemble Methods Comparison"):
                        st.markdown("#### 🎯 All Methods Comparison")
                        all_results = st.session_state.ensemble_classifier.get_all_predictions(predictions)
                        comparison_data = []
                        for method_key, result in all_results.items():
                            comparison_data.append({
                                'Method': ENSEMBLE_METHODS[method_key]['name'],
                                'Icon': ENSEMBLE_METHODS[method_key]['icon'],
                                'Prediction': result['label'],
                                'Confidence': f"{result['confidence']:.2%}",
                                'Spam Prob': f"{result['spam_probability']:.2%}"
                            })
                        df_comparison = pd.DataFrame(comparison_data)
                        st.dataframe(df_comparison, use_container_width=True)
                else:
                    st.warning("No predictions could be generated from the ensemble models for this message.")
        else:
            st.error("No ensemble models were loaded successfully. Cannot perform ensemble analysis.")
    if 'features' in locals(): # Only show features if analysis was successful
        col_detail1, col_detail2 = st.columns(2)
        with col_detail1:
            st.markdown("#### 📋 Message Features")
            st.markdown(f"""
            <div class="feature-card">
                <strong>Length:</strong> {features['length']} characters<br>
                <strong>Words:</strong> {features['word_count']}<br>
                <strong>Uppercase:</strong> {features['uppercase_ratio']:.1%}<br>
                <strong>Numbers:</strong> {features['digit_ratio']:.1%}<br>
                <strong>Special chars:</strong> {features['special_chars']}
            </div>
            """, unsafe_allow_html=True)
        with col_detail2:
            st.markdown("#### ⚠️ Risk Indicators")
            if risk_indicators:
                for indicator in risk_indicators:
                    st.markdown(f"- {indicator}")
            else:
                st.markdown("✅ No significant risk indicators detected")

with col2:
    st.markdown("""
    <div class="analysis-header">
        <h3 style="color: #00d4aa; margin: 0;">📈 Analytics</h3>
    </div>
    """, unsafe_allow_html=True)

    # Analytics Section - Visuals
    
    if analysis_mode == "Single Model":
        st.markdown("#### 📊 Single Model Performance")
        
        # Check if there's any data for any model
        if any(st.session_state.model_stats[model]['total'] > 0 for model in MODEL_OPTIONS):
            # Pie Chart for Spam/Ham Distribution of the SELECTED model
            current_model_stats = st.session_state.model_stats[selected_model_name]
            if current_model_stats['total'] > 0:
                data_selected_model = pd.DataFrame({
                    'Label': ['SPAM', 'HAM'],
                    'Count': [current_model_stats['spam'], current_model_stats['ham']]
                })
                fig_pie_single = px.pie(
                    data_selected_model, 
                    values='Count', 
                    names='Label', 
                    title=f'Spam/Ham Distribution for {selected_model_name}',
                    color_discrete_map={'SPAM': '#ff6b6b', 'HAM': '#4ecdc4'}
                )
                fig_pie_single.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=300,
                    margin=dict(t=50, b=0, l=0, r=0) # Adjust margins
                )
                st.plotly_chart(fig_pie_single, use_container_width=True)
            else:
                st.info(f"No prediction data for {selected_model_name} yet.")

            # Confidence over time for the SELECTED model
            df_single_history = pd.DataFrame(st.session_state.classification_history)
            df_selected_model_history = df_single_history[df_single_history['model'] == selected_model_name].copy()
            if not df_selected_model_history.empty:
                df_selected_model_history['time_index'] = range(len(df_selected_model_history)) # Use index for X-axis
                fig_conf_single = px.line(
                    df_selected_model_history, 
                    x='time_index', 
                    y='confidence', 
                    title=f'Confidence Over Time ({selected_model_name})',
                    color_discrete_sequence=['#00d4aa']
                )
                fig_conf_single.update_layout(
                    xaxis_title="Prediction #",
                    yaxis_title="Confidence",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=250,
                    margin=dict(t=50, b=0, l=0, r=0)
                )
                st.plotly_chart(fig_conf_single, use_container_width=True)
            else:
                st.info(f"No confidence trend history for {selected_model_name} yet.")

            # Overall Model Usage (Bar chart)
            model_usage_data = []
            for model, stats in st.session_state.model_stats.items():
                model_usage_data.append({'Model': model, 'Total Predictions': stats['total']})
            df_model_usage = pd.DataFrame(model_usage_data)

            if not df_model_usage.empty and df_model_usage['Total Predictions'].sum() > 0:
                fig_model_usage = px.bar(
                    df_model_usage,
                    x='Model',
                    y='Total Predictions',
                    title='Total Predictions per Model (All Time)',
                    color='Model',
                    color_discrete_map={name: info['color'] for name, info in MODEL_OPTIONS.items()}
                )
                fig_model_usage.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=300,
                    margin=dict(t=50, b=0, l=0, r=0)
                )
                st.plotly_chart(fig_model_usage, use_container_width=True)
            else:
                st.info("No overall model usage data yet.")
        else:
            st.info("Run an analysis in 'Single Model' mode to see analytics.")

    else: # Ensemble Analysis
        st.markdown("#### 📊 Ensemble Performance")
        if st.session_state.ensemble_history:
            df_ensemble_history = pd.DataFrame(st.session_state.ensemble_history)

            # Pie Chart for Spam/Ham Distribution (Ensemble)
            ensemble_counts = df_ensemble_history['prediction'].value_counts().reset_index()
            ensemble_counts.columns = ['Label', 'Count']
            fig_pie_ensemble = px.pie(
                ensemble_counts, 
                values='Count', 
                names='Label', 
                title='Ensemble Spam/Ham Distribution',
                color_discrete_map={'SPAM': '#ff6b6b', 'HAM': '#4ecdc4'}
            )
            fig_pie_ensemble.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=300,
                margin=dict(t=50, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_pie_ensemble, use_container_width=True)

            # Confidence over time (Ensemble)
            fig_conf_ensemble = px.line(
                df_ensemble_history, 
                x=df_ensemble_history.index, # Use index for chronological order
                y='confidence', 
                title='Ensemble Confidence Over Time',
                color_discrete_sequence=['#a855f7']
            )
            fig_conf_ensemble.update_layout(
                xaxis_title="Prediction #",
                yaxis_title="Confidence",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=250,
                margin=dict(t=50, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_conf_ensemble, use_container_width=True)

            # Ensemble Method Usage (Bar chart)
            method_usage_data = df_ensemble_history['method'].value_counts().reset_index()
            method_usage_data.columns = ['Method Key', 'Count']
            # Map method keys to display names
            method_usage_data['Method'] = method_usage_data['Method Key'].apply(lambda x: ENSEMBLE_METHODS.get(x, {}).get('name', x))
            
            fig_method_usage = px.bar(
                method_usage_data,
                x='Method',
                y='Count',
                title='Ensemble Method Usage',
                color='Method',
                color_discrete_map={info['name']: info['color'] for name, info in ENSEMBLE_METHODS.items()}
            )
            fig_method_usage.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=300,
                margin=dict(t=50, b=0, l=0, r=0)
            )
            st.plotly_chart(fig_method_usage, use_container_width=True)

        else:
            st.info("No ensemble prediction history yet. Run an analysis to see stats.")


# --- Bulk CSV Classification Section (Drag & Drop) ---
st.markdown("---")
st.subheader("📂 Drag & Drop CSV for Bulk Classification")

uploaded_csv = st.file_uploader(
    "Upload a CSV file with a 'message' column:",
    type=["csv"],
    accept_multiple_files=False
)

def classify_csv(file, ensemble_mode, selected_models_for_bulk, selected_ensemble_method_for_bulk):
    try:
        df = pd.read_csv(file)
        if 'message' not in df.columns:
            st.error("CSV file must contain a 'message' column.")
            return None
        
        # Load models once for bulk classification
        if ensemble_mode:
            models_to_use = load_all_models() # Loads all models
        else:
            # For single model, load only the selected one
            models_to_use = {selected_models_for_bulk: load_model_if_needed(selected_models_for_bulk)}
            
        if not any(models_to_use.values()):
            st.error("No models loaded for classification. Please check model loading status.")
            return None

        results = []
        progress_text = "Operation in progress. Please wait."
        bulk_progress_bar = st.progress(0, text=progress_text)

        for i, message in enumerate(df['message']):
            current_progress = (i + 1) / len(df)
            bulk_progress_bar.progress(current_progress, text=f"Classifying message {i+1}/{len(df)}...")

            if ensemble_mode:
                predictions = get_ensemble_predictions(str(message), models_to_use)
                if predictions: # Ensure there are predictions
                    ensemble_result = st.session_state.ensemble_classifier.get_ensemble_prediction(
                        predictions, selected_ensemble_method_for_bulk
                    )
                    results.append({
                        'message': message,
                        'prediction': ensemble_result['label'],
                        'confidence': ensemble_result['confidence'],
                        'spam_probability': ensemble_result['spam_probability']
                    })
                else:
                    results.append({ # Fallback for no predictions from ensemble models
                        'message': message,
                        'prediction': 'ERROR',
                        'confidence': 0.0,
                        'spam_probability': 0.0
                    })
            else: # Single Model
                model_name = selected_models_for_bulk # This will be just one model name
                classifier = models_to_use.get(model_name)
                if classifier:
                    result = classifier(str(message))[0]
                    results.append({
                        'message': message,
                        'prediction': result['label'].upper(),
                        'confidence': result['score']
                    })
                else:
                    results.append({ # Fallback if single classifier not loaded
                        'message': message,
                        'prediction': 'ERROR',
                        'confidence': 0.0
                    })
        bulk_progress_bar.empty() # Clear progress bar after completion
        df_results = pd.DataFrame(results)
        return df_results
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")
        return None

ensemble_mode_bulk = analysis_mode == "Ensemble Analysis" 
if ensemble_mode_bulk:
    selected_models_for_bulk = list(MODEL_OPTIONS.keys())
    # Ensure selected_ensemble_method is defined if in ensemble mode, fallback to majority_voting
    selected_ensemble_method_for_bulk = selected_ensemble_method if 'selected_ensemble_method' in locals() else 'majority_voting'
else:
    selected_models_for_bulk = selected_model_name
    selected_ensemble_method_for_bulk = None # Not applicable for single model

if uploaded_csv is not None:
    st.info("Initiating bulk classification. This might take a while for large files, depending on model loading status.")
    with st.spinner("Classifying messages..."):
        df_results = classify_csv(uploaded_csv, ensemble_mode_bulk, selected_models_for_bulk, selected_ensemble_method_for_bulk)
        if df_results is not None:
            st.success("Bulk classification complete!")
            st.write("### Classification Results")
            st.dataframe(df_results)
            csv_buffer = StringIO()
            df_results.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_buffer.getvalue(),
                file_name="spam_predictions.csv",
                mime="text/csv"
            )

# --- Recent Classifications (Always visible if data exists) ---
st.markdown("---") # Add a separator

if analysis_mode == "Single Model" and st.session_state.classification_history:
    st.markdown("#### 🕒 Recent Single Model Classifications")
    recent = st.session_state.classification_history[-5:] # Show last 5
    
    for item in reversed(recent):
        status_color = "#ff6b6b" if item['prediction'] == "SPAM" else "#4ecdc4"
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 3px solid {status_color};">
            <strong style="color: {status_color};">{item['prediction']}</strong> ({item['confidence']:.1%})<br>
            <small style="color: #888;">{item['message']}</small><br>
            <small style="color: #666;">{item['model']} • {item['timestamp'].strftime('%H:%M')}</small>
        </div>
        """, unsafe_allow_html=True)
    # Single Model export button
    export_results_button(st.session_state.classification_history, filename_prefix="spamlyser_singlemodel")

elif analysis_mode == "Ensemble Analysis" and st.session_state.ensemble_history:
    st.markdown("#### 🕒 Recent Ensemble Results")
    recent = st.session_state.ensemble_history[-5:] # Show last 5
    
    for item in reversed(recent):
        status_color = "#ff6b6b" if item['prediction'] == "SPAM" else "#4ecdc4"
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 3px solid {status_color};">
            <strong style="color: {status_color};">{item['prediction']}</strong> ({item['confidence']:.1%})<br>
            <small style="color: #888;">{item['message']}</small><br>
            <small style="color: #666;">{ENSEMBLE_METHODS[item['method']]['name']} • {item['timestamp'].strftime('%H:%M')}</small>
        </div>
        """, unsafe_allow_html=True)
    export_results_button(st.session_state.ensemble_history, filename_prefix="spamlyser_ensemble")

    # Ensemble performance chart (Only show if enough data for a meaningful chart)
    if len(st.session_state.ensemble_history) > 3:
        st.markdown("#### 📊 Ensemble Confidence Trend")
        df_ensemble = pd.DataFrame(st.session_state.ensemble_history)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(df_ensemble))),
            y=df_ensemble['confidence'],
            mode='lines+markers',
            name='Confidence',
            line=dict(color='#00d4aa', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Ensemble Confidence Over Time", # More specific title
            xaxis_title="Analysis #",
            yaxis_title="Confidence",
            height=250,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            margin=dict(t=50, b=0, l=0, r=0)
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; background: rgba(255,255,255,0.02); border-radius: 10px; margin-top: 30px;">
    <p style="color: #888; margin: 0;">
        🛡️ <strong>Spamlyser Pro - Ensemble Edition</strong> | Advanced Multi-Model SMS Threat Detection<br>
        Powered by Custom-Trained Transformer Models & Ensemble Learning<br>
        Developed by <a href="https://eccentriccoder01.github.io/Me" target="_blank" style="color: #1f77b4; text-decoration: none; font-weight: 600;">MrEccentric</a>
    </p>
    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1);">
        <small style="color: #666;">
            🎯 Features: Single Model Analysis | 🤖 Ensemble Methods | 📊 Performance Tracking | ⚖️ Adaptive Weights
        </small>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Advanced Features Section ---
if analysis_mode == "Ensemble Analysis":
    st.markdown("---")
    st.markdown("## 🔧 Advanced Ensemble Settings")

    col_advanced1, col_advanced2 = st.columns(2)

    with col_advanced1:
        st.markdown("### 📊 Model Performance Tracking")

        if st.button("📈 View Model Performance Stats"):
            tracker_stats = st.session_state.ensemble_tracker.get_all_stats()
            if any(stats for stats in tracker_stats.values()):
                for model_name, stats in tracker_stats.items():
                    if stats:
                        st.markdown(f"#### {MODEL_OPTIONS[model_name]['icon']} {model_name}")
                        st.write(f"- **Accuracy:** {stats.get('accuracy', 'N/A'):.2%}")
                        st.write(f"- **Total Predictions:** {stats.get('total_predictions', 0)}")
                        st.write(f"- **Trend:** {stats.get('performance_trend', 'N/A')}")
                        st.write(f"- **Current Weight:** {stats.get('current_weight', 0):.3f}")
                        st.markdown("---")
            else:
                st.info("No performance data available yet. Make some predictions to see stats!")

        if st.button("💾 Export Performance Data"):
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"spamlyser_performance_{timestamp}.json"
                st.session_state.ensemble_tracker.save_to_file(filename)
                st.success(f"Performance data exported to {filename}")
            except Exception as e:
                st.error(f"Error exporting data: {str(e)}")

    with col_advanced2:
        st.markdown("### ⚙️ Ensemble Configuration")

        # Display current weights
        current_weights = st.session_state.ensemble_classifier.get_model_weights()
        st.markdown("#### Current Model Weights:")
        for model, weight in current_weights.items():
            st.write(f"- {MODEL_OPTIONS[model]['icon']} {model}: {weight:.3f}")

        # Reset to default weights
        if st.button("🔄 Reset to Default Weights"):
            st.session_state.ensemble_classifier.update_model_weights(
                st.session_state.ensemble_classifier.default_weights
            )
            st.success("Weights reset to default values!")
            st.rerun()

# --- Real-time Ensemble Method Performance Comparison (Always visible if data exists) ---
if analysis_mode == "Ensemble Analysis" and st.session_state.ensemble_history:
    st.markdown("---")
    st.markdown("## 📊 Ensemble Method Performance Comparison")

    # Create comparison chart of different methods
    method_performance = defaultdict(list)
    for entry in st.session_state.ensemble_history:
        method_performance[entry['method']].append(entry['confidence'])

    if len(method_performance) > 1:
        comparison_data = []
        for method, confidences in method_performance.items():
            comparison_data.append({
                'Method': ENSEMBLE_METHODS[method]['name'],
                'Avg Confidence': np.mean(confidences),
                'Std Dev': np.std(confidences),
                'Count': len(confidences)
            })

        df_comparison = pd.DataFrame(comparison_data)

        # Create bar chart
        fig = px.bar(
            df_comparison, 
            x='Method', 
            y='Avg Confidence',
            title='Average Confidence by Ensemble Method',
            color='Avg Confidence',
            color_continuous_scale='viridis'
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400,
            margin=dict(t=50, b=0, l=0, r=0)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show detailed comparison table
        st.dataframe(df_comparison, use_container_width=True)
    else:
        st.info("Not enough data to compare ensemble methods. Try more predictions with different methods.")
