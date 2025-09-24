from typing import Dict, Any, Optional, List, Union, TypedDict, Callable
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
feature/voice-to-sms
import json
from datetime import datetime
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from collections import defaultdict
from io import StringIO

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timedelta

# Assuming these files exist in the same directory
from models.export_feature import export_results_button 
from models.threat_analyzer import classify_threat_type, get_threat_specific_advice, THREAT_CATEGORIES
from models.word_analyzer import WordAnalyzer
# from ensemble_classifier_method import EnsembleSpamClassifier, ModelPerformanceTracker, PredictionResult
# Dummy classes if you don't have the actual files for testing
try:
    from models.ensemble_classifier_method import EnsembleSpamClassifier, ModelPerformanceTracker, PredictionResult
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

# Custom type definitions
class MethodStats(TypedDict):
    count: int
    spam: int
    confidences: List[float]

class ModelStats(TypedDict):
    spam: int
    ham: int
    total: int

# Session state type definition
class SpamlyserSessionState:
    """Type-safe session state container for Spamlyser"""
    def __init__(self):
        self.classification_history: List[Dict[str, Any]] = []
        self.ensemble_history: List[Dict[str, Any]] = []
        self.model_stats: Dict[str, ModelStats] = {}
        self.ensemble_tracker = None
        self.ensemble_classifier = None
        self.loaded_models: Dict[str, Any] = {}

# Import app core and ensemble functionality
import ensemble_classifier_method as ecm
from ensemble_classifier_method import ModelPerformanceTracker, EnsembleSpamClassifier

# Import other local modules
from export_feature import export_results_button
from speech_handler import SpeechHandler
from speech_to_text import SpeechToText

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

# Custom type for session state
class SessionState:
    def __init__(self):
        self.classification_history: List[Dict[str, Any]] = []
        self.model_stats: Dict[str, ModelStats] = {}
        self.ensemble_tracker: ModelPerformanceTracker = ModelPerformanceTracker()
        self.ensemble_classifier: EnsembleSpamClassifier = EnsembleSpamClassifier(performance_tracker=self.ensemble_tracker)
        self.ensemble_history: List[Dict[str, Any]] = []
        self.loaded_models: Dict[str, Any] = {}
        self.speech_handler: SpeechHandler = SpeechHandler()

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

# --- Session State Initialization ---
def init_session_state():
    """Initialize session state with proper typing"""
    if 'classification_history' not in st.session_state:
        st.session_state.classification_history = []
    if 'ensemble_history' not in st.session_state:
        st.session_state.ensemble_history = []
    if 'model_stats' not in st.session_state:
        st.session_state.model_stats = {
            model: {'spam': 0, 'ham': 0, 'total': 0}
            for model in MODEL_OPTIONS.keys()
        }
    if 'ensemble_tracker' not in st.session_state:
        st.session_state.ensemble_tracker = ecm.ModelPerformanceTracker()
    if 'ensemble_classifier' not in st.session_state:
        st.session_state.ensemble_classifier = ecm.EnsembleSpamClassifier(
            performance_tracker=st.session_state.ensemble_tracker
        )
    if 'loaded_models' not in st.session_state:
        st.session_state.loaded_models = {
            model_name: None for model_name in MODEL_OPTIONS.keys()
        }
    if 'speech_handler' not in st.session_state:
        st.session_state.speech_handler = SpeechHandler()

init_session_state()

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

    # --- Initialize Session State ---
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
        st.session_state.classification_history = []
        st.session_state.ensemble_history = []
        st.session_state.model_stats = {}
        tracker = ecm.ModelPerformanceTracker()
        st.session_state.ensemble_tracker = tracker
        st.session_state.ensemble_classifier = ecm.EnsembleSpamClassifier(
            performance_tracker=tracker
        )
        st.session_state.loaded_models = {}
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(145deg, #1e1e1e, #2a2a2a); border-radius: 15px; margin-bottom: 20px;">
        <h3 style="color: #00d4aa; margin: 0;">Analysis Mode</h3>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.dark_mode = st.checkbox("🌙 Enable Dark Mode", value=st.session_state.dark_mode, help="Toggle dark mode for the app")

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
            base_threshold = st.slider("Base Threshold", 0.1, 0.9, 0.5, 0.05)

    st.markdown("---")

    # Sidebar Overall Stats
    st.markdown("### 📊 Overall Statistics")

    # Safely calculate statistics with proper error handling
    try:
        # Ensure session state is properly initialized
        if not hasattr(st.session_state, 'model_stats'):
            st.session_state.model_stats = {model: {'spam': 0, 'ham': 0, 'total': 0} for model in MODEL_OPTIONS.keys()}
        if not hasattr(st.session_state, 'ensemble_history'):
            st.session_state.ensemble_history = []

        total_single_predictions = sum(
            st.session_state.model_stats.get(model, {}).get('total', 0)
            for model in MODEL_OPTIONS.keys()
        )
        total_ensemble_predictions = len(st.session_state.ensemble_history)
        total_predictions_overall = total_single_predictions + total_ensemble_predictions

        st.markdown(f"""
        <div class="metric-container">
            <p style="color: #00d4aa; font-size: 1.2rem; margin-bottom: 5px;">Total Predictions</p>
            <h3 style="color: #eee; margin-top: 0;">{total_predictions_overall}</h3>
        </div>
        """, unsafe_allow_html=True)

        overall_spam_count = sum(
            st.session_state.model_stats.get(model, {}).get('spam', 0)
            for model in MODEL_OPTIONS.keys()
        ) + sum(1 for entry in st.session_state.ensemble_history if entry['prediction'] == 'SPAM')

        overall_ham_count = sum(
            st.session_state.model_stats.get(model, {}).get('ham', 0)
            for model in MODEL_OPTIONS.keys()
        ) + sum(1 for entry in st.session_state.ensemble_history if entry['prediction'] == 'HAM')

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
    except (AttributeError, KeyError, TypeError):
        # Handle case where session state is not properly initialized
        st.markdown("""
        <div class="metric-container">
            <p style="color: #00d4aa; font-size: 1.2rem; margin-bottom: 5px;">Total Predictions</p>
            <h3 style="color: #eee; margin-top: 0;">0</h3>
        </div>
        """, unsafe_allow_html=True)

        col_spam, col_ham = st.columns(2)
        with col_spam:
            st.markdown("""
            <div class="metric-container spam-alert" style="padding: 15px;">
                <p style="color: #ff6b6b; font-size: 1rem; margin-bottom: 5px;">Spam Count</p>
                <h4 style="color: #ff6b6b; margin-top: 0;">0</h4>
            </div>
            """, unsafe_allow_html=True)
        with col_ham:
            st.markdown("""
            <div class="metric-container ham-safe" style="padding: 15px;">
                <p style="color: #6bff6b; font-size: 1rem; margin-bottom: 5px;">Ham Count</p>
                <h4 style="color: #6bff6b; margin-top: 0;">0</h4>
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
    """Load a model if not already loaded, with optional progress callback."""
    if model_name not in st.session_state.loaded_models or st.session_state.loaded_models[model_name] is None:
        if _progress_callback:
            _progress_callback(f"Loading {model_name} model...")

        model_id = MODEL_OPTIONS[model_name]["id"]
        st.session_state.loaded_models[model_name] = _load_model_cached(model_id)

        if _progress_callback:
            _progress_callback(f"✅ {model_name} model loaded successfully!")

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

# --- Dynamic CSS for Dark Mode ---
if st.session_state.get('dark_mode', False):
    st.markdown("""
    <style>
        .main, .stApp {
            background: #181f2f;
        }
        .metric-container, .prediction-card, .ensemble-card, .feature-card, .model-info, .ensemble-method, .method-comparison {
            background: #232a3d;
            border-radius: 16px;
            border: 1px solid #324a7c;
            color: #f8fafc;
            box-shadow: 0 2px 12px rgba(44, 62, 80, 0.08);
        }
        .spam-alert {
            background: #2a3350;
            border: 2px solid #ff4444;
            color: #ff6b6b;
        }
        .ham-safe {
            background: #233d2a;
            border: 2px solid #44ff44;
            color: #6bff6b;
        }
        .analysis-header {
            background: #232a3d;
            border-left: 4px solid #324a7c;
            color: #f8fafc;
        }
        /* Input fields and dropdowns */
        .stTextInput>div>input, .stTextArea>div>textarea, .stSelectbox>div>div>div {
            background: #232a3d !important;
            color: #f8fafc !important;
            border: 1px solid #324a7c !important;
        }
        .stTextInput>div>input::placeholder, .stTextArea>div>textarea::placeholder {
            color: #b3c7f7 !important;
        }
        /* Button styling */
        .stButton>button {
            background: #324a7c;
            color: #f8fafc;
            border-radius: 8px;
            border: none;
            box-shadow: 0 2px 8px rgba(44, 62, 80, 0.08);
        }
        .stButton>button:hover {
            background: #415a9c;
            color: #fff;
        }
        /* Label and text color for clarity */
        label, .stMarkdown, .stRadio>div>label, .stSelectbox label, .stTextInput label {
            color: #f8fafc !important;
        }
        /* Scrollbar styling for dark mode */
        ::-webkit-scrollbar {
            width: 8px;
            background: #232a3d;
        }
        ::-webkit-scrollbar-thumb {
            background: #324a7c;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        .main, .stApp {
            background: #f4f8ff;
        }
        .metric-container, .prediction-card, .ensemble-card, .feature-card, .model-info, .ensemble-method, .method-comparison {
            background: #e3eafc;
            border-radius: 16px;
            border: 1px solid #b3c7f7;
            color: #232a3d;
            box-shadow: 0 2px 12px rgba(44, 62, 80, 0.06);
        }
        .spam-alert {
            background: #ffe3e3;
            border: 2px solid #ff4444;
            color: #ff6b6b;
        }
        .ham-safe {
            background: #e3ffe3;
            border: 2px solid #44ff44;
            color: #6bff6b;
        }
        .analysis-header {
            background: #e3eafc;
            border-left: 4px solid #324a7c;
            color: #232a3d;
        }
        /* Scrollbar styling for light mode */
        ::-webkit-scrollbar {
            width: 8px;
            background: #e3eafc;
        }
        ::-webkit-scrollbar-thumb {
            background: #324a7c;
            border-radius: 8px;
        }
        /* Button styling */
        .stButton>button {
            background: #324a7c;
            color: #e3eafc;
            border-radius: 8px;
            border: none;
            box-shadow: 0 2px 8px rgba(44, 62, 80, 0.08);
        }
        .stButton>button:hover {
            background: #415a9c;
            color: #fff;
        }
    </style>
    """, unsafe_allow_html=True)
    # ...existing code...

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
# This creates a comprehensive dashboard using your existing session state data

def render_spamlyser_dashboard():
    """
    Advanced Analytics Dashboard - Add this function to your app.py
    Uses existing session state data: classification_history, ensemble_history, model_stats
    """
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin: 20px 0; border: 2px solid #8b5cf6;">
        <h1 style="color: white; font-size: 2.8rem; margin: 0; text-shadow: 0 0 20px rgba(255, 255, 255, 0.3);">
            📊 Advanced Analytics Dashboard
        </h1>
        <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin: 10px 0 0 0;">
            Real-time Performance Insights & Threat Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Dashboard tabs
    # Dashboard tabs
    dashboard_tabs = st.tabs(["🎯 Overview", "🤖 Model Performance", "🧠 Ensemble Analytics", "📊 Detailed Stats", "⚡ Real-time Monitor"])

    with dashboard_tabs[0]:  # Overview Tab
        render_overview_dashboard()
    
    with dashboard_tabs[1]:  # Model Performance Tab
        render_model_performance_dashboard()
    
    with dashboard_tabs[2]:  # Ensemble Analytics Tab
        render_ensemble_dashboard()
    
    with dashboard_tabs[3]:  # Detailed Stats Tab
        render_detailed_stats_dashboard()
    
    with dashboard_tabs[4]:  # Real-time Monitor Tab
        render_realtime_monitor()

def render_overview_dashboard():
    st.markdown("""
<style>
.metric-container {
    background: linear-gradient(145deg, #1e1e1e, #2a2a2a);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);

    /* 🔑 Force same size for all cards */
    min-height: 180px;
    max-height: 180px;
    min-width: 200px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.metric-container h2 {
    margin: 0;
    font-size: 2rem;
}
.metric-container p {
    margin: 6px 0;
    color: #ccc;
}
.metric-container small {
    color: #aaa;
}
</style>
""", unsafe_allow_html=True)
    
    # --- Calculate key metrics ---
    total_single = len(st.session_state.classification_history)
    total_ensemble = len(st.session_state.ensemble_history)
    total_messages = total_single + total_ensemble

    if total_messages == 0:
        st.info("🚀 Start analyzing messages to see dashboard insights!")
        return

    col1, col2, col3, col4, col5 = st.columns(5)

    # --- SPAM ---
    with col1:
        spam_single = sum(1 for item in st.session_state.classification_history if item['prediction'] == 'SPAM')
        spam_ensemble = sum(1 for item in st.session_state.ensemble_history if item['prediction'] == 'SPAM')
        total_spam = spam_single + spam_ensemble
        spam_rate = (total_spam / total_messages * 100) if total_messages > 0 else 0

        st.markdown(f"""
        <div class="metric-container" style="border: 2px solid #ff4444;">
            <h2 style="color: #ff6b6b;">🚨 {total_spam}</h2>
            <p>SPAM Detected</p>
            <small style="color: #ff9999;">{spam_rate:.1f}% detection rate</small>
        </div>
        """, unsafe_allow_html=True)

    # --- HAM ---
    with col2:
        total_ham = total_messages - total_spam
        ham_rate = (total_ham / total_messages * 100) if total_messages > 0 else 0

        st.markdown(f"""
        <div class="metric-container" style="border: 2px solid #44ff44;">
            <h2 style="color: #4ecdc4;">✅ {total_ham}</h2>
            <p>HAM (Safe)</p>
            <small style="color: #99ff99;">{ham_rate:.1f}% legitimate</small>
        </div>
        """, unsafe_allow_html=True)

    # --- Avg Confidence ---
    with col3:
        all_confidences = [item['confidence'] for item in st.session_state.classification_history] + \
                          [item['confidence'] for item in st.session_state.ensemble_history]
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0

        st.markdown(f"""
        <div class="metric-container" style="border: 2px solid #00d4aa;">
            <h2 style="color: #00d4aa;">🎯 {avg_confidence:.1%}</h2>
            <p>Avg Confidence</p>
            <small>Model certainty</small>
        </div>
        """, unsafe_allow_html=True)

    # --- Total Analyzed ---
    with col4:
        st.markdown(f"""
        <div class="metric-container" style="border: 2px solid #a855f7;">
            <h2 style="color: #a855f7;">📱 {total_messages}</h2>
            <p>Total Analyzed</p>
            <small>Messages processed</small>
        </div>
        """, unsafe_allow_html=True)

    # --- Preferred Mode ---
    with col5:
        mode_ratio = (total_ensemble / total_messages * 100) if total_messages > 0 else 0
        preferred_mode = "Ensemble" if total_ensemble > total_single else "Ensemble"

        st.markdown(f"""
        <div class="metric-container" style="border: 2px solid #ffd93d;">
            <h2 style="color: #ffd93d;">🧠 {preferred_mode}</h2>
            <p>Preferred Mode</p>
            <small>{mode_ratio:.0f}% ensemble usage</small>
        </div>
        """, unsafe_allow_html=True)
    # Threat Level Indicator
    st.markdown("### 🛡️ Current Threat Assessment")
    
    # Calculate threat level based on recent activity
    recent_items = (st.session_state.classification_history + st.session_state.ensemble_history)[-20:]
    if recent_items:
        recent_spam_count = sum(1 for item in recent_items if item['prediction'] == 'SPAM')
        recent_spam_ratio = recent_spam_count / len(recent_items)
        
        if recent_spam_ratio > 0.7:
            threat_level = "🔴 CRITICAL"
            threat_color = "#ff4444"
            threat_desc = "High spam activity detected"
        elif recent_spam_ratio > 0.5:
            threat_level = "🟠 HIGH"
            threat_color = "#ff8800"
            threat_desc = "Elevated spam levels"
        elif recent_spam_ratio > 0.3:
            threat_level = "🟡 MODERATE"
            threat_color = "#ffcc00"
            threat_desc = "Moderate spam activity"
        elif recent_spam_ratio > 0.1:
            threat_level = "🟢 LOW"
            threat_color = "#88cc00"
            threat_desc = "Low spam activity"
        else:
            threat_level = "🔵 MINIMAL"
            threat_color = "#4ecdc4"
            threat_desc = "Very low threat level"
        
        threat_col1, threat_col2 = st.columns([2, 3])
        
        with threat_col1:
            st.markdown(f"""
            <div style="background: linear-gradient(145deg, #1a1a1a, #2d2d2d); padding: 25px; border-radius: 15px; border: 3px solid {threat_color}; text-align: center;">
                <h2 style="color: {threat_color}; margin: 0; font-size: 2rem;">{threat_level}</h2>
                <p style="color: #ccc; margin: 10px 0;">{threat_desc}</p>
                <small style="color: #888;">Based on last {len(recent_items)} messages</small>
            </div>
            """, unsafe_allow_html=True)
        
        with threat_col2:
            # Recent activity timeline
            if len(recent_items) >= 5:
                timeline_data = []
                for i, item in enumerate(recent_items[-10:]):  # Last 10 items
                    timeline_data.append({
                        'Index': i+1,
                        'Prediction': 1 if item['prediction'] == 'SPAM' else 0,
                        'Confidence': item['confidence'],
                        'Type': 'SPAM' if item['prediction'] == 'SPAM' else 'HAM'
                    })
                
                fig_timeline = go.Figure()
                
                # Add spam/ham indicators
                spam_data = [item for item in timeline_data if item['Type'] == 'SPAM']
                ham_data = [item for item in timeline_data if item['Type'] == 'HAM']
                
                if spam_data:
                    fig_timeline.add_trace(go.Scatter(
                        x=[item['Index'] for item in spam_data],
                        y=[1 for _ in spam_data],
                        mode='markers',
                        marker=dict(color='#ff6b6b', size=12, symbol='triangle-up'),
                        name='SPAM',
                        text=[f"Confidence: {item['Confidence']:.1%}" for item in spam_data]
                    ))
                
                if ham_data:
                    fig_timeline.add_trace(go.Scatter(
                        x=[item['Index'] for item in ham_data],
                        y=[0 for _ in ham_data],
                        mode='markers',
                        marker=dict(color='#4ecdc4', size=12, symbol='circle'),
                        name='HAM',
                        text=[f"Confidence: {item['Confidence']:.1%}" for item in ham_data]
                    ))
                
                fig_timeline.update_layout(
                    title="Recent Activity Timeline",
                    xaxis_title="Message Sequence",
                    yaxis=dict(tickvals=[0, 1], ticktext=['HAM', 'SPAM']),
                    height=300,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    showlegend=True
                )
                
                st.plotly_chart(fig_timeline, use_container_width=True)

def render_model_performance_dashboard():
    """Individual model performance analysis"""
    
    if not st.session_state.model_stats or all(stats['total'] == 0 for stats in st.session_state.model_stats.values()):
        st.info("🤖 No single model data available. Try the Single Model analysis mode!")
        return
    
    st.markdown("### 🎯 Individual Model Performance")
    
    # Model comparison charts
    model_names = []
    spam_counts = []
    ham_counts = []
    total_counts = []
    colors = []
    
    for model_name, stats in st.session_state.model_stats.items():
        if stats['total'] > 0:
            model_names.append(model_name)
            spam_counts.append(stats['spam'])
            ham_counts.append(stats['ham'])
            total_counts.append(stats['total'])
            colors.append(MODEL_OPTIONS[model_name]['color'])
    
    if model_names:
        col1, col2 = st.columns(2)
        
        with col1:
            # Stacked bar chart
            fig_models = go.Figure()
            fig_models.add_trace(go.Bar(name='SPAM', x=model_names, y=spam_counts, marker_color='#ff6b6b'))
            fig_models.add_trace(go.Bar(name='HAM', x=model_names, y=ham_counts, marker_color='#4ecdc4'))
            
            fig_models.update_layout(
                title='Model Predictions Breakdown',
                barmode='stack',
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_models, use_container_width=True)
        
        with col2:
            # Model usage pie chart
            fig_usage = go.Figure(data=[go.Pie(
                labels=[f"{MODEL_OPTIONS[name]['icon']} {name}" for name in model_names],
                values=total_counts,
                marker_colors=colors,
                hole=.3
            )])
            
            fig_usage.update_layout(
                title="Model Usage Distribution",
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig_usage, use_container_width=True)
        
        # Detailed model stats table
        st.markdown("### 📊 Detailed Model Statistics")
        
        model_stats_data = []
        for model_name, stats in st.session_state.model_stats.items():
            if stats['total'] > 0:
                spam_rate = (stats['spam'] / stats['total'] * 100) if stats['total'] > 0 else 0
                model_stats_data.append({
                    'Model': f"{MODEL_OPTIONS[model_name]['icon']} {model_name}",
                    'Total Predictions': stats['total'],
                    'SPAM Detected': stats['spam'],
                    'HAM Detected': stats['ham'],
                    'SPAM Rate': f"{spam_rate:.1f}%",
                    'Description': MODEL_OPTIONS[model_name]['description']
                })
        
        if model_stats_data:
            df_model_stats = pd.DataFrame(model_stats_data)
            st.dataframe(df_model_stats, use_container_width=True)

def render_ensemble_dashboard():
    """Ensemble methods performance analysis"""
    
    if not st.session_state.ensemble_history:
        st.info("🧠 No ensemble data available. Try the Ensemble Analysis mode!")
        return
    
    st.markdown("### 🧠 Ensemble Method Analytics")
    
    # Analyze ensemble history
    method_stats: Dict[str, MethodStats] = {}
    for method in ENSEMBLE_METHODS.keys():
        method_stats[method] = MethodStats(
            count=0,
            spam=0,
            confidences=[]
        )
    
    for item in st.session_state.ensemble_history:
        method = item['method']
        if method in method_stats:
            method_stats[method]['count'] += 1
            method_stats[method]['confidences'].append(item['confidence'])
            if item['prediction'] == 'SPAM':
                method_stats[method]['spam'] += 1
    
    if method_stats:
        col1, col2 = st.columns(2)
        
        with col1:
            # Method usage and performance
            methods = list(method_stats.keys())
            method_counts = [method_stats[method]['count'] for method in methods]
            avg_confidences = [sum(method_stats[method]['confidences'])/len(method_stats[method]['confidences']) for method in methods]
            
            fig_methods = go.Figure()
            
            # Bar chart for usage
            fig_methods.add_trace(go.Bar(
                name='Usage Count',
                x=methods,
                y=method_counts,
                yaxis='y',
                marker_color='#00d4aa',
                opacity=0.7
            ))
            
            # Line chart for average confidence
            fig_methods.add_trace(go.Scatter(
                name='Avg Confidence',
                x=methods,
                y=[conf * max(method_counts) for conf in avg_confidences],  # Scale for visibility
                yaxis='y2',
                mode='lines+markers',
                marker_color='#ff6b6b',
                line=dict(width=3)
            ))
            
            fig_methods.update_layout(
                title='Ensemble Method Performance',
                yaxis=dict(title='Usage Count', side='left'),
                yaxis2=dict(title='Avg Confidence (Scaled)', side='right', overlaying='y'),
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig_methods, use_container_width=True)
        
        with col2:
            # Ensemble method comparison table
            ensemble_data = []
            for method, stats in method_stats.items():
                avg_conf = sum(stats['confidences']) / len(stats['confidences'])
                spam_rate = (stats['spam'] / stats['count'] * 100) if stats['count'] > 0 else 0
                
                # Get method info from your ENSEMBLE_METHODS dict
                method_info = ENSEMBLE_METHODS.get(method, {'name': method, 'icon': '🔧'})
                
                ensemble_data.append({
                    'Method': f"{method_info['icon']} {method_info['name'][:15]}",
                    'Uses': stats['count'],
                    'Avg Confidence': f"{avg_conf:.1%}",
                    'SPAM Rate': f"{spam_rate:.1f}%",
                    'Total SPAM': stats['spam']
                })
            
            df_ensemble = pd.DataFrame(ensemble_data)
            st.dataframe(df_ensemble, use_container_width=True)
            
            # Best performing method highlight
            if ensemble_data:
                best_method = max(ensemble_data, key=lambda x: float(x['Avg Confidence'].rstrip('%'))/100)
                st.markdown(f"""
                <div style="background: linear-gradient(145deg, #1a2a3a, #2a3a4a); padding: 15px; border-radius: 10px; border: 2px solid #00d4aa; margin: 15px 0;">
                    <h4 style="color: #00d4aa; margin: 0;">🏆 Top Performer</h4>
                    <p style="color: #ccc; margin: 5px 0;">{best_method['Method']} - {best_method['Avg Confidence']} confidence</p>
                </div>
                """, unsafe_allow_html=True)

def render_detailed_stats_dashboard():
    """Detailed statistical analysis"""
    
    st.markdown("### 📊 Detailed Statistical Analysis")
    
    all_data = st.session_state.classification_history + st.session_state.ensemble_history
    
    if not all_data:
        st.info("📈 No data available for detailed analysis.")
        return
    
    # Confidence distribution analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Confidence Distribution")
        
        confidences = [item['confidence'] for item in all_data]
        
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=confidences,
            nbinsx=25,
            marker_color='rgba(0, 212, 170, 0.7)',
            marker_line_color='rgba(0, 212, 170, 1)',
            marker_line_width=1,
            name='Confidence Distribution'
        ))
        
        # Add statistical lines
        mean_conf = np.mean(confidences)
        fig_hist.add_vline(x=mean_conf, line_dash="dash", line_color="red", 
                          annotation_text=f"Mean: {mean_conf:.2f}")
        
        fig_hist.update_layout(
            title="Model Confidence Distribution",
            xaxis_title="Confidence Score",
            yaxis_title="Frequency",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=350
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        st.markdown("#### 🎯 Prediction Accuracy by Confidence")
        
        # Bin predictions by confidence ranges
        confidence_ranges = []
        accuracy_estimates = []
        
        for i in range(0, 100, 10):
            lower = i / 100
            upper = (i + 10) / 100
            range_data = [item for item in all_data if lower <= item['confidence'] < upper]
            
            if range_data:
                confidence_ranges.append(f"{i}-{i+10}%")
                # Mock accuracy calculation based on confidence (higher confidence = higher accuracy)
                accuracy_estimates.append(min(95, 60 + (i * 0.35)))
        
        if confidence_ranges:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Bar(
                x=confidence_ranges,
                y=accuracy_estimates,
                marker_color='rgba(255, 107, 107, 0.7)',
                name='Estimated Accuracy'
            ))
            
            fig_acc.update_layout(
                title="Accuracy by Confidence Range",
                xaxis_title="Confidence Range",
                yaxis_title="Estimated Accuracy %",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=350
            )
            
            st.plotly_chart(fig_acc, use_container_width=True)
    
    # Statistical summary
    st.markdown("#### 📋 Statistical Summary")
    
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    
    with summary_col1:
        confidences = [item['confidence'] for item in all_data]
        st.markdown(f"""
        <div class="feature-card">
            <h4 style="color: #00d4aa;">Confidence Statistics</h4>
            <p><strong>Mean:</strong> {np.mean(confidences):.3f}</p>
            <p><strong>Median:</strong> {np.median(confidences):.3f}</p>
            <p><strong>Std Dev:</strong> {np.std(confidences):.3f}</p>
            <p><strong>Min/Max:</strong> {np.min(confidences):.3f} / {np.max(confidences):.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with summary_col2:
        spam_predictions = [item for item in all_data if item['prediction'] == 'SPAM']
        ham_predictions = [item for item in all_data if item['prediction'] == 'HAM']
        
        st.markdown(f"""
        <div class="feature-card">
            <h4 style="color: #ff6b6b;">Classification Summary</h4>
            <p><strong>Total Messages:</strong> {len(all_data)}</p>
            <p><strong>SPAM Detected:</strong> {len(spam_predictions)}</p>
            <p><strong>HAM (Safe):</strong> {len(ham_predictions)}</p>
            <p><strong>SPAM Rate:</strong> {len(spam_predictions)/len(all_data)*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    
    with summary_col3:
        if spam_predictions and ham_predictions:
            spam_conf_avg = np.mean([item['confidence'] for item in spam_predictions])
            ham_conf_avg = np.mean([item['confidence'] for item in ham_predictions])
        else:
            spam_conf_avg = 0
            ham_conf_avg = 0
            
        st.markdown(f"""
        <div class="feature-card">
            <h4 style="color: #4ecdc4;">Confidence by Type</h4>
            <p><strong>SPAM Avg Conf:</strong> {spam_conf_avg:.3f}</p>
            <p><strong>HAM Avg Conf:</strong> {ham_conf_avg:.3f}</p>
            <p><strong>Confidence Gap:</strong> {abs(spam_conf_avg - ham_conf_avg):.3f}</p>
            <p><strong>Higher Conf:</strong> {"SPAM" if spam_conf_avg > ham_conf_avg else "HAM"}</p>
        </div>
        """, unsafe_allow_html=True)

def render_realtime_monitor():
    """Real-time monitoring dashboard"""
    
    st.markdown("### ⚡ Real-time System Monitor")
    
    # System status indicators
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)
    
    with status_col1:
        # Model loading status
        loaded_models = sum(1 for model in st.session_state.loaded_models.values() if model is not None)
        total_models = len(st.session_state.loaded_models)
        
        status_color = "#4ecdc4" if loaded_models == total_models else "#ff8800"
        st.markdown(f"""
        <div class="metric-container" style="border: 2px solid {status_color};">
            <h3 style="color: {status_color};">🤖 Models</h3>
            <h2 style="color: {status_color}; margin: 5px 0;">{loaded_models}/{total_models}</h2>
            <small style="color: #888;">Loaded & Ready</small>
        </div>
        """, unsafe_allow_html=True)
    
    with status_col2:
        # Ensemble system status
        ensemble_status = "ACTIVE" if st.session_state.ensemble_classifier else "INACTIVE"
        status_color = "#4ecdc4" if ensemble_status == "ACTIVE" else "#ff6b6b"
        
        st.markdown(f"""
        <div class="metric-container" style="border: 2px solid {status_color};">
            <h3 style="color: {status_color};">🧠 Ensemble</h3>
            <h2 style="color: {status_color}; margin: 5px 0;">{ensemble_status}</h2>
            <small style="color: #888;">System Status</small>
        </div>
        """, unsafe_allow_html=True)
    
    with status_col3:
        # Performance tracker status
        tracker_active = st.session_state.ensemble_tracker is not None
        status_color = "#4ecdc4" if tracker_active else "#ff6b6b"
        
        st.markdown(f"""
        <div class="metric-container" style="border: 2px solid {status_color};">
            <h3 style="color: {status_color};">📊 Tracker</h3>
            <h2 style="color: {status_color}; margin: 5px 0;">{"ON" if tracker_active else "OFF"}</h2>
            <small style="color: #888;">Performance Monitor</small>
        </div>
        """, unsafe_allow_html=True)
    
    with status_col4:
        # Memory usage (mock)
        memory_usage = 67  # Mock percentage
        status_color = "#4ecdc4" if memory_usage < 80 else "#ff8800" if memory_usage < 90 else "#ff6b6b"
        
        st.markdown(f"""
        <div class="metric-container" style="border: 2px solid {status_color};">
            <h3 style="color: {status_color};">💾 Memory</h3>
            <h2 style="color: {status_color}; margin: 5px 0;">{memory_usage}%</h2>
            <small style="color: #888;">System Usage</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Real-time controls
    st.markdown("#### ⚙️ Real-time Controls")
    
    control_col1, control_col2, control_col3 = st.columns(3)
    
    with control_col1:
        auto_refresh = st.checkbox("🔄 Auto-refresh Dashboard", key="dashboard_auto_refresh")
        if auto_refresh:
            refresh_interval = st.slider("Refresh Interval (seconds)", 5, 60, 10)
    
    with control_col2:
        if st.button("🧹 Clear All History", type="secondary"):
            if st.button("⚠️ Confirm Clear", type="primary"):
                st.session_state.classification_history = []
                st.session_state.ensemble_history = []
                st.session_state.model_stats = {model: {'spam': 0, 'ham': 0, 'total': 0} for model in MODEL_OPTIONS.keys()}
                st.success("✅ History cleared!")
                time.sleep(1)
                st.rerun()
    
    with control_col3:
        if st.button("💾 Export Dashboard Data"):
            # Create comprehensive export
            dashboard_data = {
                'classification_history': st.session_state.classification_history,
                'ensemble_history': st.session_state.ensemble_history,
                'model_stats': st.session_state.model_stats,
                'export_timestamp': datetime.now().isoformat(),
                'total_messages': len(st.session_state.classification_history) + len(st.session_state.ensemble_history)
            }
            
            json_data = json.dumps(dashboard_data, indent=2, default=str)
            
            st.download_button(
                label="📥 Download Dashboard Data (JSON)",
                data=json_data,
                file_name=f"spamlyser_dashboard_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# Add this to your main app.py file after your existing analysis section:

# --- ADD THE DASHBOARD SECTION ---
if st.sidebar.button("📊 Open Dashboard", key="open_dashboard", help="Open the advanced analytics dashboard"):
    st.session_state.show_dashboard = True

if st.session_state.get('show_dashboard', False):
    render_spamlyser_dashboard()
    
    if st.button("❌ Close Dashboard", key="close_dashboard"):
        st.session_state.show_dashboard = False
        st.rerun()  # Rerun to reset the state

def get_risk_indicators(message, prediction, threat_type=None):
    indicators = []
    spam_keywords = ['free', 'win', 'winner', 'congratulations', 'urgent', 'limited', 'offer', 'click', 'call now']
    found_keywords = [word for word in spam_keywords if word.lower() in message.lower()]
    
    if prediction == "SPAM":
        # Add threat-specific indicators and advice
        if threat_type and threat_type in THREAT_CATEGORIES:
            threat_info = THREAT_CATEGORIES[threat_type]
            indicators.append(f"{threat_info['icon']} {threat_type} detected: {threat_info['description']}")
            
            # Add threat-specific advice
            threat_advice = get_threat_specific_advice(threat_type)
            for advice in threat_advice:
                indicators.append(f"💡 {advice}")
    
    # General indicators (for all messages)
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

def create_predict_proba(classifier):
    """
    Creates a batch-processing prediction function for LIME.
    `classifier` is a Hugging Face pipeline object.
    """
    def predict_proba_batch(texts: List[str]) -> np.ndarray:
        # 1. Get predictions for the whole batch at once
        # The pipeline is highly optimized for this!
        predictions = classifier(texts, top_k=2) # Get probabilities for both classes

        results = []
        for pred_list in predictions:
            # 2. Create a dictionary for easy lookup of scores by label
            score_dict = {p['label'].upper(): p['score'] for p in pred_list}
            
            # 3. Get the score for SPAM, defaulting to 0.0 if not found
            spam_score = score_dict.get('SPAM', 0.0)
            
            # 4. LIME expects probabilities for all classes. Order is [HAM, SPAM]
            # The HAM score will be 1.0 - SPAM score
            results.append([1.0 - spam_score, spam_score])
            
        return np.array(results)
        
    return predict_proba_batch# --- Main Interface ---
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
    col_text, col_record = st.columns([4, 1])

    with col_text:
        user_sms = st.text_area(
            "Enter SMS message to analyse",
            value=user_sms_initial_value,
            height=120,
            placeholder="Type or paste your SMS message here...",
            help="Enter the SMS message you want to classify as spam or ham (legitimate)",
            key="sms_input"
        )
    
    with col_record:
        if st.button("🎤 Record Voice", type="primary", key="record_btn"):
            with st.spinner("🎤 Listening... Speak clearly"):
                try:
                    success, result = st.session_state.speech_handler.listen_and_transcribe()
                    if success:
                        # Get existing text and append new text
                        current_text = user_sms  # Use the widget value directly
                        if current_text and current_text.strip():
                            new_text = f"{current_text.strip()} {result}"
                        else:
                            new_text = result
                        # Update session state for persistence
                        st.session_state.user_sms_input_value = new_text
                        st.success("✅ Speech recognized!")
                        st.rerun()
                    else:
                        st.error(f"❌ {result}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    time.sleep(2)

    # Store current text_area value in session state for persistence
    st.session_state.user_sms_input_value = user_sms
    
    # Analysis controls
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        analyse_btn = st.button("🔍 Analyse Message", type="primary", use_container_width=True)
    with col_b:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    # Test word analysis button (always visible)
    if st.button("🔍 Word Analysis", key="test_word_analysis", type="primary"):
        st.markdown("### 🔍 Word Analysis")
        
        # Use the current message from the text area
        test_message = user_sms if user_sms.strip() else "Congratulations! You won a free prize, click now!"
        st.markdown(f"**Analyzing Message:** {test_message}")
        
        # Create word analyzer
        analyzer = WordAnalyzer()
        
        # Analyze the text
        with st.spinner("🔍 Analyzing message..."):
            analysis = analyzer.analyze_text(test_message)
        
        # Show the highlighted text
        st.markdown("#### 📝 Your Message with Word Analysis")
        st.markdown("**🔴 Red words** = Spam indicators | **🟢 Green words** = Ham indicators | **🟠 Orange words** = Suspicious patterns")
        
        highlighted_html = analyzer.create_highlighted_html(analysis)
        st.markdown(highlighted_html, unsafe_allow_html=True)
        
        # Show summary
        summary = analyzer.get_explanation_summary(analysis)
        
        # Always count neutral words as ham for clearer UI in HAM messages
        spam_count = len(summary['top_spam_words'])
        ham_count = len(summary['top_ham_words'])
        
        # For HAM messages, all non-spam words are considered ham indicators
        if analysis.get('predicted_class') == 'HAM':
            neutral_words = [w for w in analysis['words'] if not w.get('is_spammy', False) and not w.get('is_hammy', False)]
            ham_count += len(neutral_words)
            
            # Add neutral words to the top_ham_words list for visibility
            for word in neutral_words:
                if word['word'] not in [w['word'] for w in summary['top_ham_words']]:
                    summary['top_ham_words'].append({
                        'word': word['word'],
                        'influence': -0.2,  # Give it a small negative influence (ham)
                        'type': 'neutral-ham'
                    })
        
        st.success(f"✅ Analysis complete! Found {spam_count} spam indicators and {ham_count} ham indicators.")
        
        # Show more detailed breakdown
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Spam Indicators:**")
            if spam_count > 0:
                for word in summary['top_spam_words']:
                    influence = word.get('influence', 0.0)
                    st.markdown(f"🔴 **{word['word']}** (Score: {influence:.2f})")
            else:
                st.info("No spam indicators found")
                
        with col2:
            st.markdown("**Ham Indicators:**")
            if ham_count > 0:
                for word in summary['top_ham_words']:
                    influence = word.get('influence', 0.0)
                    # Make sure we use the absolute value for ham scores
                    st.markdown(f"🟢 **{word['word']}** (Score: {abs(influence):.2f})")
            else:
                st.info("No ham indicators found")
    
    if clear_btn:
        # Clear text area content
        st.session_state.user_sms_input_value=""
        
        if "sample_selector" in st.session_state:
            st.session_state.pop("sample_selector")
        

        st.rerun() # Rerun to update the UI with cleared values



if analyse_btn and user_sms.strip():
    if analysis_mode == "Single Model":
        from models.smart_preprocess import preprocess_message
        preprocessed = preprocess_message(user_sms)
        cleaned_sms = preprocessed["cleaned"]
        suspicious_features = preprocessed["suspicious"]
        classifier = load_model_if_needed(selected_model_name)
        if classifier is not None:
            with st.spinner(f"🤖 Analyzing with {selected_model_name}..."):
                time.sleep(0.5)
                result = classifier(cleaned_sms)[0]
                label = result['label'].upper()
                confidence = result['score']
feature/voice-to-sms
                # Safely update model statistics with proper error handling
                if selected_model_name not in st.session_state.model_stats:
                    st.session_state.model_stats[selected_model_name] = {'spam': 0, 'ham': 0, 'total': 0}


                
                # Store prediction results in session state for explanation
                st.session_state.user_sms = user_sms
                st.session_state.current_prediction_label = label
                st.session_state.current_prediction_confidence = confidence
                
                # Special case for DistilBERT - it sometimes misses obvious scams
                if selected_model_name == "DistilBERT" and label == "HAM":
                    text_lower = cleaned_sms.lower()
                    # Check for common money scam patterns
                    if any(pattern in text_lower for pattern in [
                            "won", "$", "cash", "prize", "claim", "click yes", 
                            "lottery", "winner", "congratulation"
                        ]):
                        # Check for combination patterns that are strong indicators of scams
                        if (("won" in text_lower or "win" in text_lower) and 
                            ("$" in text_lower or "cash" in text_lower or "prize" in text_lower)):
                            # Override the classification for this clear scam case
                            label = "SPAM"
                            confidence = max(confidence, 0.85)  # Boost confidence
                            st.info("💡 Scam pattern detected and corrected")
                
                # If SPAM, classify the threat type
                threat_type = None
                threat_confidence = 0.0
                threat_metadata = {}
                if label == "SPAM":
                    threat_type, threat_confidence, threat_metadata = classify_threat_type(
                        cleaned_sms, confidence
                    )
                
master
                st.session_state.model_stats[selected_model_name][label.lower()] += 1
                st.session_state.model_stats[selected_model_name]['total'] += 1
                st.session_state.classification_history.append({
                    'timestamp': datetime.now(),
                    'message': user_sms[:100] + "..." if len(user_sms) > 100 else user_sms, # Increased snippet length
                    'prediction': label,
                    'confidence': confidence,
                    'model': selected_model_name,
                    'preprocessed': cleaned_sms,
                    'suspicious_features': suspicious_features,
                    'threat_type': threat_type,
                    'threat_confidence': threat_confidence
                })
                features = analyse_message_features(cleaned_sms)
                risk_indicators = get_risk_indicators(cleaned_sms, label, threat_type)
                st.markdown("### 🎯 Classification Results")
                card_class = "spam-alert" if label == "SPAM" else "ham-safe"
                icon = "🚨" if label == "SPAM" else "✅"
                # Create prediction card with threat info if applicable
                threat_html = ""
                if label == "SPAM" and threat_type:
                    # Create the threat info section directly without using an f-string template
                    threat_info = THREAT_CATEGORIES.get(threat_type, {})
                    threat_icon = threat_info.get('icon', '⚠️')
                    threat_color = threat_info.get('color', '#ff6b6b')
                    threat_description = threat_info.get('description', 'Suspicious message')
                    
                    # Use st.markdown to create a separate HTML element for threat info
                    threat_html = f'<div style="margin-top: 15px; padding: 10px; border-radius: 10px; background: rgba(0,0,0,0.1);"><h4 style="margin: 0; color: {threat_color};">{threat_icon} {threat_type}</h4><p style="margin: 5px 0 0 0; opacity: 0.9;">{threat_description} (Confidence: {threat_confidence:.1%})</p></div>'
                    
                # Use st.markdown with proper escaping and unsafe_allow_html=True
                model_info_html = f"""
                <div class="prediction-card {card_class}">
                    <h2 style="margin: 0 0 15px 0;">{icon} {label}</h2>
                    <h3 style="margin: 0;">Confidence: {confidence:.2%}</h3>
                    <p style="margin: 15px 0 0 0; opacity: 0.8;">
                        Model: {selected_model_name} | Analysed: {datetime.now().strftime('%H:%M:%S')}
                    </p>
                </div>
                """
                st.markdown(model_info_html, unsafe_allow_html=True)
                
                # Display threat information separately if it exists
                if label == "SPAM" and threat_type:
                    st.markdown(threat_html, unsafe_allow_html=True)
                
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
                    
                    # If SPAM, classify the threat type
                    threat_type = None
                    threat_confidence = 0.0
                    threat_metadata = {}
                    if ensemble_result['label'] == "SPAM":
                        threat_type, threat_confidence, threat_metadata = classify_threat_type(
                            user_sms, ensemble_result['spam_probability']
                        )
                        # Add threat info to ensemble result
                        ensemble_result['threat_type'] = threat_type
                        ensemble_result['threat_confidence'] = threat_confidence
                        ensemble_result['metadata']['threat'] = threat_metadata
                    
                    st.session_state.ensemble_history.append({
                        'timestamp': datetime.now(),
                        'message': user_sms[:100] + "..." if len(user_sms) > 100 else user_sms, # Increased snippet length
                        'prediction': ensemble_result['label'],
                        'confidence': ensemble_result['confidence'],
                        'method': selected_ensemble_method,
                        'spam_probability': ensemble_result['spam_probability'],
                        'threat_type': threat_type,
                        'threat_confidence': threat_confidence
                    })
                    features = analyse_message_features(user_sms)
                    risk_indicators = get_risk_indicators(user_sms, ensemble_result['label'], threat_type)
                    st.markdown("### 🎯 Ensemble Classification Results")
                    card_class = "spam-alert" if ensemble_result['label'] == "SPAM" else "ham-safe"
                    icon = "🚨" if ensemble_result['label'] == "SPAM" else "✅"
                    # Create prediction card with threat info if applicable
                    threat_html = ""
                    if ensemble_result['label'] == "SPAM" and threat_type:
                        # Create the threat info section directly without using an f-string template
                        threat_info = THREAT_CATEGORIES.get(threat_type, {})
                        threat_icon = threat_info.get('icon', '⚠️')
                        threat_color = threat_info.get('color', '#ff6b6b')
                        threat_description = threat_info.get('description', 'Suspicious message')
                        
                        # Use a single-line string to avoid formatting issues
                        threat_html = f'<div style="margin-top: 15px; padding: 10px; border-radius: 10px; background: rgba(0,0,0,0.1);"><h4 style="margin: 0; color: {threat_color};">{threat_icon} {threat_type}</h4><p style="margin: 5px 0 0 0; opacity: 0.9;">{threat_description} (Confidence: {threat_confidence:.1%})</p></div>'
                    
                    # Use st.markdown with proper escaping and unsafe_allow_html=True
                    ensemble_info_html = f"""
                    <div class="prediction-card {card_class} ensemble-card">
                        <h2 style="margin: 0 0 15px 0;">{icon} {ensemble_result['label']}</h2>
                        <h3 style="margin: 0;">Confidence: {ensemble_result['confidence']:.2%}</h3>
                        <h4 style="margin: 10px 0;">Spam Probability: {ensemble_result['spam_probability']:.2%}</h4>
                        <p style="margin: 15px 0 0 0; opacity: 0.8;">
                            Method: {ENSEMBLE_METHODS[selected_ensemble_method]['name']} | 
                            Analysed: {datetime.now().strftime('%H:%M:%S')}
                        </p>
                    </div>
                    """
                    st.markdown(ensemble_info_html, unsafe_allow_html=True)
                    
                    # Display threat information separately if it exists
                    if ensemble_result['label'] == "SPAM" and threat_type:
                        st.markdown(threat_html, unsafe_allow_html=True)
                    
                    st.markdown("#### 🤖 Individual Model Predictions")
                    cols = st.columns(len(predictions))
                    for i, (model_name, pred) in enumerate(predictions.items()):
                        # Save individual model prediction to a global tracking list
                        if 'model_vote_history' not in st.session_state:
                            st.session_state.model_vote_history = []

                        st.session_state.model_vote_history.append({
                            'model': model_name,
                            'label': pred['label'],
                            'confidence': pred['score'],
                            'message': user_sms,
                            'timestamp': datetime.now()
                        })

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
        if any(st.session_state.model_stats.get(model, {}).get('total', 0) > 0 for model in MODEL_OPTIONS.keys()):
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

            st.markdown("#### 🧠 Ensemble Spam/Ham Distribution")

            # Display vote pie chart
            if 'model_vote_history' in st.session_state and st.session_state.model_vote_history:
                df_votes = pd.DataFrame(st.session_state.model_vote_history)

                vote_counts = df_votes['label'].value_counts().to_dict()
                vote_counts_fixed = {
                    'SPAM': vote_counts.get('SPAM', 0),
                    'HAM': vote_counts.get('HAM', 0)
                }

                df_vote_chart = pd.DataFrame(list(vote_counts_fixed.items()), columns=['Label', 'Count'])

                fig_model_votes = px.pie(
                    df_vote_chart,
                    values='Count',
                    names='Label',
                    title='Individual Model Votes Distribution',
                    color_discrete_map={'SPAM': '#ff6b6b', 'HAM': '#4ecdc4'}
                )

                fig_model_votes.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=300,
                    margin=dict(t=50, b=0, l=0, r=0)
                )

                st.plotly_chart(fig_model_votes, use_container_width=True)
            else:
                st.info("No individual model votes recorded yet. Run some ensemble predictions first.")

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
st.subheader("📂 Drag & Drop CSV for Bulk Classification")

uploaded_csv = st.file_uploader(
    "Upload a CSV file with a 'message' column:",
    type=["csv"],
    accept_multiple_files=False
)

def classify_csv(file, ensemble_mode, selected_models_for_bulk, selected_ensemble_method_for_bulk, batch_size=100):
    try:
        df = pd.read_csv(file)
        if 'message' not in df.columns:
            st.error("CSV file must contain a 'message' column.")
            return None

        total_messages = len(df)
        results = []

        # ✅ Load models only once
        if ensemble_mode:
            models_to_use = load_all_models()
        else:
            models_to_use = {selected_models_for_bulk: load_model_if_needed(selected_models_for_bulk)}

        if not any(models_to_use.values()):
            st.error("No models loaded for classification. Please check model loading status.")
            return None

        # ✅ Progress + ETA text
        progress_bar = st.progress(0)
        status_text = st.empty()

        start_time = time.time()

        # Process in batches
        for start in range(0, total_messages, batch_size):
            end = min(start + batch_size, total_messages)
            batch_messages = df['message'][start:end].astype(str).tolist()

            try:
                if ensemble_mode:
                    # batch predictions from multiple models
                    batch_results = []
                    for msg in batch_messages:
                        predictions = get_ensemble_predictions(msg, models_to_use)
                        if predictions:
                            ensemble_result = st.session_state.ensemble_classifier.get_ensemble_prediction(
                                predictions, selected_ensemble_method_for_bulk
                            )
                            batch_results.append({
                                'message': msg,
                                'prediction': ensemble_result['label'],
                                'confidence': ensemble_result['confidence'],
                                'spam_probability': ensemble_result['spam_probability']
                            })
                        else:
                            batch_results.append({'message': msg, 'prediction': 'ERROR', 'confidence': 0.0, 'spam_probability': 0.0})
                else:
                    classifier = models_to_use.get(selected_models_for_bulk)
                    if classifier:
                        preds = classifier(batch_messages)  # 🚀 batch inference
                        batch_results = [
                            {'message': msg, 'prediction': p['label'].upper(), 'confidence': p['score']}
                            for msg, p in zip(batch_messages, preds)
                        ]
                    else:
                        batch_results = [{'message': msg, 'prediction': 'ERROR', 'confidence': 0.0} for msg in batch_messages]

                results.extend(batch_results)

            except Exception as batch_err:
                results.extend([{'message': msg, 'prediction': 'ERROR', 'confidence': 0.0} for msg in batch_messages])

            # ✅ Progress update with ETA
            processed = end
            elapsed = time.time() - start_time
            rate = processed / elapsed
            remaining = total_messages - processed
            eta = remaining / rate if rate > 0 else 0

            progress_bar.progress(processed / total_messages)
            status_text.text(
                f"Processing message {processed}/{total_messages} - ETA: {int(eta//60)}m {int(eta%60)}s"
            )

        progress_bar.empty()
        status_text.text("✅ Classification complete!")

        return pd.DataFrame(results)

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
# --- Advanced Features Section ---
if analysis_mode == "Ensemble Analysis":
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
# --- Footer ---
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
