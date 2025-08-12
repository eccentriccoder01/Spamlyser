import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Model configurations
MODEL_OPTIONS = {
    "DistilBERT": {
        "id": "mreccentric/distilbert-base-uncased-spamlyser",
        "description": "Lightweight & Fast",
        "icon": "⚡",
    },
    "BERT": {
        "id": "mreccentric/bert-base-uncased-spamlyser",
        "description": "Balanced Performance",
        "icon": "⚖️",
    },
    "RoBERTa": {
        "id": "mreccentric/roberta-base-spamlyser",
        "description": "High Accuracy",
        "icon": "🎯",
    },
    "ALBERT": {
        "id": "mreccentric/albert-base-v2-spamlyser",
        "description": "Efficient & Precise",
        "icon": "🔍",
    }
}

import time
import re
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import json
from dataclasses import dataclass
from collections import defaultdict
import torch  # For model loading and GPU support

# Import the ensemble classifier classes
from ensemble_classifier_method import EnsembleSpamClassifier, ModelPerformanceTracker, PredictionResult

logo_path = str(Path(__file__).resolve().parent / "SpamlyserLogo.png")

# Page configuration
st.set_page_config(
    page_title="Spamlyser Pro - Ensemble Edition",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and animations
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

# Initialize session state
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
    st.session_state.loaded_models = {model_name: None for model_name in MODEL_OPTIONS}

# Header
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

# Model configurations
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

# Sidebar
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
    
    else:  # Ensemble Analysis
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
        
        # Model weights configuration for weighted average
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
        
        # Threshold adjustment for adaptive method
        if selected_ensemble_method == "adaptive_threshold":
            st.markdown("#### 🎛️ Threshold Settings")
            base_threshold = st.slider("Base Threshold", 0.1, 0.9, 0.5, 0.05)
    
    st.markdown("---")
    
    # Session stats
    st.markdown("### 📊 Session Stats")
    if analysis_mode == "Single Model":
        total_classifications = sum(st.session_state.model_stats[selected_model_name].values())
        if total_classifications > 0:
            spam_count = st.session_state.model_stats[selected_model_name]['spam']
            ham_count = st.session_state.model_stats[selected_model_name]['ham']
            spam_rate = (spam_count / total_classifications) * 100
            
            st.metric("Total Analysed", total_classifications)
            st.metric("Spam Detected", spam_count)
            st.metric("Ham (Safe)", ham_count)
            st.metric("Spam Rate", f"{spam_rate:.1f}%")
        else:
            st.info("No classifications yet")
    else:
        if st.session_state.ensemble_history:
            total_ensemble = len(st.session_state.ensemble_history)
            spam_count = sum(1 for h in st.session_state.ensemble_history if h['prediction'] == 'SPAM')
            ham_count = total_ensemble - spam_count
            spam_rate = (spam_count / total_ensemble) * 100 if total_ensemble > 0 else 0
            
            st.metric("Ensemble Analyses", total_ensemble)
            st.metric("Spam Detected", spam_count)
            st.metric("Ham (Safe)", ham_count)
            st.metric("Spam Rate", f"{spam_rate:.1f}%")
        else:
            st.info("No ensemble analyses yet")

# Cache individual components for better reusability
@st.cache_resource
def load_tokenizer(model_id):
    """Load and return a tokenizer for the specified model"""
    try:
        return AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        st.error(f"❌ Error loading tokenizer for {model_id}: {str(e)}")
        return None

@st.cache_resource
def load_model(model_id):
    """Load and return a model with device management"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    except Exception as e:
        st.error(f"❌ Error loading model {model_id}: {str(e)}")
        return None

@st.cache_resource
def _load_model_cached(model_id):
    """Load a complete pipeline with tokenizer and model"""
    try:
        tokenizer = load_tokenizer(model_id)
        model = load_model(model_id)
        
        if tokenizer is None or model is None:
            return None
            
        # Create pipeline with device management
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
    """Lazily load a model only when needed with progress feedback"""
    if st.session_state.loaded_models[model_name] is None:
        model_id = MODEL_OPTIONS[model_name]["id"]
        
        # Create a status container for this model
        status_container = st.empty()
        
        def update_status(message):
            if status_container:
                status_container.info(message)
            if _progress_callback:
                _progress_callback(message)
        
        try:
            update_status(f"Starting to load {model_name}...")
            
            # Update status for tokenizer
            update_status(f"🔄 Loading tokenizer for {model_name}...")
            
            # Load the model (cached)
            update_status(f"🤖 Loading {model_name} model... (This may take a few minutes)")
            model = _load_model_cached(model_id)
            
            if model is not None:
                update_status(f"✅ Successfully loaded {model_name}")
                st.session_state.loaded_models[model_name] = model
            else:
                update_status(f"❌ Failed to load {model_name}")
                return None
                
            time.sleep(1)  # Let the user see the success message
            
        except Exception as e:
            update_status(f"❌ Error loading {model_name}: {str(e)}")
            return None
        finally:
            # Clear the status message after a delay
            time.sleep(1)
            status_container.empty()
            
    return st.session_state.loaded_models[model_name]

def get_loaded_models():
    # Get all loaded models with a progress bar
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
            (i / total_models) * 0.9,  # Leave 10% for completion message
            f"Loading {name} model ({i+1}/{total_models})..."
        )
        
        # Load the model with progress updates
        models[name] = load_model_if_needed(
            name, 
            _progress_callback=lambda msg: update_progress(
                (i / total_models) * 0.9, 
                f"{name}: {msg}"
            )
        )
    
    # Final update
    update_progress(1.0, "✅ All models loaded successfully!")
    time.sleep(1)  # Let users see the success message
    progress_bar.empty()
    status_text.empty()
    
    return models

# Replace the old load_all_models with the new implementation
load_all_models = get_loaded_models

# Helper functions
def analyse_message_features(message):
    # Analyse message characteristics
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
    # Get risk indicators for the message
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
    # Get predictions from all models for ensemble analysis
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

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
    <div class="analysis-header">
        <h3 style="color: #00d4aa; margin: 0;">🔍 {analysis_mode} Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Message input
    user_sms = st.text_area(
        "Enter SMS message to analyse",
        height=120,
        placeholder="Type or paste your SMS message here...",
        help="Enter the SMS message you want to classify as spam or ham (legitimate)"
    )
    
    # Analysis controls
    col_a, col_b, col_c = st.columns([1, 1, 2])
    
    with col_a:
        analyse_btn = st.button("🔍 Analyse Message", type="primary", use_container_width=True)
    
    with col_b:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_btn:
        st.rerun()

# Analysis logic
if analyse_btn and user_sms.strip():
    
    if analysis_mode == "Single Model":
        # Single model analysis
        classifier = load_model_if_needed(selected_model_name)
        
        if classifier is not None:
            with st.spinner(f"🤖 Analyzing with {selected_model_name}..."):
                time.sleep(0.5)
                result = classifier(user_sms)[0]
                label = result['label'].upper()
                confidence = result['score']
                
                # Update stats
                st.session_state.model_stats[selected_model_name][label.lower()] += 1
                st.session_state.model_stats[selected_model_name]['total'] += 1
                
                # Add to history
                st.session_state.classification_history.append({
                    'timestamp': datetime.now(),
                    'message': user_sms[:50] + "..." if len(user_sms) > 50 else user_sms,
                    'prediction': label,
                    'confidence': confidence,
                    'model': selected_model_name
                })
                
                features = analyse_message_features(user_sms)
                risk_indicators = get_risk_indicators(user_sms, label)
                
                # Display results
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
    
    else:
        # Ensemble analysis
        with st.spinner("🤖 Loading all models for ensemble analysis..."):
            models = {}
            for model_name in MODEL_OPTIONS:
                models[model_name] = load_model_if_needed(model_name)
        
        if any(models.values()):  # Check if any models loaded successfully
            with st.spinner("🔍 Running ensemble analysis..."):
                predictions = get_ensemble_predictions(user_sms, models)
                
                if predictions:
                    # Get ensemble result
                    if selected_ensemble_method == "adaptive_threshold":
                        ensemble_result = st.session_state.ensemble_classifier.get_ensemble_prediction(
                            predictions, selected_ensemble_method
                        )
                    else:
                        ensemble_result = st.session_state.ensemble_classifier.get_ensemble_prediction(
                            predictions, selected_ensemble_method
                        )
                    
                    # Add to ensemble history
                    st.session_state.ensemble_history.append({
                        'timestamp': datetime.now(),
                        'message': user_sms[:50] + "..." if len(user_sms) > 50 else user_sms,
                        'prediction': ensemble_result['label'],
                        'confidence': ensemble_result['confidence'],
                        'method': selected_ensemble_method,
                        'spam_probability': ensemble_result['spam_probability']
                    })
                    
                    features = analyse_message_features(user_sms)
                    risk_indicators = get_risk_indicators(user_sms, ensemble_result['label'])
                    
                    # Display ensemble results
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
                    
                    # Show individual model predictions
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
                    
                    # Show ensemble method details
                    st.markdown("#### 📊 Ensemble Method Details")
                    st.markdown(f"**Method:** {ensemble_result['method']}")
                    st.markdown(f"**Details:** {ensemble_result['details']}")
                    
                    if 'model_contributions' in ensemble_result:
                        st.markdown("##### Model Contributions:")
                        for contrib in ensemble_result['model_contributions']:
                            st.write(f"- {contrib['model']}: Weight {contrib['weight']:.3f}, "
                                   f"Contribution: {contrib['contribution']:.3f}")
                    
                    # Comparison with all methods
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
    
    # Common feature analysis
    if 'features' in locals():
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
    
    if analysis_mode == "Single Model" and st.session_state.classification_history:
        # Single model analytics
        st.markdown("#### 🕒 Recent Classifications")
        recent = st.session_state.classification_history[-5:]
        
        for item in reversed(recent):
            status_color = "#ff6b6b" if item['prediction'] == "SPAM" else "#4ecdc4"
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 3px solid {status_color};">
                <strong style="color: {status_color};">{item['prediction']}</strong> ({item['confidence']:.1%})<br>
                <small style="color: #888;">{item['message']}</small><br>
                <small style="color: #666;">{item['model']} • {item['timestamp'].strftime('%H:%M')}</small>
            </div>
            """, unsafe_allow_html=True)
    
    elif analysis_mode == "Ensemble Analysis" and st.session_state.ensemble_history:
        # Ensemble analytics
        st.markdown("#### 🕒 Recent Ensemble Results")
        recent = st.session_state.ensemble_history[-5:]
        
        for item in reversed(recent):
            status_color = "#ff6b6b" if item['prediction'] == "SPAM" else "#4ecdc4"
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 3px solid {status_color};">
                <strong style="color: {status_color};">{item['prediction']}</strong> ({item['confidence']:.1%})<br>
                <small style="color: #888;">{item['message']}</small><br>
                <small style="color: #666;">{ENSEMBLE_METHODS[item['method']]['name']} • {item['timestamp'].strftime('%H:%M')}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Ensemble performance chart
        if len(st.session_state.ensemble_history) > 3:
            st.markdown("#### 📊 Ensemble Performance")
            df_ensemble = pd.DataFrame(st.session_state.ensemble_history)
            
            # Create performance chart
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
                title="Confidence Over Time",
                xaxis_title="Analysis #",
                yaxis_title="Confidence",
                height=250,
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("📝 No classifications yet. Analyse some messages to see statistics!")

# Footer
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

# Advanced Features Section
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
                        st.write(f"- **Accuracy:** {stats['accuracy']:.2%}")
                        st.write(f"- **Total Predictions:** {stats['total_predictions']}")
                        st.write(f"- **Trend:** {stats['performance_trend']}")
                        st.write(f"- **Current Weight:** {stats['current_weight']:.3f}")
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
        
        # Batch analysis option
        st.markdown("#### 📁 Batch Analysis")
        uploaded_file = st.file_uploader(
            "Upload CSV file for batch analysis",
            type=['csv'],
            help="Upload a CSV file with 'message' column for batch spam detection"
        )
        
        if uploaded_file is not None:
            try:
                df_batch = pd.read_csv(uploaded_file)
                if 'message' in df_batch.columns:
                    st.write(f"Found {len(df_batch)} messages to analyze")
                    
                    if st.button("🚀 Run Batch Analysis"):
                        with st.spinner("Running batch analysis..."):
                            models = load_all_models()
                            batch_results = []
                            
                            progress_bar = st.progress(0)
                            for i, message in enumerate(df_batch['message']):
                                if pd.notna(message):
                                    predictions = get_ensemble_predictions(str(message), models)
                                    if predictions:
                                        ensemble_result = st.session_state.ensemble_classifier.get_ensemble_prediction(
                                            predictions, selected_ensemble_method
                                        )
                                        batch_results.append({
                                            'message': str(message)[:100] + "..." if len(str(message)) > 100 else str(message),
                                            'prediction': ensemble_result['label'],
                                            'confidence': ensemble_result['confidence'],
                                            'spam_probability': ensemble_result['spam_probability']
                                        })
                                progress_bar.progress((i + 1) / len(df_batch))
                            
                            # Display results
                            if batch_results:
                                df_results = pd.DataFrame(batch_results)
                                st.markdown("#### 📊 Batch Analysis Results")
                                st.dataframe(df_results, use_container_width=True)
                                
                                # Summary statistics
                                spam_count = len(df_results[df_results['prediction'] == 'SPAM'])
                                ham_count = len(df_results[df_results['prediction'] == 'HAM'])
                                avg_confidence = df_results['confidence'].mean()
                                
                                col_summary1, col_summary2, col_summary3 = st.columns(3)
                                with col_summary1:
                                    st.metric("Total Messages", len(df_results))
                                with col_summary2:
                                    st.metric("Spam Detected", spam_count)
                                with col_summary3:
                                    st.metric("Average Confidence", f"{avg_confidence:.2%}")
                                
                                # Download results
                                csv_results = df_results.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results CSV",
                                    data=csv_results,
                                    file_name=f"spamlyser_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                else:
                    st.error("CSV file must contain a 'message' column")
            except Exception as e:
                st.error(f"Error processing CSV file: {str(e)}")

# Error handling
if analyse_btn and user_sms.strip():
    if analysis_mode == "Single Model":
        # Load the model using the cached function
        model_id = MODEL_OPTIONS[selected_model_name]["id"]
        with st.spinner(f"Loading {selected_model_name}..."):
            classifier = _load_model_cached(model_id)
            if not classifier:
                st.error("❌ Failed to load the selected model. Please try again or select a different model.")
    else:
        models = load_all_models()
        if not any(models.values()):
            st.error("❌ Failed to load ensemble models. Please check your internet connection and try again.")

if analyse_btn and not user_sms.strip():
    st.warning("⚠️ Please enter an SMS message to analyse.")

# Real-time model comparison (if ensemble mode)
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
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed comparison table
        st.dataframe(df_comparison, use_container_width=True)