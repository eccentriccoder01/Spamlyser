import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import time
import re
from datetime import datetime
import hashlib
from pathlib import Path

logo_path = str(Path(__file__).resolve().parent.parent / "SpamlyserLogo.png")

# Page configuration
st.set_page_config(
    page_title="Spamlyser Pro",
    page_icon=logo_path,
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
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'classification_history' not in st.session_state:
    st.session_state.classification_history = []
if 'model_stats' not in st.session_state:
    st.session_state.model_stats = {model: {'spam': 0, 'ham': 0, 'total': 0} for model in ["DistilBERT", "BERT", "RoBERTa", "ALBERT"]}

# Header
st.markdown("""
<div style="text-align: center; padding: 20px 0; background: linear-gradient(90deg, #1a1a1a, #2d2d2d); border-radius: 15px; margin-bottom: 30px; border: 1px solid #404040;">
    <h1 style="color: #00d4aa; font-size: 3rem; margin: 0; text-shadow: 0 0 20px rgba(0, 212, 170, 0.3);">
        🛡️ Spamlyser Pro
    </h1>
    <p style="color: #888; font-size: 1.2rem; margin: 10px 0 0 0;">
        Advanced SMS Threat Detection & Analysis Platform
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

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(145deg, #1e1e1e, #2a2a2a); border-radius: 15px; margin-bottom: 20px;">
        <h3 style="color: #00d4aa; margin: 0;">Model Selection</h3>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📊 Session Stats")
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

# Load model pipeline
@st.cache_resource
def load_pipeline(model_id):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        return pipeline("text-classification", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Helper functions
def analyse_message_features(message):
    """Analyse message characteristics"""
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
    """Get risk indicators for the message"""
    indicators = []
    
    # Common spam keywords
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

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="analysis-header">
        <h3 style="color: #00d4aa; margin: 0;">🔍 Message Analysis</h3>
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

# Load the selected model
classifier = load_pipeline(model_info["id"])

if analyse_btn and user_sms.strip() and classifier:
    with st.spinner("🤖 Analyzing message..."):
        # Simulate processing time for better UX
        time.sleep(0.5)
        
        # Get prediction
        result = classifier(user_sms)[0]
        label = result['label'].upper()
        confidence = result['score']
        
        # Update session state
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
        
        # Analyse message features
        features = analyse_message_features(user_sms)
        risk_indicators = get_risk_indicators(user_sms, label)
        
        # Display results
        st.markdown("### 🎯 Classification Results")
        
        # Main prediction card
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
        
        # Detailed analysis
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
    
    # Model comparison
    if st.session_state.classification_history:
        # Recent history
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
        
        # Stats visualization
        if len(st.session_state.classification_history) > 1:
            st.markdown("#### 📊 Model Performance")
            
            # Create dataframe for visualization
            df = pd.DataFrame(st.session_state.classification_history)
            
            # Pie chart for spam/ham distribution
            spam_count = len(df[df['prediction'] == 'SPAM'])
            ham_count = len(df[df['prediction'] == 'HAM'])
            
            if spam_count > 0 or ham_count > 0:
                fig = go.Figure(data=[go.Pie(
                    labels=['SPAM', 'HAM'],
                    values=[spam_count, ham_count],
                    hole=0.6,
                    marker_colors=['#ff6b6b', '#4ecdc4']
                )])
                
                fig.update_layout(
                    title="Classification Distribution",
                    showlegend=True,
                    height=300,
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
    <p style="color: #666; margin: 0;">
        🛡️ <strong>Spamlyser Pro</strong> | Advanced SMS Threat Detection<br>
        Powered by Custom-Trained Transformer Models
    </p>
</div>
""", unsafe_allow_html=True)

# Error handling
if analyse_btn and user_sms.strip() and not classifier:
    st.error("❌ Failed to load the selected model. Please try again or select a different model.")

if analyse_btn and not user_sms.strip():
    st.warning("⚠️ Please enter an SMS message to analyse.")