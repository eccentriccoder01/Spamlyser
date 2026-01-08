"""
Model initialization and verification module

This module handles the initialization and verification of ML models with
comprehensive error handling to prevent app crashes when models are missing.

Features:
- Graceful degradation when models are unavailable
- User-friendly error messages
- Detailed logging for debugging
- Fallback mechanisms for missing dependencies
"""
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Global flag to track model availability
MODEL_STATUS = False
MODEL_ERROR_MESSAGE = ""
MODEL_WARNINGS = []

def verify_model_availability() -> Tuple[bool, str, list]:
    """
    Verify that required ML frameworks and models are available.
    
    Returns:
        tuple: (success: bool, error_message: str, warnings: list)
    """
    warnings = []
    
    # Check PyTorch availability
    try:
        import torch
        if not torch.cuda.is_available():
            warnings.append("⚠️ CUDA not available. Using CPU for model inference (slower performance).")
    except ImportError as e:
        error_msg = (
            "❌ PyTorch is not installed. Please install it with:\n"
            "   pip install torch torchvision torchaudio\n"
            f"   Error details: {str(e)}"
        )
        return False, error_msg, warnings
    
    # Check transformers library
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as e:
        error_msg = (
            "❌ Transformers library is not installed. Please install it with:\n"
            "   pip install transformers\n"
            f"   Error details: {str(e)}"
        )
        return False, error_msg, warnings
    
    # Check if models directory exists
    models_dir = Path(__file__).parent
    if not models_dir.exists():
        error_msg = (
            f"❌ Models directory not found at: {models_dir}\n"
            "   Please ensure the models directory exists."
        )
        return False, error_msg, warnings
    
    # Try to load a test model to verify transformers setup
    try:
        print("🔄 Verifying model availability... This may take a moment on first run.")
        
        # Use a smaller model for testing to reduce download time
        test_model_name = "distilbert-base-uncased"
        
        # Check if model is cached
        cache_dir = Path.home() / ".cache" / "huggingface" / "transformers"
        model_cached = cache_dir.exists() and any(
            test_model_name in str(p) for p in cache_dir.glob("*")
        )
        
        if not model_cached:
            warnings.append(
                f"📥 Downloading {test_model_name} model for the first time. "
                "This may take a few minutes depending on your internet connection."
            )
        
        # Attempt to load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(test_model_name)
        except Exception as e:
            error_msg = (
                f"❌ Failed to load tokenizer for {test_model_name}.\n"
                "   This could be due to:\n"
                "   - No internet connection (required for first-time download)\n"
                "   - Corrupted cache files\n"
                "   - Insufficient disk space\n"
                f"   Error details: {str(e)}\n\n"
                "   Try clearing the cache:\n"
                f"   rm -rf {cache_dir}"
            )
            return False, error_msg, warnings
        
        # Attempt to load model
        try:
            model = AutoModelForSequenceClassification.from_pretrained(test_model_name)
        except Exception as e:
            error_msg = (
                f"❌ Failed to load model {test_model_name}.\n"
                "   This could be due to:\n"
                "   - No internet connection (required for first-time download)\n"
                "   - Corrupted cache files\n"
                "   - Insufficient disk space\n"
                "   - Insufficient RAM (model requires ~500MB)\n"
                f"   Error details: {str(e)}\n\n"
                "   Try clearing the cache:\n"
                f"   rm -rf {cache_dir}"
            )
            return False, error_msg, warnings
        
        print("✅ Model verification successful!")
        return True, "", warnings
        
    except Exception as e:
        error_msg = (
            "❌ Unexpected error during model verification.\n"
            f"   Error details: {str(e)}\n"
            f"   Error type: {type(e).__name__}\n\n"
            "   Please check:\n"
            "   1. Internet connection (required for first-time model download)\n"
            "   2. Disk space (models require ~500MB-2GB)\n"
            "   3. RAM availability (at least 2GB free recommended)\n"
            "   4. Python version (3.8+ required)\n\n"
            "   If the problem persists, try:\n"
            "   - Reinstalling dependencies: pip install --upgrade transformers torch\n"
            "   - Clearing the cache: rm -rf ~/.cache/huggingface\n"
            "   - Checking system logs for more details"
        )
        return False, error_msg, warnings

def get_model_status_info() -> Dict[str, any]:
    """
    Get detailed information about model status.
    
    Returns:
        dict: Status information including availability, errors, and warnings
    """
    return {
        "available": MODEL_STATUS,
        "error_message": MODEL_ERROR_MESSAGE,
        "warnings": MODEL_WARNINGS,
        "has_errors": bool(MODEL_ERROR_MESSAGE),
        "has_warnings": bool(MODEL_WARNINGS)
    }

def display_model_status_ui():
    """
    Display model status in Streamlit UI with appropriate styling.
    Should be called from app.py to show status to users.
    """
    try:
        import streamlit as st
        
        status_info = get_model_status_info()
        
        if not status_info["available"]:
            # Show error in an expandable section
            with st.expander("⚠️ Model Loading Error - Click to see details", expanded=True):
                st.error("**AI Models Failed to Load**")
                st.markdown(status_info["error_message"])
                st.info(
                    "**What does this mean?**\n\n"
                    "The app requires AI models to function properly. "
                    "Without them, spam detection features will not work.\n\n"
                    "**What can you do?**\n"
                    "1. Follow the instructions above to resolve the issue\n"
                    "2. Restart the app after fixing the problem\n"
                    "3. Contact support if the issue persists"
                )
        
        # Show warnings even if models loaded successfully
        if status_info["warnings"]:
            for warning in status_info["warnings"]:
                st.warning(warning)
                
    except ImportError:
        # Streamlit not available, just print to console
        status_info = get_model_status_info()
        if not status_info["available"]:
            print("\n" + "="*60)
            print("MODEL LOADING ERROR")
            print("="*60)
            print(status_info["error_message"])
            print("="*60 + "\n")
        for warning in status_info["warnings"]:
            print(warning)

# Initialize models on module load
try:
    MODEL_STATUS, MODEL_ERROR_MESSAGE, MODEL_WARNINGS = verify_model_availability()
except Exception as e:
    MODEL_STATUS = False
    MODEL_ERROR_MESSAGE = (
        f"❌ Critical error during model initialization: {str(e)}\n"
        "   Please check your Python environment and dependencies."
    )
    MODEL_WARNINGS = []

# Print status to console for debugging
if not MODEL_STATUS:
    print("\n" + "="*60)
    print("⚠️  MODEL INITIALIZATION FAILED")
    print("="*60)
    print(MODEL_ERROR_MESSAGE)
    print("="*60 + "\n")
else:
    print("✅ Models initialized successfully")
    for warning in MODEL_WARNINGS:
        print(warning)
