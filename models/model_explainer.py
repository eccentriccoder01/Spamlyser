"""
Model explainability module for Spamlyser Pro using LIME

This module provides functionality to explain the model predictions using LIME
(Local Interpretable Model-agnostic Explanations).
"""

import numpy as np
import pandas as pd
import lime
import lime.lime_text
from typing import Dict, List, Any, Callable

class ModelExplainer:
    """Class for explaining model predictions using LIME"""
    
    def __init__(self, predict_fn: Callable, class_names: List[str] = None):
        """
        Initialize the explainer.
        
        Args:
            predict_fn: Function that takes text and returns class probabilities
            class_names: List of class names (default: ["HAM", "SPAM"])
        """
        self.class_names = class_names or ["HAM", "SPAM"]
        self.predict_fn = predict_fn
        
        # Initialize LIME text explainer
        self.explainer = lime.lime_text.LimeTextExplainer(
            class_names=self.class_names,
            split_expression='\W+'  # Split by non-word characters
        )
    
    def explain_prediction(self, text: str, num_features: int = 10) -> Dict[str, Any]:
        """
        Explain a prediction for a given text.
        
        Args:
            text: Text to explain
            num_features: Number of features to include in the explanation
            
        Returns:
            Dict containing the explanation
        """
        try:
            # Generate explanation
            explanation = self.explainer.explain_instance(
                text, 
                self.predict_fn, 
                num_features=num_features
            )
            
            # Get the explanation data
            explanation_data = {
                'text': text,
                'class_names': self.class_names,
                'explanation': explanation,
                'features': []
            }
            
            # Get most important features for each class
            for i, class_name in enumerate(self.class_names):
                try:
                    features = explanation.as_list(label=i)
                    explanation_data['features'].append({
                        'class': class_name,
                        'important_words': [
                            {
                                'word': word,
                                'weight': weight,
                                'is_positive': weight > 0
                            } for word, weight in features
                        ]
                    })
                except Exception as e:
                    # If we can't get features for this class, add empty list
                    explanation_data['features'].append({
                        'class': class_name,
                        'important_words': [],
                        'error': str(e)
                    })
            
            return explanation_data
        except Exception as e:
            # Return a simple structure if explanation fails
            return {
                'text': text,
                'class_names': self.class_names,
                'error': str(e),
                'features': [
                    {'class': 'HAM', 'important_words': []},
                    {'class': 'SPAM', 'important_words': []}
                ]
            }

    def visualize_explanation(self, explanation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create visualization data for the explanation.
        
        Args:
            explanation_data: Explanation data from explain_prediction
            
        Returns:
            Dict with visualization data
        """
        visualization = {
            'highlighted_text': {},
            'feature_importance': {}
        }
        
        # Check if we have an explanation object or error
        if 'error' in explanation_data:
            visualization['error'] = explanation_data['error']
            # Initialize empty feature importance for each class
            for class_name in self.class_names:
                visualization['feature_importance'][class_name] = []
            return visualization
        
        # Try to get HTML visualization if explanation exists
        try:
            visualization['html'] = explanation_data['explanation'].as_html()
        except Exception:
            visualization['html'] = "HTML visualization could not be generated"
        
        # Process each class
        for class_data in explanation_data['features']:
            class_name = class_data['class']
            # Initialize with empty list to avoid KeyError
            visualization['feature_importance'][class_name] = []
            
            # Skip if there are no words or there was an error
            if 'error' in class_data or not class_data['important_words']:
                continue
                
            # Process word importance data
            feature_importance = []
            for item in class_data['important_words']:
                feature_importance.append({
                    'feature': item['word'],
                    'importance': abs(item['weight']),
                    'effect': 'positive' if item['is_positive'] else 'negative',
                    'weight': item['weight']
                })
            
            # Sort by absolute importance
            feature_importance = sorted(feature_importance, key=lambda x: abs(x['weight']), reverse=True)
            visualization['feature_importance'][class_name] = feature_importance
            
        return visualization
            
        return visualization
        
    def get_threat_explanation(self, text: str, threat_type: str) -> Dict[str, Any]:
        """
        Generate an explanation specific to the threat type.
        
        Args:
            text: Text to explain
            threat_type: Type of threat to focus explanation on
            
        Returns:
            Dict with explanation focused on threat-specific features
        """
        # Return a simple explanation if threat_type is None or invalid
        if not threat_type:
            return {
                "matching_keywords": [],
                "explanation": "No specific threat type identified"
            }
            
        # Threat-specific keywords to look for
        threat_keywords = {
            "Phishing": [
                "account", "verify", "bank", "click", "secure", "login", "password", 
                "update", "confirm", "information", "urgent", "security"
            ],
            "Scam/Fraud": [
                "money", "prize", "winner", "cash", "claim", "million", "lottery", 
                "fortune", "urgent", "won", "inheritance", "fund"
            ],
            "Unwanted Marketing": [
                "discount", "offer", "free", "trial", "buy", "limited", "shop", 
                "exclusive", "promotion", "sale", "subscribe", "deal"
            ],
            "Other": [
                "suspicious", "warning", "important", "attention", "alert", "notice"
            ]
        }
        
        try:
            # Get general explanation
            explanation_data = self.explain_prediction(text)
            
            # Extract keywords relevant to the threat type
            keywords = threat_keywords.get(threat_type, [])
            text_lower = text.lower()
            
            # Find matches in the text
            matches = []
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    matches.append(keyword)
            
            # Find feature weights for these keywords
            threat_features = []
            for class_data in explanation_data['features']:
                for word_data in class_data.get('important_words', []):
                    word = word_data.get('word', '')
                    if any(keyword.lower() in word.lower() for keyword in keywords):
                        threat_features.append({
                            'word': word,
                            'weight': word_data.get('weight', 0),
                            'class': class_data.get('class', ''),
                            'is_positive': word_data.get('is_positive', False)
                        })
            
            # Sort by absolute weight
            threat_features = sorted(threat_features, key=lambda x: abs(x.get('weight', 0)), reverse=True)
            
            # Create threat-specific explanation
            threat_explanation = {
                'threat_type': threat_type,
                'matching_keywords': matches,
                'threat_features': threat_features
            }
            
            return threat_explanation
            
        except Exception as e:
            # Return a simple structure if explanation fails
            return {
                'threat_type': threat_type,
                'matching_keywords': [],
                'threat_features': [],
                'error': str(e)
            }
