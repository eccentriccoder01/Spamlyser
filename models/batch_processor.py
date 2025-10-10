"""
Module for handling batch processing of SMS messages using ensemble models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from .ensemble_classifier_method import EnsembleSpamClassifier, ModelPerformanceTracker
from datetime import datetime

class BatchProcessor:
    """Handles batch processing of SMS messages using ensemble models."""
    
    def __init__(self, ensemble_classifier: Optional[EnsembleSpamClassifier] = None):
        """
        Initialize the batch processor.
        
        Args:
            ensemble_classifier: Optional pre-configured ensemble classifier
        """
        if ensemble_classifier is None:
            performance_tracker = ModelPerformanceTracker()
            ensemble_classifier = EnsembleSpamClassifier(performance_tracker)
        
        self.ensemble_classifier = ensemble_classifier
        self.batch_stats: Dict[str, Any] = {
            'total_messages': 0,
            'processed_messages': 0,
            'spam_detected': 0,
            'ham_detected': 0,
            'avg_confidence': 0.0,
            'start_time': None,
            'end_time': None
        }
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process a single message using all models.
        
        Args:
            message: The SMS message to analyze
            
        Returns:
            Dict containing analysis results
        """
        # Get predictions from all models
        predictions = {}
        for model_name in ['DistilBERT', 'BERT', 'RoBERTa', 'ALBERT']:
            try:
                # Get prediction from individual model
                result = self.ensemble_classifier.get_model_prediction(model_name, message)
                predictions[model_name] = {
                    'label': result.label,
                    'score': result.score,
                    'spam_probability': result.spam_probability
                }
            except Exception as e:
                predictions[model_name] = {
                    'label': 'ERROR',
                    'score': 0.0,
                    'spam_probability': 0.0,
                    'error': str(e)
                }
        
        # Get ensemble prediction using all available methods
        ensemble_results = self.ensemble_classifier.get_all_predictions(predictions)
        
        # Analyze text for risk indicators
        risk_indicators = self._analyze_risk_indicators(message)
        
        return {
            'message': message,
            'model_predictions': predictions,
            'ensemble_predictions': ensemble_results,
            'risk_indicators': risk_indicators,
            'timestamp': datetime.now().isoformat()
        }
    
    def _analyze_risk_indicators(self, message: str) -> Dict[str, bool]:
        """
        Analyze message for common spam/threat indicators.
        
        Args:
            message: The SMS message to analyze
            
        Returns:
            Dict of risk indicators and their presence (True/False)
        """
        message = message.lower()
        
        # Common risk patterns
        patterns = {
            'urls': any(x in message for x in ['http://', 'https://', '.com', '.net', '.org']),
            'urgency': any(x in message for x in ['urgent', 'immediately', 'act now', 'limited time']),
            'money': any(x in message for x in ['$', '€', '£', 'win', 'cash', 'prize', 'money']),
            'personal_info': any(x in message for x in ['password', 'account', 'login', 'ssn', 'credit card']),
            'all_caps': any(word.isupper() and len(word) > 2 for word in message.split()),
            'suspicious_chars': len([c for c in message if not c.isalnum() and not c.isspace()]) / len(message) > 0.1
        }
        
        return patterns
    
    def process_batch(self, 
                     messages: List[str], 
                     batch_size: int = 100,
                     progress_callback: Optional[callable] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process a batch of messages with progress tracking.
        
        Args:
            messages: List of SMS messages to analyze
            batch_size: Number of messages to process in parallel
            progress_callback: Optional callback function to report progress
            
        Returns:
            Tuple of (list of results, batch statistics)
        """
        self.batch_stats['total_messages'] = len(messages)
        self.batch_stats['processed_messages'] = 0
        self.batch_stats['start_time'] = datetime.now()
        
        results = []
        
        # Process messages in parallel batches
        with ThreadPoolExecutor(max_workers=min(batch_size, 10)) as executor:
            future_to_message = {executor.submit(self.process_message, msg): msg 
                               for msg in messages}
            
            for future in future_to_message:
                result = future.result()
                results.append(result)
                
                # Update statistics
                self.batch_stats['processed_messages'] += 1
                if result['ensemble_predictions']['majority_voting']['label'] == 'SPAM':
                    self.batch_stats['spam_detected'] += 1
                else:
                    self.batch_stats['ham_detected'] += 1
                
                # Calculate running average confidence
                self.batch_stats['avg_confidence'] = ((self.batch_stats['avg_confidence'] * 
                    (self.batch_stats['processed_messages'] - 1) +
                    result['ensemble_predictions']['majority_voting']['confidence']) / 
                    self.batch_stats['processed_messages'])
                
                # Report progress if callback provided
                if progress_callback:
                    progress = self.batch_stats['processed_messages'] / self.batch_stats['total_messages']
                    progress_callback(progress)
        
        self.batch_stats['end_time'] = datetime.now()
        processing_time = (self.batch_stats['end_time'] - self.batch_stats['start_time']).total_seconds()
        self.batch_stats['processing_time'] = processing_time
        self.batch_stats['messages_per_second'] = self.batch_stats['total_messages'] / processing_time
        
        return results, self.batch_stats
    
    def generate_report(self, results: List[Dict[str, Any]], format: str = 'csv') -> pd.DataFrame:
        """
        Generate a detailed report from batch processing results.
        
        Args:
            results: List of processing results
            format: Output format ('csv' or 'excel')
            
        Returns:
            DataFrame containing the report data
        """
        report_data = []
        
        for result in results:
            # Prepare row data
            row = {
                'Message': result['message'],
                'Timestamp': result['timestamp']
            }
            
            # Add individual model predictions
            for model, pred in result['model_predictions'].items():
                row[f'{model}_Classification'] = pred['label']
                row[f'{model}_Confidence'] = pred['score']
                row[f'{model}_SpamProbability'] = pred['spam_probability']
            
            # Add ensemble predictions
            for method, pred in result['ensemble_predictions'].items():
                row[f'Ensemble_{method}_Classification'] = pred['label']
                row[f'Ensemble_{method}_Confidence'] = pred['confidence']
                row[f'Ensemble_{method}_SpamProbability'] = pred['spam_probability']
            
            # Add risk indicators
            for indicator, present in result['risk_indicators'].items():
                row[f'Risk_{indicator}'] = present
            
            report_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(report_data)
        
        return df