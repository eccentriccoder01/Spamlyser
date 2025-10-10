"""
Feedback handler for Spamlyser Pro.
Handles storing and retrieving user feedback.
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import streamlit as st

class FeedbackHandler:
    """Handles user feedback operations."""
    
    def __init__(self, feedback_file: str = "feedback_data.json"):
        """
        Initialize the FeedbackHandler.
        
        Args:
            feedback_file: Path to the JSON file for storing feedback.
        """
        self.feedback_file = feedback_file
        self._ensure_feedback_file()
    
    def _ensure_feedback_file(self) -> None:
        """Ensure the feedback file exists, create if it doesn't."""
        if not os.path.exists(self.feedback_file):
            with open(self.feedback_file, 'w') as f:
                json.dump([], f)
    
    def save_feedback(self, feedback_data: Dict[str, Any]) -> bool:
        """
        Save feedback to the feedback file.
        
        Args:
            feedback_data: Dictionary containing feedback information.
                For general feedback, should include:
                - feedback_type: str (bug, feature, suggestion, etc.)
                - rating: int (1-5)
                - message: str
                - email: Optional[str]
                - timestamp: str
                
                For Word Analysis feedback, should include:
                - context: str ("Word Analysis")
                - message_analyzed: str
                - prediction_accuracy: str
                - words_relevance: str
                - word_feedback: str
                - additional_feedback: str
                - correct_classification: Optional[str]
                - timestamp: str
        
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            # Add timestamp if not already present
            if 'timestamp' not in feedback_data:
                feedback_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Read existing feedback
            feedbacks = self.get_all_feedback()
            
            # Add new feedback
            feedbacks.append(feedback_data)
            
            # Write back to file
            with open(self.feedback_file, 'w') as f:
                json.dump(feedbacks, f, indent=2)
            
            return True
        except Exception as e:
            st.error(f"Error saving feedback: {str(e)}")
            return False
    
    def get_all_feedback(self) -> List[Dict[str, Any]]:
        """
        Get all feedback entries.
        
        Returns:
            List of feedback dictionaries.
        """
        try:
            with open(self.feedback_file, 'r') as f:
                return json.load(f)
        except Exception:
            return []
    
    def get_feedback_by_type(self, feedback_type: str) -> List[Dict[str, Any]]:
        """
        Get feedback filtered by type.
        
        Args:
            feedback_type: Type of feedback to filter by.
        
        Returns:
            List of filtered feedback dictionaries.
        """
        all_feedback = self.get_all_feedback()
        return [f for f in all_feedback if f.get('feedback_type') == feedback_type]
    
    def get_word_analysis_feedback(self) -> List[Dict[str, Any]]:
        """
        Get all Word Analysis specific feedback.
        
        Returns:
            List of Word Analysis feedback entries.
        """
        all_feedback = self.get_all_feedback()
        return [f for f in all_feedback if f.get('context') == "Word Analysis"]
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the feedback.
        
        Returns:
            Dictionary with feedback statistics.
        """
        feedbacks = self.get_all_feedback()
        
        if not feedbacks:
            return {
                "total": 0,
                "average_rating": 0,
                "by_type": {},
                "has_email": 0
            }
        
        stats = {
            "total": len(feedbacks),
            "by_type": {},
            "has_email": sum(1 for f in feedbacks if f.get('email')),
        }
        
        # Calculate average rating
        ratings = [f.get('rating', 0) for f in feedbacks if f.get('rating') is not None]
        stats["average_rating"] = sum(ratings) / len(ratings) if ratings else 0
        
        # Count by type
        for feedback in feedbacks:
            feedback_type = feedback.get('feedback_type', 'unknown')
            stats["by_type"][feedback_type] = stats["by_type"].get(feedback_type, 0) + 1
        
        return stats
    
    def export_to_github_issue(self, feedback_id: int) -> str:
        """
        Format feedback as a GitHub issue body.
        
        Args:
            feedback_id: Index of the feedback in the feedback list.
        
        Returns:
            Formatted GitHub issue body text.
        """
        feedbacks = self.get_all_feedback()
        
        if feedback_id < 0 or feedback_id >= len(feedbacks):
            return "Invalid feedback ID"
        
        feedback = feedbacks[feedback_id]
        
        issue_body = f"""
## User Feedback

**Type:** {feedback.get('feedback_type', 'Not specified')}
**Rating:** {'⭐' * feedback.get('rating', 0)}
**Date:** {feedback.get('timestamp', 'Not recorded')}

### Message:
{feedback.get('message', 'No message provided')}

### Contact:
{feedback.get('email', 'No contact information provided')}
"""
        return issue_body