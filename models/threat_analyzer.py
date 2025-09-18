"""
Threat Analyzer for Spamlyser Pro - Multi-class Threat Analysis Module

This module enhances Spamlyser's binary classification (SPAM/HAM) by adding
multi-class threat categorization capabilities. It analyzes messages to determine
the specific type of threat they represent.

Threat Categories:
- 🎣 Phishing: Messages designed to steal sensitive information
- 💸 Scam/Fraud: Deceptive messages aimed at tricking the user into sending money
- 📢 Unwanted Marketing: Legitimate but unsolicited advertising or promotional content
- 🤖 Other: A general category for spam that doesn't fit neatly into the others
"""

import re
from typing import Dict, List, Union, Any, Tuple

# Threat category indicators
THREAT_CATEGORIES = {
    "Phishing": {
        "icon": "🎣",
        "color": "#ff6b6b",  # Reddish
        "description": "Attempts to steal sensitive information",
        "keywords": [
            "verify", "account", "login", "password", "bank", "credit card", "update your",
            "security", "verify your", "confirm your", "log-in", "suspicious activity",
            "unusual activity", "unauthorized", "click here", "click the link", "your account",
            "limited access", "disabled", "locked", "suspended", "authenticate"
        ],
        "advice": [
            "Never click on suspicious links in messages",
            "Don't share your passwords or personal information",
            "Legitimate companies never ask for sensitive information via SMS",
            "Contact the company directly through their official website or phone number"
        ]
    },
    
    "Scam/Fraud": {
        "icon": "💸",
        "color": "#ff9f1c",  # Orange
        "description": "Attempts to trick you into sending money",
        "keywords": [
            "won", "winner", "prize", "lottery", "claim", "cash", "award", "million",
            "inheritance", "donate", "charity", "help", "urgent", "fund", "offer",
            "investment", "opportunity", "money", "transfer", "fee", "wealthy",
            "fortune", "prince", "princess", "billion", "dollars", "cash prize"
        ],
        "advice": [
            "Remember: If it sounds too good to be true, it probably is",
            "Never send money to claim a 'prize' or 'lottery winnings'",
            "Legitimate contests don't require payment to claim prizes",
            "Be skeptical of unexpected winnings or inheritance claims"
        ]
    },
    
    "Unwanted Marketing": {
        "icon": "📢",
        "color": "#ffca3a",  # Yellow
        "description": "Unsolicited promotional content",
        "keywords": [
            "discount", "sale", "offer", "buy", "purchase", "subscribe", "limited time",
            "deal", "promotion", "off", "free", "save", "shop", "exclusive", "only",
            "membership", "trial", "join", "sign up", "subscription", "coupon", "discount code"
        ],
        "advice": [
            "You can opt out of marketing messages with 'STOP' replies",
            "Check if the sender is a legitimate business you've interacted with",
            "Consider reporting unwanted marketing to your carrier",
            "Review your privacy settings and where you share your phone number"
        ]
    },
    
    "Other": {
        "icon": "🤖",
        "color": "#8338ec",  # Purple
        "description": "Other suspicious or unwanted messages",
        "keywords": [],  # This is a catch-all category
        "advice": [
            "Be cautious with any unexpected or suspicious messages",
            "Don't reply to unknown senders",
            "Consider blocking the sender if messages persist",
            "Report suspicious messages to your carrier"
        ]
    }
}

def classify_threat_type(message: str, spam_probability: float) -> Tuple[str, float, Dict[str, Any]]:
    """
    Classify a spam message into a specific threat category.
    
    Args:
        message: The message content
        spam_probability: The probability that the message is spam
        
    Returns:
        tuple: (threat_type, confidence, metadata)
    """
    if spam_probability < 0.5:
        return None, 0.0, {}  # Not spam, no threat type
        
    # Convert to lowercase for matching
    message_lower = message.lower()
    
    # Initialize scores for each category
    scores = {
        "Phishing": 0.0,
        "Scam/Fraud": 0.0,
        "Unwanted Marketing": 0.0,
        "Other": 0.1  # Base score for Other category
    }
    
    # Pre-check for common scam patterns and boost spam probability if found
    if any(phrase in message_lower for phrase in [
            "you have won", "cash prize", "claim your", "free money", 
            "click yes", "lottery winner", "$5000", "$1000"
        ]):
        # These are very likely spam/scam messages
        spam_probability = max(spam_probability, 0.85)
        scores["Scam/Fraud"] = 0.2  # Give scam category a head start
    
    # Score each category based on keyword matches
    for category, data in THREAT_CATEGORIES.items():
        if category == "Other":
            continue  # Skip, it's our fallback category
        
        # Calculate keyword matches
        keywords = data["keywords"]
        matches = sum(1 for keyword in keywords if keyword.lower() in message_lower)
        
        # Calculate score based on matches and keyword list length
        if keywords:
            scores[category] = min(1.0, (matches / len(keywords)) * 2.5) * spam_probability
    
    # Specific regex patterns for higher confidence in certain categories
    # Phishing indicators - links with urgency
    if re.search(r'(verify|confirm|update).{0,20}(account|password|information)', message_lower):
        scores["Phishing"] += 0.25
    if re.search(r'(click|tap|follow).{0,10}(link|here)', message_lower) and \
       re.search(r'(account|login|password|bank|verify)', message_lower):
        scores["Phishing"] += 0.3
        
    # Scam indicators - money and urgency
    if re.search(r'(won|winner|claim|prize).{0,20}(click|call|send)', message_lower):
        scores["Scam/Fraud"] += 0.3
        # This is clearly spam, so boost the overall spam score
        spam_probability = max(spam_probability, 0.85)
    
    # Look for cash/money mentions with YES/claim/click patterns (common in scams)
    if re.search(r'(cash|\$\d+|money).{0,30}(click|yes|claim)', message_lower, re.IGNORECASE):
        scores["Scam/Fraud"] += 0.4
        # This is clearly spam, so boost the overall spam score
        spam_probability = max(spam_probability, 0.9)
        
    if re.search(r'(million|lottery|inheritance|cash|money).{0,30}(urgent|now|today)', message_lower):
        scores["Scam/Fraud"] += 0.25
    
    # Marketing indicators
    if re.search(r'(discount|sale|off|save).{0,15}(\d+%|\$\d+)', message_lower):
        scores["Unwanted Marketing"] += 0.3
    if re.search(r'(buy|purchase|shop|order).{0,20}(now|today|online)', message_lower) and \
       not re.search(r'(account|password|verify|won|claim)', message_lower):
        scores["Unwanted Marketing"] += 0.2
    
    # Special pattern for money scams - detect even with low spam probability
    if re.search(r'(you have won|won .{0,15}\$\d+|claim .{0,15}\$\d+|\$\d+ .{0,15}cash)', message_lower, re.IGNORECASE):
        spam_probability = max(spam_probability, 0.95)  # Definitely spam
        scores["Scam/Fraud"] = max(scores["Scam/Fraud"], 0.8)  # Boost scam score
    
    # Find the category with the highest score
    best_category = max(scores.items(), key=lambda x: x[1])
    threat_type = best_category[0]
    confidence = best_category[1]
    
    # If confidence is too low, default to "Other"
    if confidence < 0.3 and threat_type != "Other":
        threat_type = "Other"
        confidence = max(0.3, scores["Other"])
    
    # Prepare metadata
    metadata = {
        "category_scores": scores,
        "category_icon": THREAT_CATEGORIES[threat_type]["icon"],
        "category_color": THREAT_CATEGORIES[threat_type]["color"],
        "category_description": THREAT_CATEGORIES[threat_type]["description"],
    }
    
    return threat_type, confidence, metadata

def get_threat_specific_advice(threat_type: str) -> List[str]:
    """
    Get advice specific to a threat category.
    
    Args:
        threat_type: The identified threat category
        
    Returns:
        list: Specific advice for handling this type of threat
    """
    if not threat_type or threat_type not in THREAT_CATEGORIES:
        return []
        
    return THREAT_CATEGORIES[threat_type]["advice"]
