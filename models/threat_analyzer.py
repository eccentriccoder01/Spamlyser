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

Performance Optimizations:
- Pre-compiled regex patterns for faster matching
- Cached keyword sets for O(1) lookups
- Optimized pattern matching with early exits
"""

import re
from typing import Dict, List, Union, Any, Tuple, Set
from functools import lru_cache

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

# Pre-compile regex patterns for performance optimization
_COMPILED_PATTERNS = {
    # Phishing patterns
    "phishing_verify": re.compile(r'(verify|confirm|update).{0,20}(account|password|information)', re.IGNORECASE),
    "phishing_click": re.compile(r'(click|tap|follow).{0,10}(link|here)', re.IGNORECASE),
    "phishing_account": re.compile(r'(account|login|password|bank|verify)', re.IGNORECASE),
    
    # Scam patterns
    "scam_prize": re.compile(r'(won|winner|claim|prize).{0,20}(click|call|send)', re.IGNORECASE),
    "scam_money": re.compile(r'(cash|\$\d+|money).{0,30}(click|yes|claim)', re.IGNORECASE),
    "scam_urgent": re.compile(r'(million|lottery|inheritance|cash|money).{0,30}(urgent|now|today)', re.IGNORECASE),
    "scam_won": re.compile(r'(you have won|won .{0,15}\$\d+|claim .{0,15}\$\d+|\$\d+ .{0,15}cash)', re.IGNORECASE),
    
    # Marketing patterns
    "marketing_discount": re.compile(r'(discount|sale|off|save).{0,15}(\d+%|\$\d+)', re.IGNORECASE),
    "marketing_buy": re.compile(r'(buy|purchase|shop|order).{0,20}(now|today|online)', re.IGNORECASE),
    "marketing_exclude": re.compile(r'(account|password|verify|won|claim)', re.IGNORECASE),
}

# Pre-compile common scam phrases for faster checking
_COMMON_SCAM_PHRASES = frozenset([
    "you have won", "cash prize", "claim your", "free money",
    "click yes", "lottery winner", "$5000", "$1000"
])

# Cache keyword sets for O(1) lookups instead of O(n) list searches
_KEYWORD_SETS: Dict[str, Set[str]] = {}

def _initialize_keyword_sets():
    """Initialize keyword sets for faster lookups (called once on module load)"""
    global _KEYWORD_SETS
    for category, data in THREAT_CATEGORIES.items():
        if data["keywords"]:
            _KEYWORD_SETS[category] = set(kw.lower() for kw in data["keywords"])

# Initialize keyword sets on module load
_initialize_keyword_sets()

@lru_cache(maxsize=1024)
def _check_scam_phrases(message_lower: str) -> bool:
    """
    Cached check for common scam phrases.
    Uses LRU cache to avoid repeated checks on similar messages.
    """
    return any(phrase in message_lower for phrase in _COMMON_SCAM_PHRASES)

def _count_keyword_matches(message_lower: str, category: str) -> int:
    """
    Optimized keyword matching using pre-computed sets.
    Uses set lookups (O(1)) instead of list iteration (O(n)).
    """
    if category not in _KEYWORD_SETS:
        return 0
    
    # Split message into words for exact matching
    message_words = set(message_lower.split())
    
    # Count exact word matches
    exact_matches = len(message_words & _KEYWORD_SETS[category])
    
    # Count phrase matches (keywords that contain spaces)
    phrase_matches = sum(
        1 for keyword in _KEYWORD_SETS[category]
        if ' ' in keyword and keyword in message_lower
    )
    
    return exact_matches + phrase_matches

def classify_threat_type(message: str, spam_probability: float) -> Tuple[str, float, Dict[str, Any]]:
    """
    Classify a spam message into a specific threat category.
    
    Optimizations:
    - Pre-compiled regex patterns for faster matching
    - Cached keyword sets for O(1) lookups
    - Early exit conditions to avoid unnecessary processing
    - LRU cache for common phrase checks
    
    Args:
        message: The message content
        spam_probability: The probability that the message is spam
        
    Returns:
        tuple: (threat_type, confidence, metadata)
    """
    # Early exit if not spam
    if spam_probability < 0.5:
        return None, 0.0, {}
        
    # Convert to lowercase once for all checks
    message_lower = message.lower()
    
    # Initialize scores for each category
    scores = {
        "Phishing": 0.0,
        "Scam/Fraud": 0.0,
        "Unwanted Marketing": 0.0,
        "Other": 0.1  # Base score for Other category
    }
    
    # Pre-check for common scam patterns using cached function
    if _check_scam_phrases(message_lower):
        spam_probability = max(spam_probability, 0.85)
        scores["Scam/Fraud"] = 0.2
    
    # Score each category based on keyword matches using optimized function
    for category in ["Phishing", "Scam/Fraud", "Unwanted Marketing"]:
        keyword_count = len(_KEYWORD_SETS.get(category, set()))
        if keyword_count > 0:
            matches = _count_keyword_matches(message_lower, category)
            scores[category] = min(1.0, (matches / keyword_count) * 2.5) * spam_probability
    
    # Specific regex patterns for higher confidence - using pre-compiled patterns
    # Phishing indicators
    if _COMPILED_PATTERNS["phishing_verify"].search(message_lower):
        scores["Phishing"] += 0.25
    
    if (_COMPILED_PATTERNS["phishing_click"].search(message_lower) and 
        _COMPILED_PATTERNS["phishing_account"].search(message_lower)):
        scores["Phishing"] += 0.3
        
    # Scam indicators
    if _COMPILED_PATTERNS["scam_prize"].search(message_lower):
        scores["Scam/Fraud"] += 0.3
        spam_probability = max(spam_probability, 0.85)
    
    if _COMPILED_PATTERNS["scam_money"].search(message_lower):
        scores["Scam/Fraud"] += 0.4
        spam_probability = max(spam_probability, 0.9)
        
    if _COMPILED_PATTERNS["scam_urgent"].search(message_lower):
        scores["Scam/Fraud"] += 0.25
    
    # Special pattern for money scams
    if _COMPILED_PATTERNS["scam_won"].search(message_lower):
        spam_probability = max(spam_probability, 0.95)
        scores["Scam/Fraud"] = max(scores["Scam/Fraud"], 0.8)
    
    # Marketing indicators
    if _COMPILED_PATTERNS["marketing_discount"].search(message_lower):
        scores["Unwanted Marketing"] += 0.3
    
    if (_COMPILED_PATTERNS["marketing_buy"].search(message_lower) and 
        not _COMPILED_PATTERNS["marketing_exclude"].search(message_lower)):
        scores["Unwanted Marketing"] += 0.2
    
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
