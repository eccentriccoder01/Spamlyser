"""
Advanced word analysis system for Spamlyser Pro
Provides sophisticated word-level analysis with context awareness
"""

import re
from typing import Dict, List, Any, Tuple
import math

class WordAnalyzer:
    """Advanced and intelligent word analysis system"""
    
    def __init__(self):
        # Enhanced spam indicators with context-aware weights
        self.spam_indicators = {
            # Financial spam (high impact)
            'free': 0.8, 'win': 0.9, 'prize': 0.9, 'cash': 0.8, 'money': 0.7,
            'million': 0.8, 'billion': 0.8, 'dollar': 0.6, 'euro': 0.6, 'pound': 0.6,
            'lottery': 0.9, 'inheritance': 0.8, 'fund': 0.7, 'investment': 0.6,
            'crypto': 0.7, 'bitcoin': 0.7, 'trading': 0.6, 'profit': 0.6,
            'guaranteed': 0.7, 'risk-free': 0.8, 'no cost': 0.8, 'no fee': 0.8,
            
            # Urgency and pressure tactics
            'urgent': 0.7, 'immediately': 0.7, 'instant': 0.6, 'limited': 0.7,
            'expires': 0.6, 'deadline': 0.6, 'limited time': 0.8, 'act now': 0.9,
            'call now': 0.8, 'text stop': 0.9, 'once in a lifetime': 0.9,
            'never again': 0.8, 'last chance': 0.8, 'hurry': 0.6, 'rush': 0.6,
            
            # Action words
            'click': 0.6, 'claim': 0.8, 'collect': 0.7, 'receive': 0.6,
            'unsubscribe': 0.6, 'opt out': 0.6, 'remove': 0.5, 'stop': 0.5,
            
            # Emotional manipulation
            'congratulations': 0.8, 'congrats': 0.8, 'amazing': 0.6, 'incredible': 0.6,
            'exclusive': 0.7, 'secret': 0.7, 'special': 0.6, 'bonus': 0.7,
            'reward': 0.6, 'winner': 0.9, 'lucky': 0.6, 'fortunate': 0.6,
            
            # Offers and deals
            'offer': 0.6, 'discount': 0.5, 'save': 0.4, 'deal': 0.5,
            'sale': 0.4, 'promotion': 0.5, 'bargain': 0.5, 'cheap': 0.4,
            
            # Time pressure
            'today': 0.5, 'tonight': 0.5, 'now': 0.5, 'quick': 0.5,
            'fast': 0.5, 'easy': 0.4, 'simple': 0.4, 'instant': 0.6,
            
            # Suspicious phrases
            'no obligation': 0.7, 'no strings': 0.7, 'no catch': 0.7,
            'guaranteed': 0.7, 'promise': 0.6, 'sure': 0.5, 'certain': 0.5
        }
        
        # Enhanced ham indicators with context awareness
        self.ham_indicators = {
            # Personal relationships
            'thanks': 0.6, 'thank you': 0.7, 'please': 0.5, 'hello': 0.4, 'hi': 0.4,
            'friend': 0.6, 'family': 0.7, 'love': 0.7, 'dear': 0.6, 'sweetie': 0.6,
            'honey': 0.6, 'darling': 0.6, 'buddy': 0.5, 'pal': 0.5,
            
            # Social activities
            'meeting': 0.7, 'appointment': 0.7, 'schedule': 0.6, 'lunch': 0.6,
            'dinner': 0.6, 'coffee': 0.6, 'movie': 0.5, 'party': 0.6,
            'celebration': 0.6, 'birthday': 0.7, 'anniversary': 0.7,
            'weekend': 0.5, 'vacation': 0.6, 'holiday': 0.6, 'trip': 0.6,
            
            # Work and professional
            'work': 0.6, 'office': 0.6, 'project': 0.6, 'conference': 0.7,
            'presentation': 0.7, 'meeting': 0.7, 'deadline': 0.6, 'report': 0.6,
            'email': 0.5, 'call': 0.5, 'phone': 0.5, 'message': 0.5,
            
            # Health and personal
            'doctor': 0.7, 'hospital': 0.7, 'visit': 0.6, 'health': 0.6,
            'school': 0.6, 'university': 0.6, 'college': 0.6, 'class': 0.6,
            'home': 0.5, 'house': 0.5, 'apartment': 0.5, 'room': 0.5,
            
            # Polite language
            'sorry': 0.6, 'apologize': 0.6, 'excuse': 0.5, 'forgive': 0.6,
            'help': 0.5, 'assist': 0.6, 'support': 0.6, 'advice': 0.6,
            'question': 0.5, 'ask': 0.5, 'wonder': 0.5, 'curious': 0.5,
            
            # Time references (normal context)
            'tomorrow': 0.6, 'yesterday': 0.5, 'next week': 0.6, 'last week': 0.5,
            'morning': 0.5, 'afternoon': 0.5, 'evening': 0.5, 'night': 0.5,
            
            # Memory and planning
            'remember': 0.5, 'forgot': 0.5, 'remind': 0.5, 'recall': 0.5,
            'plan': 0.5, 'planning': 0.5, 'arrange': 0.5, 'organize': 0.5,
            
            # Casual conversation
            'see': 0.5, 'talk': 0.5, 'catch up': 0.6, 'see you': 0.6,
            'good luck': 0.6, 'take care': 0.6, 'have fun': 0.5, 'enjoy': 0.5
        }
        
        # Enhanced suspicious patterns with better detection
        self.suspicious_patterns = [
            # Financial patterns
            (r'\$[\d,]+(?:\.\d{2})?', 0.8),  # Dollar amounts
            (r'€[\d,]+(?:\.\d{2})?', 0.8),   # Euro amounts
            (r'£[\d,]+(?:\.\d{2})?', 0.8),   # Pound amounts
            (r'\b\d{4,}\b', 0.6),            # Long numbers (account numbers, etc.)
            
            # URLs and links
            (r'http[s]?://[^\s]+', 0.9),     # Full URLs
            (r'www\.[^\s]+', 0.8),           # Web addresses
            (r'bit\.ly/[^\s]+', 0.9),        # Shortened URLs
            (r'tinyurl\.com/[^\s]+', 0.9),   # Shortened URLs
            
            # Contact information
            (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', 0.7),  # Phone numbers
            (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 0.6),  # Email
            
            # Suspicious formatting
            (r'\b[A-Z]{3,}\b', 0.5),         # All caps words
            (r'[!]{2,}', 0.6),               # Multiple exclamation marks
            (r'[?]{2,}', 0.4),               # Multiple question marks
            (r'[.]{3,}', 0.4),               # Multiple periods
            
            # Spammy phrases
            (r'\bclick here\b', 0.8),        # Click here
            (r'\bact now\b', 0.9),           # Act now
            (r'\blimited time\b', 0.8),      # Limited time
            (r'\bno obligation\b', 0.7),     # No obligation
            (r'\brisk free\b', 0.8),         # Risk free
            (r'\bguaranteed\b', 0.7),        # Guaranteed
            (r'\bwinner\b', 0.9),            # Winner
            (r'\bcongratulations\b', 0.8),   # Congratulations
        ]
        
        # Context modifiers for better analysis
        self.context_modifiers = {
            'negation': ['not', 'no', 'never', 'none', 'nothing', 'nobody'],
            'intensifiers': ['very', 'really', 'extremely', 'absolutely', 'completely'],
            'time_words': ['now', 'today', 'immediately', 'urgent', 'asap']
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text with advanced context-aware logic"""
        # Preprocess text
        original_text = text
        text_lower = text.lower()
        words = text_lower.split()
        word_count = len(words)
        char_count = len(text)
        
        # Analyze each word with context awareness
        word_analysis = []
        spam_score = 0
        ham_score = 0
        context_bonus = 0
        
        # Add a significant base ham score for messages with no explicit spam patterns
        # This helps messages like "you got into google" be recognized as ham with high confidence
        base_ham_score = 0.7 if word_count > 1 else 0.3
        ham_score += base_ham_score
        
        for i, word in enumerate(words):
            # Clean word (remove punctuation but keep original for display)
            clean_word = re.sub(r'[^\w]', '', word).lower()
            original_word = original_text.split()[i] if i < len(original_text.split()) else word
            
            # Get context (previous and next words)
            prev_word = words[i-1] if i > 0 else ""
            next_word = words[i+1] if i < len(words)-1 else ""
            context = f"{prev_word} {word} {next_word}".strip()
            
            # Initialize weights
            spam_weight = 0
            ham_weight = 0
            pattern_weight = 0
            word_type = "neutral"
            context_multiplier = 1.0
            
            # Check for context modifiers
            context_multiplier = self._get_context_multiplier(word, prev_word, next_word, context)
            
            # Check spam indicators
            if clean_word in self.spam_indicators:
                spam_weight = self.spam_indicators[clean_word] * context_multiplier
                spam_score += spam_weight
                word_type = "spam"
            elif clean_word in self.ham_indicators:
                ham_weight = self.ham_indicators[clean_word] * context_multiplier
                ham_score += ham_weight
                word_type = "ham"
            
            # Check for suspicious patterns (case-sensitive for better detection)
            for pattern, weight in self.suspicious_patterns:
                if re.search(pattern, original_word, re.IGNORECASE):
                    pattern_weight = weight * context_multiplier
                    spam_score += pattern_weight
                    if word_type == "neutral":
                        word_type = "suspicious"
                    break
            
            # Check for multi-word phrases
            phrase_weight = self._check_phrases(context, i, words)
            if phrase_weight > 0:
                spam_score += phrase_weight
                if word_type == "neutral":
                    word_type = "suspicious"
                pattern_weight += phrase_weight
            
            # Calculate total influence
            total_influence = spam_weight + ham_weight + pattern_weight
            
            # Determine if word is influential
            is_influential = total_influence > 0.3 or (spam_weight > 0.5 or ham_weight > 0.5)
            
            # If this is a neutral word with no pattern weight, give it a ham weight
            # This will help with highlighting neutral words in the UI
            if word_type == "neutral" and pattern_weight == 0:
                ham_weight = 0.15  # Significant ham weight for neutral words
            
            # Add bias for non-spam words in the word analysis
            # This will help ensure neutral words appear as ham in the UI
            is_spammy = spam_weight > ham_weight and spam_weight > 0
            is_hammy = not is_spammy or word_type == "neutral" or word_type == "ham"
            
            word_analysis.append({
                'word': original_word,
                'clean_word': clean_word,
                'position': i,
                'spam_weight': spam_weight,
                'ham_weight': ham_weight,
                'pattern_weight': pattern_weight,
                'total_influence': total_influence,
                'word_type': word_type,
                'is_spammy': is_spammy,
                'is_hammy': is_hammy,
                'is_hammy': ham_weight > spam_weight and ham_weight > 0,
                'is_influential': is_influential,
                'context_multiplier': context_multiplier,
                'context': context
            })
        
        # Calculate advanced scoring
        ham_score += base_ham_score  # Add the base ham score for natural language
        total_score = spam_score + ham_score
        spam_ratio = spam_score / total_score if total_score > 0 else 0.5
        ham_ratio = ham_score / total_score if total_score > 0 else 0.5
        
        # Apply length and complexity adjustments
        length_factor = self._calculate_length_factor(word_count, char_count)
        complexity_factor = self._calculate_complexity_factor(text)
        
        # Adjust scores based on text characteristics
        spam_score *= length_factor
        ham_score *= length_factor
        
        # Recalculate ratios after adjustments
        total_score = spam_score + ham_score
        spam_ratio = spam_score / total_score if total_score > 0 else 0.5
        ham_ratio = ham_score / total_score if total_score > 0 else 0.5
        
        # Determine prediction with confidence
        predicted_class, confidence = self._determine_prediction(spam_ratio, ham_ratio, total_score, complexity_factor)
        
        return {
            'text': original_text,
            'words': word_analysis,
            'spam_score': spam_score,
            'ham_score': ham_score,
            'spam_ratio': spam_ratio,
            'ham_ratio': ham_ratio,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'word_count': word_count,
            'char_count': char_count,
            'total_influence': total_score,
            'length_factor': length_factor,
            'complexity_factor': complexity_factor,
            'influential_words': [w for w in word_analysis if w['is_influential']]
        }
    
    def _get_context_multiplier(self, word: str, prev_word: str, next_word: str, context: str) -> float:
        """Calculate context multiplier based on surrounding words"""
        multiplier = 1.0
        
        # Check for negation (reduces spam weight)
        if prev_word in self.context_modifiers['negation']:
            multiplier *= 0.3
        
        # Check for intensifiers (increases weight)
        if prev_word in self.context_modifiers['intensifiers']:
            multiplier *= 1.5
        
        # Check for time pressure context
        if any(time_word in context for time_word in self.context_modifiers['time_words']):
            if word in self.spam_indicators:
                multiplier *= 1.3
        
        # Check for personal context (reduces spam weight)
        personal_words = ['friend', 'family', 'dear', 'love', 'thanks', 'please']
        if any(personal in context for personal in personal_words):
            if word in self.spam_indicators:
                multiplier *= 0.7
        
        return multiplier
    
    def _check_phrases(self, context: str, position: int, words: List[str]) -> float:
        """Check for multi-word spam phrases and only apply weight to tokens
        that are actually part of the matched phrase at the current position.
        Previously this function used a simple substring check on a 3-word
        context window which caused neighboring neutral words to be flagged as
        suspicious. This implementation matches phrases token-by-token against
        the full message and returns weight only when the current token index
        falls within a matched phrase span.
        """
        phrase_weights = {
            'click here': 0.8,
            'act now': 0.9,
            'limited time': 0.8,
            'no obligation': 0.7,
            'risk free': 0.8,
            'guaranteed': 0.7,
            'winner': 0.9,
            'congratulations': 0.8,
            'free money': 0.9,
            'win prize': 0.9,
            'cash prize': 0.8,
            'million dollar': 0.8,
            'text stop': 0.9,
            'call now': 0.8,
            'hurry up': 0.7,
            'last chance': 0.8,
            'once in a lifetime': 0.9,
            'never again': 0.8,
            'no strings attached': 0.8,
            'no catch': 0.7
        }
        # Prepare cleaned, lower-cased tokens for matching
        cleaned_words = [re.sub(r'[^\w]', '', w.lower()) for w in words]

        for phrase, weight in phrase_weights.items():
            phrase_tokens = [re.sub(r'[^\w]', '', t.lower()) for t in phrase.split()]
            if not phrase_tokens:
                continue
            L = len(phrase_tokens)
            # Slide over the full token list to find phrase occurrences
            for start in range(0, max(0, len(cleaned_words) - L + 1)):
                if cleaned_words[start:start + L] == phrase_tokens:
                    # If current position is within this matched phrase span, return weight
                    if start <= position < start + L:
                        return weight
        return 0.0
    
    def _calculate_length_factor(self, word_count: int, char_count: int) -> float:
        """Calculate factor based on message length"""
        if word_count < 3:
            return 0.5  # Very short messages are suspicious
        elif word_count < 10:
            return 0.8  # Short messages
        elif word_count < 20:
            return 1.0  # Normal length
        elif word_count < 50:
            return 1.1  # Longer messages (more context)
        else:
            return 1.2  # Very long messages
    
    def _calculate_complexity_factor(self, text: str) -> float:
        """Calculate complexity factor based on text characteristics"""
        # Count special characters, numbers, caps
        special_chars = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', text))
        numbers = len(re.findall(r'\d', text))
        caps = len(re.findall(r'[A-Z]', text))
        
        # Calculate complexity score
        complexity = (special_chars + numbers + caps) / len(text) if text else 0
        
        if complexity > 0.3:
            return 1.3  # High complexity (suspicious)
        elif complexity > 0.15:
            return 1.1  # Medium complexity
        else:
            return 1.0  # Normal complexity
    
    def _determine_prediction(self, spam_ratio: float, ham_ratio: float, total_score: float, complexity_factor: float) -> Tuple[str, float]:
        """Determine final prediction with confidence"""
        # Adjust ratios based on complexity
        if complexity_factor > 1.2:
            spam_ratio *= 1.2
            ham_ratio *= 0.8
            
        # Bias toward HAM for low-spam-score messages
        if spam_ratio < 0.4:  # Lower threshold means more messages default to HAM
            ham_ratio = max(ham_ratio, 0.7)  # Ensure strong ham signal
        
        # Determine class
        if spam_ratio > 0.65:
            predicted_class = "SPAM"
            confidence = min(0.95, spam_ratio)
        elif ham_ratio > 0.4:  # Lower threshold for HAM classification
            predicted_class = "HAM"
            confidence = min(0.95, ham_ratio)
        elif abs(spam_ratio - ham_ratio) < 0.1:
            predicted_class = "HAM"  # Default to HAM in uncertain cases
            confidence = 0.5
        else:
            predicted_class = "SPAM" if spam_ratio > ham_ratio else "HAM"
            confidence = max(spam_ratio, ham_ratio) * 0.8
        
        # Adjust confidence based on total score
        if total_score < 1.0:
            confidence *= 0.7  # Low confidence for low scores
        
        return predicted_class, confidence
    
    def create_highlighted_html(self, analysis: Dict[str, Any]) -> str:
        """Create beautiful HTML with highlighted words based on analysis"""
        words = analysis['text'].split()
        highlighted_words = []
        
        # Force ham prediction for non-spammy messages
        is_ham_message = analysis.get('predicted_class') == 'HAM' or analysis.get('ham_score', 0) > 0
        
        for i, word in enumerate(words):
            if i < len(analysis['words']):
                word_data = analysis['words'][i]
                influence = word_data['total_influence']
                word_type = word_data['word_type']
                is_influential = word_data.get('is_influential', False)
                
                # Very aggressive fallback ham highlighting - ALL words in HAM messages 
                # should be green unless they're explicitly spam indicators
                fallback_ham = (
                    is_ham_message and
                    not word_data.get('is_spammy')
                )

                if influence > 0 or fallback_ham:
                    # Calculate color intensity and effects
                    # Give a small baseline intensity for fallback ham
                    base_influence = 0.15 if fallback_ham and influence == 0 else influence
                    intensity = min(base_influence, 1.0)
                    glow_intensity = min(influence * 1.5, 1.0)
                    
                    if word_data['is_spammy']:
                        # Red gradient for spam with glow effect
                        color = f"hsl(0, {int(60 + intensity * 40)}%, {int(30 + intensity * 20)}%)"
                        bg_color = f"rgba(255, 0, 0, {intensity * 0.2})"
                        border_color = f"hsl(0, {int(70 + intensity * 30)}%, {int(40 + intensity * 20)}%)"
                        shadow_color = f"rgba(255, 0, 0, {glow_intensity * 0.6})"
                        icon = "🚨" if intensity > 0.7 else "⚠️"
                    elif word_data['is_hammy'] or fallback_ham:
                        # Green gradient for ham with glow effect
                        color = f"hsl(120, {int(50 + intensity * 30)}%, {int(25 + intensity * 25)}%)"
                        bg_color = f"rgba(0, 150, 0, {intensity * 0.2})"
                        border_color = f"hsl(120, {int(60 + intensity * 20)}%, {int(35 + intensity * 15)}%)"
                        shadow_color = f"rgba(0, 150, 0, {glow_intensity * 0.6})"
                        icon = "✅" if intensity > 0.7 else "✓"
                    else:
                        # Orange gradient for suspicious patterns
                        color = f"hsl(30, {int(70 + intensity * 20)}%, {int(40 + intensity * 20)}%)"
                        bg_color = f"rgba(255, 165, 0, {intensity * 0.2})"
                        border_color = f"hsl(30, {int(80 + intensity * 10)}%, {int(50 + intensity * 10)}%)"
                        shadow_color = f"rgba(255, 165, 0, {glow_intensity * 0.6})"
                        icon = "🔍" if intensity > 0.7 else "?"
                    
                    # Create enhanced highlighted word with tooltip
                    tooltip_text = f"Influence: {influence:.2f} | Type: {word_type}"
                    if word_data.get('context_multiplier', 1.0) != 1.0:
                        tooltip_text += f" | Context: {word_data['context_multiplier']:.1f}x"
                    
                    highlighted_word = f'''<span style="
                        background: linear-gradient(135deg, {bg_color}, {bg_color});
                        color: {color}; 
                        padding: 4px 8px; 
                        border-radius: 6px; 
                        font-weight: {'bold' if is_influential else '600'};
                        border: 2px solid {border_color};
                        margin: 2px;
                        display: inline-block;
                        box-shadow: 0 0 {int(3 + glow_intensity * 5)}px {shadow_color};
                        transition: all 0.3s ease;
                        position: relative;
                        cursor: help;
                    " 
                    title="{tooltip_text}"
                    onmouseover="this.style.transform='scale(1.05)'; this.style.boxShadow='0 0 {int(5 + glow_intensity * 8)}px {shadow_color}'"
                    onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='0 0 {int(3 + glow_intensity * 5)}px {shadow_color}'"
                    >{icon if is_influential else ''} {word}</span>'''
                    highlighted_words.append(highlighted_word)
                else:
                    # For neutral words, show as ham in HAM messages or neutral in SPAM messages
                    if analysis.get('predicted_class') == 'HAM':
                        # Show as subtle green for neutral words in HAM messages
                        highlighted_words.append(f'<span style="color: #28a745; background-color: rgba(40, 167, 69, 0.1); padding: 3px 6px; border-radius: 4px; border: 1px solid rgba(40, 167, 69, 0.2); margin: 1px;">{word}</span>')
                    else:
                        # No influence, show with subtle styling
                        highlighted_words.append(f'<span style="color: #6c757d; padding: 2px 4px;">{word}</span>')
            else:
                highlighted_words.append(word)
        
        # Create enhanced container with better styling
        return f'''<div style="
            line-height: 2.2; 
            font-size: 19px; 
            padding: 25px; 
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 12px; 
            border: 2px solid #dee2e6;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            position: relative;
            overflow: hidden;
        ">
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 4px;
                background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
            "></div>
            <div style="margin-top: 10px;">
                {" ".join(highlighted_words)}
            </div>
        </div>'''
    
    def get_explanation_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get an enhanced summary of the analysis for display"""
        # Get all words for HAM messages, or just influential words for SPAM messages
        if analysis.get('predicted_class') == 'HAM':
            # For HAM messages, include all words
            influential_words = analysis['words']
        else:
            # For SPAM messages, only include influential words
            influential_words = [w for w in analysis['words'] if w['total_influence'] > 0]
            influential_words.sort(key=lambda x: x['total_influence'], reverse=True)
        
        # Separate by type
        spam_words = [w for w in influential_words if w['is_spammy']]
        ham_words = [w for w in influential_words if w['is_hammy']]
        
        # For HAM messages, treat ALL neutral words as ham indicators
        is_ham_message = analysis.get('predicted_class') == 'HAM'
        
        if is_ham_message:
            # Consider all non-spam words as ham indicators for the summary
            neutral_words = [
                w for w in analysis['words']
                if (not w.get('is_spammy'))  # All non-spam words qualify
            ]
            
            # Force these words to be strong ham indicators
            for w in neutral_words:
                w['is_hammy'] = True
                w['ham_weight'] = max(0.3, w.get('ham_weight', 0))
                w['total_influence'] = max(0.3, w.get('total_influence', 0))
                w['word_type'] = 'ham'
            
            # Replace ham_words with all the identified neutral/ham words
            ham_words = neutral_words
            
            # Sort by position to maintain natural reading order
            ham_words.sort(key=lambda w: w.get('position', 0))
        suspicious_words = [w for w in influential_words if w['word_type'] == 'suspicious']
        
        # Calculate additional metrics
        high_influence_words = [w for w in influential_words if w['total_influence'] > 0.7]
        medium_influence_words = [w for w in influential_words if 0.3 < w['total_influence'] <= 0.7]
        low_influence_words = [w for w in influential_words if 0 < w['total_influence'] <= 0.3]
        
        # Get context insights
        context_insights = self._get_context_insights(analysis)
        
        # Format word lists for display with consistent keys
        display_spam_words = [
            {
                'word': w['word'],
                'influence': w.get('total_influence', 0),
                'type': w.get('word_type', 'spam')
            }
            for w in spam_words
        ]
        
        display_ham_words = [
            {
                'word': w['word'],
                'influence': -w.get('ham_weight', 0.1),  # Negative influence for ham words
                'type': w.get('word_type', 'ham')
            }
            for w in ham_words
        ]
        
        display_suspicious_words = [
            {
                'word': w['word'],
                'influence': w.get('pattern_weight', 0),
                'type': 'suspicious'
            }
            for w in suspicious_words
        ]
        
        return {
            'predicted_class': analysis['predicted_class'],
            'confidence': analysis['confidence'],
            'spam_score': analysis['spam_score'],
            'ham_score': analysis['ham_score'],
            'top_spam_words': display_spam_words[:5],
            'top_ham_words': display_ham_words[:5],
            'suspicious_words': display_suspicious_words[:3],
            'total_words': analysis['word_count'],
            'influential_words_count': len(influential_words),
            'high_influence_count': len(high_influence_words),
            'medium_influence_count': len(medium_influence_words),
            'low_influence_count': len(low_influence_words),
            'length_factor': analysis.get('length_factor', 1.0),
            'complexity_factor': analysis.get('complexity_factor', 1.0),
            'context_insights': context_insights,
            'analysis': self._get_enhanced_analysis_text(analysis),
            'risk_level': self._calculate_risk_level(analysis)
        }
    
    def _get_context_insights(self, analysis: Dict[str, Any]) -> List[str]:
        """Get insights about context and patterns"""
        insights = []
        
        # Check for context modifiers
        context_multipliers = [w.get('context_multiplier', 1.0) for w in analysis['words'] if w.get('context_multiplier', 1.0) != 1.0]
        if context_multipliers:
            avg_multiplier = sum(context_multipliers) / len(context_multipliers)
            if avg_multiplier > 1.2:
                insights.append("High urgency/pressure language detected")
            elif avg_multiplier < 0.8:
                insights.append("Negation or personal context detected")
        
        # Check for complexity
        complexity = analysis.get('complexity_factor', 1.0)
        if complexity > 1.2:
            insights.append("High complexity (special chars, numbers, caps)")
        elif complexity < 0.9:
            insights.append("Simple, natural language")
        
        # Check for length patterns
        word_count = analysis['word_count']
        if word_count < 5:
            insights.append("Very short message (suspicious)")
        elif word_count > 30:
            insights.append("Long message (more context)")
        
        # Check for phrase patterns
        phrase_words = [w for w in analysis['words'] if w.get('pattern_weight', 0) > 0]
        if len(phrase_words) > 2:
            insights.append("Multiple suspicious patterns detected")
        
        return insights
    
    def _calculate_risk_level(self, analysis: Dict[str, Any]) -> str:
        """Calculate overall risk level"""
        spam_ratio = analysis['spam_ratio']
        confidence = analysis['confidence']
        complexity = analysis.get('complexity_factor', 1.0)
        
        risk_score = spam_ratio * confidence * complexity
        
        if risk_score > 0.8:
            return "HIGH RISK"
        elif risk_score > 0.6:
            return "MEDIUM RISK"
        elif risk_score > 0.4:
            return "LOW RISK"
        else:
            return "SAFE"
    
    def _get_enhanced_analysis_text(self, analysis: Dict[str, Any]) -> str:
        """Generate enhanced human-readable analysis text"""
        spam_ratio = analysis['spam_ratio']
        ham_ratio = analysis['ham_ratio']
        confidence = analysis['confidence']
        complexity = analysis.get('complexity_factor', 1.0)
        risk_level = self._calculate_risk_level(analysis)
        
        # Base analysis
        if spam_ratio > 0.75:
            base_analysis = "🚨 STRONG SPAM INDICATORS - Multiple high-impact spam words and patterns detected"
        elif spam_ratio > 0.6:
            base_analysis = "⚠️ MODERATE SPAM INDICATORS - Several concerning words and patterns found"
        elif spam_ratio > 0.4:
            base_analysis = "🤔 MIXED SIGNALS - Some spam indicators but also legitimate language"
        elif ham_ratio > 0.75:
            base_analysis = "✅ STRONG HAM INDICATORS - Message appears legitimate and personal"
        elif ham_ratio > 0.6:
            base_analysis = "✓ MODERATE HAM INDICATORS - Mostly normal language detected"
        else:
            base_analysis = "⚪ NEUTRAL ANALYSIS - No strong indicators either way"
        
        # Add confidence and complexity context
        confidence_text = f" (Confidence: {confidence:.1%})"
        complexity_text = ""
        
        if complexity > 1.2:
            complexity_text = " High complexity suggests automated/spam content."
        elif complexity < 0.9:
            complexity_text = " Simple, natural language suggests human communication."
        
        # Add risk level
        risk_text = f" Risk Level: {risk_level}"
        
        return f"{base_analysis}{confidence_text}.{complexity_text}{risk_text}"
