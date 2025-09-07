import re
from typing import List

ABBREVIATIONS = {
    "u": "you",
    "r": "are",
    "ur": "your",
    "pls": "please",
    "msg": "message",
    "b4": "before",
    "gr8": "great",
    "l8r": "later",
    "thx": "thanks",
    "plz": "please",
    "im": "I am",
    "btw": "by the way",
    "idk": "I don't know",
    "omg": "oh my god",
    "ttyl": "talk to you later",
    "asap": "as soon as possible"
}
LEETSPEAK = {
    "0": "o",
    "1": "i",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t"
}

def expand_abbreviations(text: str) -> str:
    words = text.split()
    expanded = [ABBREVIATIONS.get(w.lower(), w) for w in words]
    return " ".join(expanded)

def correct_leetspeak(text: str) -> str:
    for k, v in LEETSPEAK.items():
        text = re.sub(rf"{k}", v, text)
    return text

def count_suspicious_elements(text: str) -> dict:
    suspicious = {
        "all_caps": sum(1 for w in text.split() if w.isupper() and len(w) > 2),
        "excessive_punct": len(re.findall(r"!{2,}|\?{2,}|\${2,}", text)),
        "phone_numbers": len(re.findall(r"\b\d{10,}\b", text)),
        "urls": len(re.findall(r"https?://|www\.", text)),
    }
    return suspicious

def preprocess_message(text: str) -> dict:
    cleaned = expand_abbreviations(text)
    cleaned = correct_leetspeak(cleaned)
    suspicious = count_suspicious_elements(cleaned)
    return {
        "cleaned": cleaned,
        "suspicious": suspicious
    }

# Example usage
if __name__ == "__main__":
    msg = "Fr33 M0n3y!!! u r winner! Call 9876543210 now!"
    result = preprocess_message(msg)
    print(result)
