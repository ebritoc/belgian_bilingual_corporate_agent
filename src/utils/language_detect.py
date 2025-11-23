def detect_language(text):
    """
    Detects the language of the given text.
    Placeholder for actual language detection.
    """
    if "le" in text or "la" in text:
        return "fr"
    elif "de" in text or "het" in text:
        return "nl"
    return "unknown"
