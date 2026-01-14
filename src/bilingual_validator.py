"""
Bilingual validation module for Belgian banking reports.

Handles language detection, query expansion, and cross-language consistency checking
for French, Dutch, and English documents.
"""
import re
from typing import Dict, List, Any, Optional
import langdetect


class BilingualValidator:
    """
    Validates consistency across bilingual banking reports.

    Key features:
    - Language detection using langdetect library
    - Query expansion with domain-specific term mappings
    - Numeric consistency checking across languages
    """

    # Domain-specific term mappings for banking/finance
    TERM_MAPPINGS = {
        ('fr', 'nl'): [
            ('résultat annuel', 'jaarresultaat'),
            ('résultat net', 'nettowinst'),
            ('bénéfice', 'winst'),
            ('capitaux propres', 'eigen vermogen'),
            ('ratio', 'ratio'),
            ('capital', 'kapitaal'),
        ],
        ('nl', 'fr'): [
            ('jaarresultaat', 'résultat annuel'),
            ('nettowinst', 'résultat net'),
            ('winst', 'bénéfice'),
            ('eigen vermogen', 'capitaux propres'),
            ('kapitaal', 'capital'),
        ],
        ('en', 'fr'): [
            ('annual result', 'résultat annuel'),
            ('net profit', 'résultat net'),
            ('profit', 'bénéfice'),
            ('equity', 'capitaux propres'),
            ('capital', 'capital'),
        ],
        ('en', 'nl'): [
            ('annual result', 'jaarresultaat'),
            ('net profit', 'nettowinst'),
            ('profit', 'winst'),
            ('equity', 'eigen vermogen'),
            ('capital', 'kapitaal'),
        ],
        ('fr', 'en'): [
            ('résultat annuel', 'annual result'),
            ('résultat net', 'net profit'),
            ('bénéfice', 'profit'),
            ('capitaux propres', 'equity'),
        ],
        ('nl', 'en'): [
            ('jaarresultaat', 'annual result'),
            ('nettowinst', 'net profit'),
            ('winst', 'profit'),
            ('eigen vermogen', 'equity'),
        ],
    }

    def detect_language(self, text: str) -> str:
        """
        Detect the language of a text.

        Args:
            text: Input text to analyze

        Returns:
            Language code ('fr', 'nl', or 'en')
        """
        try:
            detected = langdetect.detect(text)
            # Map langdetect codes to our codes
            if detected in ['fr', 'nl', 'en']:
                return detected
            # Fallback to English for unrecognized languages
            return 'en'
        except:
            # Fallback if detection fails
            return 'en'

    def expand_query(self, query: str) -> Dict[str, Any]:
        """
        Expand a query into multiple language variants.

        Args:
            query: Original query text

        Returns:
            Dict with:
                - detected: Detected language
                - variants: List of dicts with 'lang' and 'text' keys
        """
        detected_lang = self.detect_language(query)

        variants = []

        # Always include the original query
        variants.append({'lang': detected_lang, 'text': query})

        # Create translated variants for other languages
        target_languages = ['fr', 'nl', 'en']
        target_languages.remove(detected_lang)

        for target_lang in target_languages:
            translated = self._translate_terms(query, detected_lang, target_lang)
            variants.append({'lang': target_lang, 'text': translated})

        return {
            'detected': detected_lang,
            'variants': variants
        }

    def _translate_terms(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate domain-specific terms in text.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Text with terms translated
        """
        mappings = self.TERM_MAPPINGS.get((source_lang, target_lang), [])

        result = text
        text_lower = text.lower()

        for source_term, target_term in mappings:
            if source_term.lower() in text_lower:
                # Case-insensitive replacement
                result = re.sub(
                    re.escape(source_term),
                    target_term,
                    result,
                    flags=re.IGNORECASE
                )

        return result

    def check_numeric_consistency(self, language_results: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Check if numeric figures are consistent across language results.

        Args:
            language_results: Dict mapping language codes to retrieval results
                             Each result should have 'documents' key

        Returns:
            Dict with:
                - status: 'ok' or 'discrepancy'
                - confidence: Float 0-1
                - notes: Explanation string
        """
        # Extract numbers from each language's documents
        numbers_by_lang = {}

        for lang, result in language_results.items():
            docs = result.get('documents', [])

            # Flatten documents if nested
            flat_docs = []
            for doc in docs:
                if isinstance(doc, list):
                    flat_docs.extend(doc)
                else:
                    flat_docs.append(doc)

            # Extract numbers from first 3 documents
            numbers = set()
            for doc in flat_docs[:3]:
                numbers.update(self._extract_numbers(doc))

            numbers_by_lang[lang] = numbers

        # Check consistency
        if len(numbers_by_lang) < 2:
            return {
                'status': 'ok',
                'confidence': 0.5,
                'notes': 'Insufficient language coverage for comparison'
            }

        # Find common numbers across all languages
        all_numbers = list(numbers_by_lang.values())
        common_numbers = set.intersection(*all_numbers) if all_numbers else set()

        if common_numbers:
            return {
                'status': 'ok',
                'confidence': 0.8,
                'notes': f'Found {len(common_numbers)} common numeric values across languages'
            }

        # Check if any language has numbers but they don't overlap
        has_numbers = any(len(nums) > 0 for nums in all_numbers)
        if has_numbers:
            return {
                'status': 'discrepancy',
                'confidence': 0.7,
                'notes': 'Different numeric figures found across languages'
            }

        return {
            'status': 'ok',
            'confidence': 0.6,
            'notes': 'No numeric figures detected in any language'
        }

    def _extract_numbers(self, text: str) -> set:
        """
        Extract numeric values from text.

        Args:
            text: Text to extract numbers from

        Returns:
            Set of number strings found
        """
        # Match numbers with optional decimal/thousand separators
        # Handles formats like: 1,234.56 or 1.234,56 or 13.9% or 35000
        pattern = r'\b\d+[\d.,]*\b'
        numbers = re.findall(pattern, text)

        # Normalize numbers: remove separators for comparison
        normalized = set()
        for num in numbers:
            # Keep the number but normalize separators
            # Replace both , and . with empty string for comparison
            normalized_num = num.replace(',', '').replace('.', '')
            if normalized_num:  # Only add if not empty
                normalized.add(num)  # Keep original format

        return normalized
