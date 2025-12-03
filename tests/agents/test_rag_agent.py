import os
import sys
from pathlib import Path

# Ensure src path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from agents.rag_agent import RAGAgent


def test_language_detection_basic():
    rag = RAGAgent()
    assert rag._detect_language("Quel est le résultat annuel?") == 'fr'
    assert rag._detect_language("Wat is het jaarresultaat?") == 'nl'
    assert rag._detect_language("What is the annual result?") == 'en'


def test_detect_and_expand_query_variants():
    rag = RAGAgent()
    exp = rag.detect_and_expand_query("Quel est le résultat annuel de BNP Paribas Fortis ?")
    langs = [v['lang'] for v in exp['variants']]
    assert set(langs) == {'fr','nl','en'}


def test_compare_across_languages_discrepancy():
    rag = RAGAgent()
    buckets = {
        'fr': {'documents': [["Le résultat net est de 1,234 millions EUR."]], 'metadatas': [[]]},
        'nl': {'documents': [["De nettowinst bedraagt 1.100 miljoen EUR."]], 'metadatas': [[]]},
    }
    res = rag.compare_across_languages(buckets)
    assert res['status'] in {'ok','discrepancy'}


def test_answer_question_bilingual_disabled(monkeypatch):
    rag = RAGAgent({'enable_bilingual_check': False})

    # Mock retrieval to avoid Chroma dependency
    def fake_retrieve(q, n_results=8, language_filter=None):
        return {
            'documents': ["Sample doc FR"],
            'metadatas': [{'bank': 'BNP Paribas Fortis', 'page': 10, 'language': 'fr'}],
        }
    monkeypatch.setattr(rag, 'hybrid_retrieve', fake_retrieve)

    result = rag.answer_question("Quel est le résultat annuel?", use_llm=False)
    assert result['language'] == 'fr'
    assert result['num_sources'] == 1
    assert 'consistency' in result


def test_answer_question_bilingual_enabled(monkeypatch):
    rag = RAGAgent({'enable_bilingual_check': True})

    def fake_multi(exp, n_per_lang=4):
        buckets = {
            'fr': {'documents': ["Résultat net: 1 234"], 'metadatas': [{'bank':'BNP','page':1,'language':'fr','source':'FR.pdf'}]},
            'nl': {'documents': ["Nettowinst: 1 234"], 'metadatas': [{'bank':'BNP','page':1,'language':'nl','source':'NL.pdf'}]},
        }
        merged = {
            'documents': ["Résultat net: 1 234", "Nettowinst: 1 234"],
            'metadatas': [
                {'bank':'BNP','page':1,'language':'fr','source':'FR.pdf'},
                {'bank':'BNP','page':1,'language':'nl','source':'NL.pdf'}
            ]
        }
        return {'merged': merged, 'buckets': buckets}

    monkeypatch.setattr(rag, 'multi_retrieve', fake_multi)

    result = rag.answer_question("Wat is het jaarresultaat?", use_llm=False, bilingual_check=True)
    assert result['language'] == 'nl'
    assert result['num_sources'] == 2
    assert result['consistency']['status'] in {'ok','discrepancy','skip'}
