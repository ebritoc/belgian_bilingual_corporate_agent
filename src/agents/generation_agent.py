from typing import Optional

import requests


class GenerationAgent:
    def __init__(self, ollama_url: str, model_name: str, timeout: int = 180):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.timeout = timeout

    def build_prompts(self, context: str, question: str, consistency_note: str = ""):
        system_prompt = (
            "You are a financial analyst assistant specializing in European banking reports.\n\n"
            "Your task:\n"
            "- Answer questions using ONLY the provided context\n"
            "- Always cite sources (bank name, page number)\n"
            "- If information is not in the context, explicitly state this\n"
            "- Provide specific numbers and facts when available\n"
            "- Keep answers concise but complete\n\n"
            "Respond in the same language as the question.\n\n"
            "If there is a cross-language consistency warning, mention it briefly at the end.\n"
        )

        user_prompt = (
            "Based on the following excerpts from banking reports, please answer the question.\n\n"
            f"Context:\n\n{context}\n\n---\n\n"
            f"Question: {question}\n\n"
            f"Provide a clear, factual answer with citations.{consistency_note}"
        )
        return system_prompt, user_prompt

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "top_p": 0.9},
            }
            if system_prompt:
                payload["system"] = system_prompt
            resp = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json().get('response', '')
            return f"Error: Ollama returned status {resp.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
