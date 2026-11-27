"""Conector opcional para refinamento de respostas com LLM/RAG sobre o acervo SQLite."""

import os
from typing import Dict, Optional, Sequence

import requests

from src.chatbot.models import ChatCitation


class OptionalLlmClient:
    """Cliente leve compatível com API OpenAI Chat Completions; opcional via env vars."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 15.0,
    ) -> None:
        self.api_key = api_key or os.getenv("CHATBOT_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("CHATBOT_LLM_MODEL", "gpt-4o-mini")
        self.base_url = (
            base_url or os.getenv("CHATBOT_LLM_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.timeout = timeout

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def synthesize_rag_answer(
        self,
        user_message: str,
        context_summary: str,
        citations: Sequence[ChatCitation],
        history: Sequence[Dict[str, str]] = (),
    ) -> Optional[str]:
        """Gera resposta fundamentada apenas nos dados recuperados; retorna None se indisponível."""
        if not self.is_configured:
            return None

        citations_block = "\n".join(
            f"- [{c.effective_label} | {c.source} | {c.published_date or 'sem data'}] "
            f"{c.title} ({c.url})"
            for c in citations[:6]
        )
        system_prompt = (
            "Você é o assistente EcoMonitor IA especializado em notícias ambientais. "
            "Responda em português de forma objetiva usando EXCLUSIVAMENTE os fatos "
            "recuperados abaixo do banco de monitoramento.\n\n"
            f"Dados agregados:\n{context_summary}\n\n"
            f"Notícias/posts recuperados:\n{citations_block or '(nenhum item)'}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        for turn in history[-4:]:
            role = turn.get("role", "user")
            content = (turn.get("content") or "").strip()
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_message})

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.2,
                    "max_tokens": 450,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            choices = payload.get("choices") or []
            if not choices:
                return None
            content = choices[0].get("message", {}).get("content")
            return content.strip() if isinstance(content, str) and content.strip() else None
        except Exception:
            return None
