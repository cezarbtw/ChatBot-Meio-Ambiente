"""Assistente conversacional ambiental integrado ao acervo e ao modelo BERT+MLP."""

from src.chatbot.llm import OptionalLlmClient
from src.chatbot.models import ChatCitation, ChatClassification, ChatReply
from src.chatbot.nlu import ParsedIntent, parse_user_message
from src.chatbot.service import ChatBotService

__all__ = [
    "ChatBotService",
    "ChatCitation",
    "ChatClassification",
    "ChatReply",
    "OptionalLlmClient",
    "ParsedIntent",
    "parse_user_message",
]
