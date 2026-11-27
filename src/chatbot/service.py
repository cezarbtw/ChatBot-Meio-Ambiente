"""Serviço conversacional integrando NLU local, SQLite, BERT+MLP e LLM opcional."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.chatbot.llm import OptionalLlmClient
from src.chatbot.models import ChatCitation, ChatClassification, ChatReply
from src.chatbot.nlu import ParsedIntent, parse_user_message
from src.config import ARTIFACTS_DIR, METRICS_FILENAME
from src.nlp.classifier import NewsClassifier
from src.storage import NewsRepository, extract_keyword_counts

DEFAULT_FOLLOWUPS = [
    "Qual o panorama ambiental desta semana?",
    "Quais são as piores notícias recentes?",
    "Mostre notícias sobre desmatamento",
    "Resumo dos protocolos de rede identificados",
]


def _item_to_citation(item: Dict[str, Any]) -> ChatCitation:
    return ChatCitation(
        id=int(item["id"]),
        title=str(item["title"]),
        url=str(item["url"]),
        source=item.get("source"),
        effective_label=str(item["effective_label"]),
        confidence=float(item["confidence"]),
        published_date=item.get("published_date"),
        protocol_summary=str(item.get("protocol_summary") or "https"),
    )


class ChatBotService:
    """Orquestra respostas conversacionais locais e modo RAG+LLM opcional."""

    def __init__(
        self,
        repository: NewsRepository,
        classifier: Optional[NewsClassifier] = None,
        artifacts_dir: Path = ARTIFACTS_DIR,
        llm_client: Optional[OptionalLlmClient] = None,
    ) -> None:
        self.repository = repository
        self.classifier = classifier
        self.artifacts_dir = artifacts_dir
        self.llm_client = llm_client or OptionalLlmClient()

    def _load_model_metrics(self) -> Optional[Dict[str, Any]]:
        metrics_path = self.artifacts_dir / METRICS_FILENAME
        if not metrics_path.exists():
            return None
        try:
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _handle_classify_intent(self, parsed: ParsedIntent) -> ChatReply:
        if self.classifier is None:
            return ChatReply(
                answer="O classificador PLN não está carregado no momento.",
                intent="classify_text",
                mode="local",
                suggested_followups=DEFAULT_FOLLOWUPS,
            )

        text = parsed.text_to_classify or ""
        parts = [p.strip() for p in text.split(".", 1)]
        title = parts[0]
        body = parts[1] if len(parts) > 1 and parts[1] else ""
        pred = self.classifier.predict_detailed(title, body)
        kw_counts = extract_keyword_counts(title, body)
        kw_list = [{"keyword": k, "occurrences": v} for k, v in kw_counts.items()]

        amb_note = (
            f" Atenção: classificação sinalizada como ambígua ({pred.ambiguous_reason})."
            if pred.is_ambiguous and pred.ambiguous_reason
            else ""
        )
        kw_note = (
            f" Palavras-chave detectadas: {', '.join(kw_counts.keys())}."
            if kw_counts
            else ""
        )
        answer = (
            f"Classifiquei o texto informado como **{pred.label}** com "
            f"**{pred.confidence:.1%}** de confiança.{amb_note}{kw_note}"
        )

        return ChatReply(
            answer=answer,
            intent="classify_text",
            mode="local",
            classification=ChatClassification(
                label=pred.label,
                confidence=pred.confidence,
                probabilities=pred.probabilities,
                is_ambiguous=pred.is_ambiguous,
                ambiguous_reason=pred.ambiguous_reason,
                keywords=kw_list,
            ),
            suggested_followups=[
                "Qual o panorama ambiental desta semana?",
                "Quais casos ambíguos precisam de revisão?",
            ],
        )

    def _handle_protocol_summary(self) -> ChatReply:
        proto = self.repository.get_protocol_stats()
        total = proto["total_records"]
        if total == 0:
            return ChatReply(
                answer=(
                    "Ainda não há registros de telemetria de rede salvos no banco. "
                    "Execute uma coleta na aba Feed & Curadoria ou via `/api/collect`."
                ),
                intent="protocol_summary",
                mode="local",
                analytics_snapshot={"protocol_stats": proto},
                suggested_followups=DEFAULT_FOLLOWUPS,
            )

        items, _ = self.repository.list_news(limit=4, offset=0)
        citations = [_item_to_citation(it) for it in items]

        alpn_str = ", ".join(f"{k}: {v}" for k, v in proto["alpn_protocols"].items()) or "—"
        tls_str = ", ".join(f"{k}: {v}" for k, v in proto["tls_versions"].items()) or "—"
        http_str = ", ".join(f"{k}: {v}" for k, v in proto["http_versions"].items()) or "—"
        dns_avg = f"{proto['avg_dns_ms']} ms" if proto["avg_dns_ms"] is not None else "—"
        tls_avg = f"{proto['avg_tls_ms']} ms" if proto["avg_tls_ms"] is not None else "—"
        ttfb_avg = f"{proto['avg_ttfb_ms']} ms" if proto["avg_ttfb_ms"] is not None else "—"

        answer = (
            f"Foram inspecionadas **{total}** conexões nas coletas armazenadas. "
            f"Protocolos HTTP: {http_str} | ALPN negociado: {alpn_str} | TLS: {tls_str}. "
            f"Latências médias -> DNS: {dns_avg}, TLS: {tls_avg}, TTFB: {ttfb_avg} "
            f"(erros registrados: {proto['errors_count']})."
        )
        return ChatReply(
            answer=answer,
            intent="protocol_summary",
            mode="local",
            citations=citations,
            analytics_snapshot={"protocol_stats": proto},
            suggested_followups=[
                "Qual o panorama ambiental desta semana?",
                "Quais são as piores notícias recentes?",
            ],
        )

    def _handle_model_performance(self) -> ChatReply:
        metrics = self._load_model_metrics()
        if not metrics:
            return ChatReply(
                answer="Ainda não há relatório `metrics_report.json` gerado em `artifacts/`.",
                intent="model_performance",
                mode="local",
                suggested_followups=DEFAULT_FOLLOWUPS,
            )

        cfg = metrics.get("model_config") or {}
        holdout = metrics.get("holdout") or {}
        cv = metrics.get("cross_validation") or {}
        pol = holdout.get("polarization_errors") or {}
        acc = holdout.get("accuracy", 0.0)
        f1 = holdout.get("f1_macro", 0.0)
        cv_f1 = cv.get("f1_macro_mean")

        cv_part = (
            f" Na validação cruzada {cv.get('n_splits')}-fold, o F1-Macro médio é **{cv_f1:.1%}**."
            if cv_f1 is not None
            else ""
        )
        pol_part = (
            f" Erros de polarização Boa->Ruim: {pol.get('boa_classified_as_ruim_count', 0)} "
            f"({pol.get('boa_classified_as_ruim_rate', 0.0):.1%})."
            if pol
            else ""
        )
        answer = (
            f"O classificador ativo usa `{cfg.get('bert_model_name')}` "
            f"(pooling `{cfg.get('pooling')}`, `StandardScaler={cfg.get('use_scaler')}`). "
            f"No conjunto de teste estratificado, atingiu **{acc:.1%}** de acurácia e **{f1:.1%}** de F1-Macro."
            f"{cv_part}{pol_part}"
        )
        return ChatReply(
            answer=answer,
            intent="model_performance",
            mode="local",
            analytics_snapshot={"model_metrics": metrics},
            suggested_followups=[
                "Quais casos ambíguos precisam de revisão?",
                "Classifique: Nova usina solar reduz emissões de carbono",
            ],
        )

    def _handle_trend_or_search(
        self,
        parsed: ParsedIntent,
        raw_message: str,
        history: Sequence[Dict[str, str]],
        use_llm: Optional[bool],
    ) -> ChatReply:
        overview = self.repository.get_overview_stats()
        daily = self.repository.get_daily_series(
            date_from=parsed.date_from,
            source=parsed.source_filter,
        )
        kw_dist = self.repository.get_keyword_distribution()

        if overview["total_items"] == 0:
            return ChatReply(
                answer=(
                    "O banco de monitoramento ainda está vazio. Use o botão **Coletar Agora** "
                    "no Feed & Curadoria ou envie `Classifique: <texto>` aqui no chat."
                ),
                intent=parsed.intent,
                mode="local",
                analytics_snapshot={"overview": overview},
                suggested_followups=[
                    "Classifique: Pesquisadores criam filtro sustentável para água",
                    "Resumo do desempenho do modelo de IA",
                ],
            )

        items, matched_total = self.repository.list_news(
            label=parsed.label_filter,
            source=parsed.source_filter,
            date_from=parsed.date_from,
            query=parsed.search_query,
            ambiguous_only=parsed.ambiguous_only,
            limit=5,
            offset=0,
        )
        citations = [_item_to_citation(it) for it in items]

        by_lbl = overview["by_label"]
        top_kw = sorted(
            kw_dist["totals_by_keyword"].items(), key=lambda x: x[1], reverse=True
        )
        top_kw_str = ", ".join(f"{k} ({v}x)" for k, v in top_kw if v > 0) or "nenhuma registrada"

        if parsed.intent == "trend_summary":
            recent_days_str = (
                "; ".join(
                    f"{d['date']}: {d['total']} (Boa {d['Boa']}, Ruim {d['Ruim']}, Irrel. {d['Irrelevante']})"
                    for d in daily[-4:]
                )
                or "sem datas registradas"
            )
            local_answer = (
                f"Há **{overview['total_items']}** itens monitorados no acervo "
                f"(Boa: **{by_lbl.get('Boa', 0)}**, Ruim: **{by_lbl.get('Ruim', 0)}**, "
                f"Irrelevante: **{by_lbl.get('Irrelevante', 0)}**; ambíguos: {overview['ambiguous_items']}). "
                f"Principais termos: {top_kw_str}. "
                f"Últimos dias: {recent_days_str}."
            )
        elif matched_total == 0:
            local_answer = (
                "Não encontrei notícias/publicações no banco correspondentes a esses filtros. "
                f"No momento o acervo total possui {overview['total_items']} itens "
                f"(Boa: {by_lbl.get('Boa', 0)}, Ruim: {by_lbl.get('Ruim', 0)}, Irrelevante: {by_lbl.get('Irrelevante', 0)})."
            )
        else:
            filt_desc: List[str] = []
            if parsed.label_filter:
                filt_desc.append(f"classe {parsed.label_filter}")
            if parsed.source_filter:
                filt_desc.append(f"fonte {parsed.source_filter}")
            if parsed.search_query:
                filt_desc.append(f"termo '{parsed.search_query}'")
            if parsed.ambiguous_only:
                filt_desc.append("casos ambíguos")
            scope = f" ({', '.join(filt_desc)})" if filt_desc else ""
            headlines = "; ".join(f'"{c.title}"' for c in citations[:3])
            local_answer = (
                f"Encontrei **{matched_total}** registro(s){scope}. "
                f"Destaques mais recentes: {headlines}."
            )

        mode = "local"
        final_answer = local_answer
        should_use_llm = (
            self.llm_client.is_configured if use_llm is None else (use_llm and self.llm_client.is_configured)
        )
        if should_use_llm:
            rag_answer = self.llm_client.synthesize_rag_answer(
                user_message=raw_message,
                context_summary=local_answer,
                citations=citations,
                history=history,
            )
            if rag_answer:
                final_answer = rag_answer
                mode = "llm_rag"

        return ChatReply(
            answer=final_answer,
            intent=parsed.intent,
            mode=mode,
            citations=citations,
            analytics_snapshot={"overview": overview, "daily_series": daily[-7:]},
            suggested_followups=[
                "Resumo dos protocolos de rede identificados",
                "Quais casos ambíguos precisam de revisão?",
                "Classifique: Desmatamento cai 30% na Amazônia neste ano",
            ],
        )

    def reply(
        self,
        message: str,
        history: Sequence[Dict[str, str]] = (),
        use_llm: Optional[bool] = None,
    ) -> ChatReply:
        """Processa uma pergunta do usuário e retorna resposta estruturada."""
        parsed = parse_user_message(message)

        if parsed.intent == "classify_text":
            return self._handle_classify_intent(parsed)
        if parsed.intent == "protocol_summary":
            return self._handle_protocol_summary()
        if parsed.intent == "model_performance":
            return self._handle_model_performance()

        return self._handle_trend_or_search(parsed, message, history, use_llm)
