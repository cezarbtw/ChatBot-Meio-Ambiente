# ChatBot Meio Ambiente — Classificação de Notícias Ambientais com IA

Sistema de coleta multi-fonte (portais de notícias + redes sociais), inspeção real de protocolos de rede/TLS, persistência histórica em SQLite, API REST em FastAPI e classificação automática de conteúdos sobre meio ambiente em **Boa**, **Ruim** ou **Irrelevante**, utilizando embeddings Transformer em português (`neuralmind/bert-large-portuguese-cased` e modelos compatíveis via `AutoModel`) combinados com pipeline `StandardScaler + MLPClassifier` e calibração de confiança.

## Estrutura do Projeto

```text
ChatBot-Meio-Ambiente/
├── artifacts/                  # Artefatos leves, métricas e telemetria JSON
├── scripts/
│   ├── validate_dataset.py     # Validação e higienização do CSV de treino
│   ├── benchmark_models.py     # Benchmark unificado (TF-IDF + embeddings + fine-tuning)
│   ├── monitor_pipeline.py     # Coleta incremental e agendada no banco SQLite
│   └── run_api.py              # CLI para subir o servidor FastAPI (Uvicorn)
├── src/
│   ├── config.py               # Caminhos e parâmetros padrão
│   ├── api/                    # Backend REST FastAPI (OpenAPI, CORS, lifespan)
│   │   ├── app.py              # Fábrica FastAPI + CORS + lifespan
│   │   ├── dependencies.py     # Container único em app.state (modelo + SQLite)
│   │   ├── extractor.py        # Extração de conteúdo/telemetria a partir de URL
│   │   ├── main.py             # Entrypoint ASGI (src.api.main:app)
│   │   ├── routes.py           # Rotas /api/health, classify, collect, news, analytics
│   │   └── schemas.py          # Schemas Pydantic de requisição/resposta
│   ├── collectors/             # Coletores multi-fonte com retry e extração de datas
│   │   ├── base.py             # BaseCollector e sessão HTTP resiliente
│   │   ├── date_parser.py      # Extração JSON-LD, OpenGraph, RSS, URL e PT-BR
│   │   ├── g1.py               # Coletor G1 Meio Ambiente
│   │   ├── rss.py              # Coletor genérico RSS/Atom (Agência Brasil, ((o))eco, Mongabay)
│   │   ├── social.py           # Conectores sociais públicos (Mastodon, Reddit)
│   │   └── manager.py          # Orquestração e consolidação da telemetria
│   ├── network/                # Inspeção de protocolos de rede e captura opcional
│   │   ├── models.py           # NetworkTelemetry, CertificateInfo e agregações
│   │   ├── inspector.py        # DNS, TCP, TLS/ALPN (HTTP/2 vs 1.1), cipher, redirects
│   │   └── packet_capture.py   # Captura opcional controlada via tshark ou scapy
│   ├── storage/                # Persistência SQLite/SQLAlchemy, deduplicação e analytics
│   │   ├── database.py         # Engine SQLite, PRAGMA foreign_keys e sessões
│   │   ├── models.py           # Tabelas ORM (sources, runs, news_items, keywords, reviews, telemetry)
│   │   ├── dedup.py            # Canonicalização de URL, hash de conteúdo e palavras-chave
│   │   └── repository.py       # NewsRepository (CRUD, filtros, curadoria e séries diárias)
│   ├── pipeline/
│   │   └── monitor.py          # Ciclos incrementais e agendamento contínuo
│   └── nlp/
│       ├── preprocessing.py    # Limpeza e normalização de texto
│       ├── dataset.py          # Leitura, limpeza e validação do dataset
│       ├── embedder.py         # Extração de embeddings ([CLS] ou mean pooling)
│       ├── classifier.py       # Treino estratificado, CV, calibração e inferência
│       └── benchmark.py        # Comparativo entre modelos e exportação JSON
├── tests/                      # Suíte de testes unitários e de API (pytest)
├── noticias_classificadas.csv  # Base rotulada (Boa, Ruim, Irrelevante)
├── TreinarMLP.py               # Entrypoint de treinamento e relatório de métricas
├── ClassificarMLP.py           # Entrypoint de coleta, telemetria de rede e classificação
└── trabalho.py                 # Entrypoint do benchmark comparativo de modelos
```

## Instalação

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1

pip install -r requirements-dev.txt
```

## Uso

### 1. Subir a API REST (FastAPI + Swagger `/docs`)

```bash
python scripts/run_api.py --host 127.0.0.1 --port 8000
# ou via uvicorn diretamente:
python -m uvicorn src.api.main:app --reload
```

Documentação interativa disponível em `http://127.0.0.1:8000/docs`. Principais endpoints:
- `GET /api/health` — status do servidor, modelo carregado e contagem de itens.
- `POST /api/classify` — classifica título + texto manual ou extrai conteúdo/telemetria de uma URL.
- `POST /api/collect` — dispara coleta sob demanda nas fontes selecionadas e persiste no SQLite.
- `GET /api/news` / `GET /api/news/export` — feed paginado com filtros e exportação CSV.
- `PATCH /api/news/{id}/label` — correção humana do rótulo (*Active Learning*).
- `GET /api/analytics` — série temporal diária, palavras-chave, estatísticas de rede e métricas do modelo.

### 2. Validar (ou higienizar) o dataset

```bash
python scripts/validate_dataset.py
# Para corrigir espaços e duplicatas automaticamente:
python scripts/validate_dataset.py --fix
```

### 3. Comparar modelos e configurações (Benchmark)

```bash
python trabalho.py --no-plot
python trabalho.py --include-embeddings --no-plot
```

### 4. Treinar o classificador e gerar relatório de métricas

```bash
python TreinarMLP.py --no-plot
python TreinarMLP.py --pooling mean --classifier svc --max-length 256 --no-plot
```

### 5. Coletar, inspecionar protocolos de rede e classificar via CLI

```bash
python ClassificarMLP.py --source all --limit 5 --save-db --no-plot
python scripts/monitor_pipeline.py --once --source all --limit 10 --stats
```

## Testes

```bash
python -m pytest
```
