# ChatBot Meio Ambiente — Classificação de Notícias Ambientais com IA

Sistema de coleta multi-fonte (portais de notícias + redes sociais) e classificação automática de conteúdos sobre meio ambiente em **Boa**, **Ruim** ou **Irrelevante**, utilizando embeddings Transformer em português (`neuralmind/bert-large-portuguese-cased` e modelos compatíveis via `AutoModel`) combinados com pipeline `StandardScaler + MLPClassifier` e calibração de confiança.

## Estrutura do Projeto

```text
ChatBot-Meio-Ambiente/
├── artifacts/                  # Artefatos leves, metadados e relatórios JSON
├── scripts/
│   ├── validate_dataset.py     # Validação e higienização do CSV de treino
│   └── benchmark_models.py     # Benchmark unificado (TF-IDF + embeddings + fine-tuning)
├── src/
│   ├── config.py               # Caminhos e parâmetros padrão
│   ├── collectors/             # Coletores multi-fonte com retry e extração de datas
│   │   ├── base.py             # BaseCollector e sessão HTTP resiliente
│   │   ├── date_parser.py      # Extração JSON-LD, OpenGraph, RSS, URL e PT-BR
│   │   ├── g1.py               # Coletor G1 Meio Ambiente
│   │   ├── rss.py              # Coletor genérico RSS/Atom (Agência Brasil, ((o))eco, Mongabay)
│   │   ├── social.py           # Conectores sociais públicos (Mastodon, Reddit)
│   │   └── manager.py          # Orquestração isolando falhas por fonte
│   └── nlp/
│       ├── preprocessing.py    # Limpeza e normalização de texto
│       ├── dataset.py          # Leitura, limpeza e validação do dataset
│       ├── embedder.py         # Extração de embeddings ([CLS] ou mean pooling)
│       ├── classifier.py       # Treino estratificado, CV, calibração e inferência
│       └── benchmark.py        # Comparativo entre modelos e exportação JSON
├── tests/                      # Suíte de testes unitários (pytest)
├── noticias_classificadas.csv  # Base rotulada (Boa, Ruim, Irrelevante)
├── TreinarMLP.py               # Entrypoint de treinamento e relatório de métricas
├── ClassificarMLP.py           # Entrypoint de coleta multi-fonte e classificação
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

### 1. Validar (ou higienizar) o dataset

```bash
python scripts/validate_dataset.py
# Para corrigir espaços e duplicatas automaticamente:
python scripts/validate_dataset.py --fix
```

### 2. Comparar modelos e configurações (Benchmark)

Compara os 10 classificadores clássicos sobre TF-IDF sem vazamento de vocabulário e, opcionalmente, compara `[CLS]` vs *mean pooling*, `StandardScaler` e cabeças (`mlp`, `mlp_calibrated`, `logreg`, `svc`) exportando `artifacts/benchmark_report.json`:

```bash
python trabalho.py --no-plot
python trabalho.py --include-embeddings --no-plot
```

### 3. Treinar o classificador e gerar relatório de métricas

Treina com divisão estratificada, validação cruzada `StratifiedKFold` (5-fold), `StandardScaler` e saída probabilística calibrada, salvando o modelo leve, metadados e `artifacts/metrics_report.json`:

```bash
python TreinarMLP.py --no-plot
# Exemplo comparando pooling/classificador/limiar de confiança:
python TreinarMLP.py --pooling mean --classifier svc --max-length 256 --no-plot
```

### 4. Coletar e classificar notícias/posts

```bash
# Todas as fontes padrão (G1, Agência Brasil, ((o))eco, Mongabay e Mastodon):
python ClassificarMLP.py --source all --limit 5 --no-plot

# Fonte específica (g1, agencia_brasil, oeco, mongabay, mastodon, reddit, manual):
python ClassificarMLP.py --source g1 --no-plot
python ClassificarMLP.py --source mastodon --limit 5 --no-plot
python ClassificarMLP.py --source manual --no-plot
```

## Testes

```bash
python -m pytest
```
