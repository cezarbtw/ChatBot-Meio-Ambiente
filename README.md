# ChatBot Meio Ambiente — Classificação de Notícias Ambientais com IA

Sistema de coleta e classificação automática de notícias sobre meio ambiente em **Boa**, **Ruim** ou **Irrelevante**, utilizando embeddings Transformer em português (`neuralmind/bert-large-portuguese-cased` e modelos compatíveis via `AutoModel`) combinados com pipeline `StandardScaler + MLPClassifier` e calibração de confiança.

## Estrutura do Projeto

```text
ChatBot-Meio-Ambiente/
├── artifacts/                  # Artefatos leves, metadados e relatórios JSON
├── scripts/
│   ├── validate_dataset.py     # Validação e higienização do CSV de treino
│   └── benchmark_models.py     # Benchmark unificado (TF-IDF + embeddings + fine-tuning)
├── src/
│   ├── config.py               # Caminhos e parâmetros padrão
│   └── nlp/
│       ├── preprocessing.py    # Limpeza e normalização de texto
│       ├── dataset.py          # Leitura, limpeza e validação do dataset
│       ├── embedder.py         # Extração de embeddings ([CLS] ou mean pooling)
│       ├── classifier.py       # Treino estratificado, CV, calibração e inferência
│       └── benchmark.py        # Comparativo entre modelos e exportação JSON
├── tests/                      # Suíte de testes unitários (pytest)
├── noticias_classificadas.csv  # Base rotulada (Boa, Ruim, Irrelevante)
├── TreinarMLP.py               # Entrypoint de treinamento e relatório de métricas
├── ClassificarMLP.py           # Entrypoint de coleta (G1) / exemplos manuais e inferência
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

### 4. Classificar notícias com probabilidade e detecção de ambiguidade

```bash
# Coletando do G1 Meio Ambiente:
python ClassificarMLP.py --source g1 --no-plot

# Testando notícias manuais de exemplo:
python ClassificarMLP.py --source manual --no-plot
```

## Testes

```bash
python -m pytest
```
