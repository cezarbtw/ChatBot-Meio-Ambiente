# ChatBot Meio Ambiente — Classificação de Notícias Ambientais com IA

Sistema de coleta e classificação automática de notícias sobre meio ambiente em **Boa**, **Ruim** ou **Irrelevante**, utilizando embeddings em português (`neuralmind/bert-large-portuguese-cased`) combinados com um classificador `MLPClassifier`.

## Estrutura do Projeto

```text
ChatBot-Meio-Ambiente/
├── artifacts/                  # Artefatos leves do classificador MLP e metadados
├── scripts/
│   └── validate_dataset.py     # Validação e higienização do CSV de treino
├── src/
│   ├── config.py               # Caminhos e parâmetros padrão
│   └── nlp/
│       ├── preprocessing.py    # Limpeza e normalização de texto
│       ├── dataset.py          # Leitura, limpeza e validação do dataset
│       ├── embedder.py         # Extração de embeddings BERT sob demanda
│       └── classifier.py       # Treino, persistência leve e inferência
├── tests/                      # Suíte de testes unitários (pytest)
├── noticias_classificadas.csv  # Base rotulada (Boa, Ruim, Irrelevante)
├── TreinarMLP.py               # Entrypoint de treinamento e gráfico de palavras-chave
├── ClassificarMLP.py           # Entrypoint de coleta (G1) / exemplos manuais e inferência
└── trabalho.py                 # Comparativo exploratório de modelos com TF-IDF
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

### 2. Treinar o classificador

O modelo BERT pré-treinado é carregado diretamente do HuggingFace Hub (e cacheado localmente), salvando apenas os artefatos leves do classificador MLP em `artifacts/`:

```bash
python TreinarMLP.py --no-plot
```

### 3. Classificar notícias

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
