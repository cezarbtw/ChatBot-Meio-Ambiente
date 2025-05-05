import torch
import joblib
import numpy as np
from transformers import BertTokenizer, BertModel
import re

# Função pra limpar o texto (igual a usada no treino)


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text.lower()

# Função para carregar e classificar


def classificar_noticia(titulo, texto):
    # Carrega modelos
    mlp = joblib.load("mlp_classifier.pkl")
    bert_model = BertModel.from_pretrained("bert_finetuned_noticias")
    tokenizer = BertTokenizer.from_pretrained(
        "neuralmind/bert-large-portuguese-cased")
    encoder = joblib.load("label_encoder.pkl")

    # Prepara texto completo (igual ao treino)
    titulo = clean_text(titulo)
    texto = clean_text(texto)
    conteudo = titulo + ". " + texto

    # Tokeniza e gera embedding
    inputs = tokenizer(conteudo, return_tensors="pt",
                       truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(
        0).numpy().reshape(1, -1)

    # Classifica
    pred = mlp.predict(cls_embedding)
    return encoder.inverse_transform(pred)[0]


if __name__ == "__main__":
    # Carregar modelos
    mlp = joblib.load("mlp_classifier.pkl")
    bert_model = BertModel.from_pretrained("bert_finetuned_noticias")
    tokenizer = BertTokenizer.from_pretrained(
        "neuralmind/bert-large-portuguese-cased")
    encoder = joblib.load("label_encoder.pkl")

    # Lista de notícias
    noticias = [
        ("Homem tenta atravessar oceano em bola inflável gigante",
         "Um aventureiro decidiu cruzar o Oceano Atlântico usando apenas uma bola inflável gigante, afirmando que quer provar a resistência do plástico utilizado na fabricação do objeto. Autoridades locais acompanham o caso com curiosidade."),
        ("Poluição do ar atinge níveis críticos em grandes capitais brasileiras",
         "Relatório do Ministério do Meio Ambiente aponta que cidades como São Paulo e Belo Horizonte apresentaram índices alarmantes de material particulado no ar durante o último mês."),
        ("Novo filme brasileiro vence festival internacional",
         "Uma produção nacional surpreende jurados e leva prêmio máximo em festival europeu, destacando-se pela originalidade e temática social."),
        ("Trânsito congestionado após acidente em rodovia",
         "Um acidente envolvendo três veículos causou congestionamento na rodovia BR-101, próximo ao km 250. Ninguém se feriu gravemente."),
        ("Energia solar bate recorde de produção no Brasil em 2025",
         "S país superou a marca de 30 GW de capacidade instalada em energia solar, impulsionado por incentivos fiscais e queda no preço de painéis fotovoltaicos.")
    ]

    # Classificar todas
    resultados = []
    for i, (titulo, texto) in enumerate(noticias, start=1):
        classe = classificar_noticia(
            titulo, texto, mlp, bert_model, tokenizer, encoder)
        resultados.append(classe)
        print(f"Notícia {i}: {classe}")

    # Contar classes e plotar
    contagem_classes = pd.Series(resultados).value_counts()

    plt.figure(figsize=(8, 6))
    contagem_classes.plot(kind="bar", color=["gray", "red", "green"])
    plt.title("Classificação das Novas Notícias")
    plt.xlabel("Classificação")
    plt.ylabel("Quantidade")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
