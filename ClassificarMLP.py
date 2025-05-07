import requests
from bs4 import BeautifulSoup
import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
import re


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def classificar_noticia(titulo, texto, mlp, bert_model, tokenizer, encoder):
    titulo = clean_text(titulo)
    texto = clean_text(texto)
    conteudo = titulo + ". " + texto

    inputs = tokenizer(conteudo, return_tensors="pt",
                       truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(
        0).numpy().reshape(1, -1)

    pred = mlp.predict(cls_embedding)
    return encoder.inverse_transform(pred)[0]


def coletar_noticias_g1():
    url = 'https://g1.globo.com/meio-ambiente/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    noticias_coletadas = []
    artigos = soup.select('.feed-post')[:5]
    for artigo in artigos:
        titulo_tag = artigo.select_one('.feed-post-body-title')
        texto_tag = artigo.select_one('.feed-post-body-resumo')

        if titulo_tag:
            titulo = titulo_tag.get_text(strip=True)
            texto = texto_tag.get_text(strip=True) if texto_tag else ""
            noticias_coletadas.append((titulo, texto))

    return noticias_coletadas


if __name__ == "__main__":
    # Carregar modelos
    mlp = joblib.load("mlp_classifier.pkl")
    bert_model = BertModel.from_pretrained("bert_finetuned_noticias")
    tokenizer = BertTokenizer.from_pretrained(
        "neuralmind/bert-large-portuguese-cased")
    encoder = joblib.load("label_encoder.pkl")

    # Lista de notícias
    # Coleta de notícias com web scraping
    noticias = coletar_noticias_g1()

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
