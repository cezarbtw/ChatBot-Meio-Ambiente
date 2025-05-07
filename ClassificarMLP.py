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
        link_tag = artigo.select_one('a')
        texto_tag = artigo.select_one('.feed-post-body-resumo')

        if titulo_tag and link_tag:
            titulo = titulo_tag.get_text(strip=True)
            texto = texto_tag.get_text(strip=True) if texto_tag else ""
            link = link_tag.get('href')

            # Extrai o protocolo (http ou https)
            protocolo = link.split(':')[0] if link else "desconhecido"

            noticias_coletadas.append((titulo, texto, link, protocolo))

    return noticias_coletadas


noticia_manual = [
    ("Nova medida de reflorestamento é anunciada",
     "O governo lançou uma iniciativa para recuperar áreas desmatadas na Amazônia."),
    ("Empresa polui rio com resíduos tóxicos",
     "Moradores denunciam despejo ilegal de resíduos químicos em rio local."),
]


if __name__ == "__main__":
    # Carregar modelos
    mlp = joblib.load("mlp_classifier.pkl")
    bert_model = BertModel.from_pretrained("bert_finetuned_noticias")
    tokenizer = BertTokenizer.from_pretrained(
        "neuralmind/bert-large-portuguese-cased")
    encoder = joblib.load("label_encoder.pkl")

    # Classificar todas
noticias = coletar_noticias_g1()
# noticias = noticia_manual

resultados = []
for i, (titulo, texto, link, protocolo) in enumerate(noticias, start=1):
    classe = classificar_noticia(
        titulo, texto, mlp, bert_model, tokenizer, encoder)
    resultados.append(classe)

    print(f"Notícia {i}:")
    print(f"  Título    : {titulo}")
    print(f"  Classe    : {classe}")
    print(f"  Link      : {link}")
    print(f"  Protocolo : {protocolo}")

contagem_classes = pd.Series(resultados).value_counts()

# Mapear cores
cores = {
    "Boa": "green",
    "Ruim": "red",
    "Irrelevante": "gray"
}
# Obter a lista de cores na ordem correta
cores_ordenadas = [cores[classe] for classe in contagem_classes.index]

plt.figure(figsize=(8, 6))
contagem_classes.plot(kind="bar", color=cores_ordenadas)
plt.title("Classificação das Novas Notícias")
plt.xlabel("Classificação")
plt.ylabel("Quantidade")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
