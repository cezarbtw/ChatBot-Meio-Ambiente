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


def extrair_data_do_link(link):
    import re
    match = re.search(r'/noticia/(\d{4}/\d{2}/\d{2})/', link)
    if match:
        return match.group(1).replace('/', '-')
    return "Data não disponível"


def coletar_noticias_g1():
    url = 'https://g1.globo.com/meio-ambiente/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    noticias_coletadas = []
    artigos = soup.select('.feed-post')[:10]
    for artigo in artigos:
        titulo_tag = artigo.select_one('.feed-post-body-title')
        link_tag = artigo.select_one('a')
        texto_tag = artigo.select_one('.feed-post-body-resumo')

        if titulo_tag and link_tag:
            titulo = titulo_tag.get_text(strip=True)
            texto = texto_tag.get_text(strip=True) if texto_tag else ""
            link = link_tag.get('href')

            protocolo = link.split(':')[0] if link else "desconhecido"
            data = extrair_data_do_link(link)

            noticias_coletadas.append((titulo, texto, link, protocolo, data))

    return noticias_coletadas


noticia_manual = [
    ("NO satélite que vai descobrir quanto a Amazônia 'pesa'",
     "Sistema de observação conta com um radar especial que vai medir a quantidade de gás armazenado pelas árvores nas florestas tropicais."),
    ("Financiamento, metas e combustíveis: veja três desafios que o governo Lula prevê nas negociações da COP30",
     "País sediará conferência do clima em novembro deste ano, em Belém (PA). Lula tem dito que planeta está 'farto' de promessas não cumpridas e que é o momento de agir."),
]

mlp = joblib.load("mlp_classifier.pkl")
bert_model = BertModel.from_pretrained("bert_finetuned_noticias")
tokenizer = BertTokenizer.from_pretrained(
    "neuralmind/bert-large-portuguese-cased")
encoder = joblib.load("label_encoder.pkl")

noticias = coletar_noticias_g1()

resultados = []
resultados = []
dados_resultado = []

for i, dados in enumerate(noticias, start=1):
    if len(dados) == 2:
        titulo, texto = dados
        link = "Não informado"
        protocolo = "Não informado"
        data = "Não informado"
    elif len(dados) == 5:
        titulo, texto, link, protocolo, data = dados
    else:
        raise ValueError("Formato de notícia inválido.")

    classe = classificar_noticia(
        titulo, texto, mlp, bert_model, tokenizer, encoder)
    resultados.append(classe)

    dados_resultado.append({
        "Data": data,
        "Classe": classe,
        "Título": titulo,
        "Link": link
    })

    print(f"Notícia {i}:")
    print(f"  Título    : {titulo}")
    print(f"  Classe    : {classe}")
    print(f"  Link      : {link}")
    print(f"  Protocolo : {protocolo}")
    print(f"  Data      : {data}")


contagem_classes = pd.Series(resultados).value_counts()

cores = {
    "Boa": "green",
    "Ruim": "red",
    "Irrelevante": "gray"
}
df_resultados = pd.DataFrame(dados_resultado)
df_resultados = df_resultados[df_resultados["Data"] != "Data não disponível"]

grupo = df_resultados.groupby(["Data", "Classe"]).size().unstack(fill_value=0)

for data, linha in grupo.iterrows():
    plt.figure(figsize=(10, 5))
    linha.plot(kind="bar", color=[cores.get(
        classe, "gray") for classe in linha.index])
    plt.title(f"Classificação das Notícias em {data}")
    plt.xlabel("Classificação")
    plt.ylabel("Quantidade")
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()
