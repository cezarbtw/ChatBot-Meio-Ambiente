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


# Teste rápido
if __name__ == "__main__":
    titulo = "Homem tenta atravessar oceano em bola inflável gigante"
    texto = "Um aventureiro decidiu cruzar o Oceano Atlântico usando apenas uma bola inflável gigante, afirmando que quer provar a resistência do plástico utilizado na fabricação do objeto. Autoridades locais acompanham o caso com curiosidade."
    classificacao = classificar_noticia(titulo, texto)
    print(f"Classificação: {classificacao}")
