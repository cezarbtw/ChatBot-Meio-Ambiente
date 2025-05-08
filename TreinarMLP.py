import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np
import matplotlib.pyplot as plt


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\\S+|www\\S+|https\\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text.lower()


df = pd.read_csv("noticias_classificadas.csv")
df["titulo"] = df["titulo"].apply(clean_text)
df["texto"] = df["texto"].apply(clean_text)
df["conteudo"] = df["titulo"] + ". " + df["texto"]

X = df["conteudo"]
y = df["rotulo"]

tokenizer = BertTokenizer.from_pretrained(
    "neuralmind/bert-large-portuguese-cased")
bert_model = BertModel.from_pretrained(
    "neuralmind/bert-large-portuguese-cased")
bert_model.eval()

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)


def gerar_embedding(texto):
    inputs = tokenizer(texto, return_tensors="pt",
                       truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(
        0)
    return cls_embedding.numpy()


X_train_embeddings = np.array([gerar_embedding(texto) for texto in X_train])
X_test_embeddings = np.array([gerar_embedding(texto) for texto in X_test])

mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
    early_stopping=True
)

mlp.fit(X_train_embeddings, y_train)

y_pred = mlp.predict(X_test_embeddings)

print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(
    y_test, y_pred, target_names=encoder.classes_))

joblib.dump(mlp, "mlp_classifier.pkl")
bert_model.save_pretrained("bert_finetuned_noticias")
joblib.dump(encoder, "label_encoder.pkl")


def classificar_nova_noticia(titulo, texto):
    mlp = joblib.load("mlp_classifier.pkl")
    bert_model = BertModel.from_pretrained("bert_finetuned_noticias")
    tokenizer = BertTokenizer.from_pretrained(
        "neuralmind/bert-large-portuguese-cased")
    encoder = joblib.load("label_encoder.pkl")

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


titulo = "Pesquisadores criam método inovador"
texto = "para purificar água usando algas naturais."
print("Classificação da nova notícia:",
      classificar_nova_noticia(titulo, texto))


palavras_chave = ["incêndios", "desmatamento",
                  "aquecimento", "temperatura", "poluição"]

if "conteudo" not in df.columns:
    df["conteudo"] = (df["titulo"] + ". " + df["texto"]).str.lower()

for palavra in palavras_chave:
    df[palavra] = df["conteudo"].str.count(palavra)

ocorrencias = df.groupby("rotulo")[palavras_chave].sum()

ocorrencias_percentual = ocorrencias.div(ocorrencias.sum(axis=0), axis=1) * 100

ocorrencias_percentual.plot(kind="bar", figsize=(12, 6))
plt.title("Distribuição Percentual das Palavras-Chave por Tipo de Notícia")
plt.ylabel("Porcentagem (%)")
plt.xlabel("Classificação da Notícia")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
