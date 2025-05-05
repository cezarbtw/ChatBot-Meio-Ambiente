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

# Dividir treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42)

# Função para gerar vetor [CLS]


def gerar_embedding(texto):
    inputs = tokenizer(texto, return_tensors="pt",
                       truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(
        0)
    return cls_embedding.numpy()


# Gerar embeddings
X_train_embeddings = np.array([gerar_embedding(texto) for texto in X_train])
X_test_embeddings = np.array([gerar_embedding(texto) for texto in X_test])

# Treinar MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Camadas ocultas
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
    early_stopping=True
)

mlp.fit(X_train_embeddings, y_train)

# Avaliação
y_pred = mlp.predict(X_test_embeddings)

print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(
    y_test, y_pred, target_names=encoder.classes_))

joblib.dump(mlp, "mlp_classifier.pkl")
bert_model.save_pretrained("bert_finetuned_noticias")
joblib.dump(encoder, "label_encoder.pkl")


def classificar_nova_noticia(titulo, texto):
    # Carregar os modelos
    mlp = joblib.load("mlp_classifier.pkl")
    bert_model = BertModel.from_pretrained("bert_finetuned_noticias")
    tokenizer = BertTokenizer.from_pretrained(
        "neuralmind/bert-large-portuguese-cased")
    encoder = joblib.load("label_encoder.pkl")

    titulo = clean_text(titulo)
    texto = clean_text(texto)

    conteudo = titulo + ". " + texto

    # Tokenizar e gerar embedding
    inputs = tokenizer(conteudo, return_tensors="pt",
                       truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(
        0).numpy().reshape(1, -1)

    # Classificar
    pred = mlp.predict(cls_embedding)
    return encoder.inverse_transform(pred)[0]


titulo = "Pesquisadores criam método inovador"
texto = "para purificar água usando algas naturais."
print("Classificação da nova notícia:",
      classificar_nova_noticia(titulo, texto))
