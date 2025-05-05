import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from time import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("noticias_classificadas.csv")

df['conteudo'] = df['titulo'] + " " + df['texto']

vectorizer = TfidfVectorizer(max_features=1000)
x_data = vectorizer.fit_transform(df['conteudo'])

y_data = df['rotulo']

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.3, random_state=42)

print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


def mostrar_desempenho(x_train, y_train, x_test, y_test, model, name):
    inicio = time()
    model.fit(x_train, y_train)
    fim = time()
    tempo_treinamento = (fim - inicio) * 1000

    inicio = time()
    y_predicted = model.predict(x_test)
    fim = time()
    tempo_previsao = (fim - inicio) * 1000

    print(f'\nRelatório utilizando algoritmo {name}')
    print('\nMatriz de Confusão:')
    print(confusion_matrix(y_test, y_predicted))
    print('\nRelatório de Classificação:')
    print(classification_report(y_test, y_predicted))

    accuracy = accuracy_score(y_test, y_predicted)
    relatorio = classification_report(y_test, y_predicted, output_dict=True)

    print('Accuracy:', accuracy)
    print('Precision:', relatorio['macro avg']['precision'])
    print('Tempo de treinamento (ms):', tempo_treinamento)
    print('Tempo de previsão (ms):', tempo_previsao)

    return accuracy, tempo_treinamento, tempo_previsao


model_mlp = MLPClassifier(hidden_layer_sizes=(
    10, 10, 10), max_iter=1000, random_state=42)
acc_mlp, tt_mlp, tp_mlp = mostrar_desempenho(
    x_train, y_train, x_test, y_test, model_mlp, 'MLP')

model_arvore = tree.DecisionTreeClassifier(random_state=42)
acc_dt, tt_dt, tp_dt = mostrar_desempenho(
    x_train, y_train, x_test, y_test, model_arvore, 'DecisionTree')

model_rf = RandomForestClassifier(
    max_depth=5, n_estimators=10, max_features=1, random_state=42)
acc_rf, tt_rf, tp_rf = mostrar_desempenho(
    x_train, y_train, x_test, y_test, model_rf, 'RandomForest')

model_ada = AdaBoostClassifier(random_state=42)
acc_ada, tt_ada, tp_ada = mostrar_desempenho(
    x_train, y_train, x_test, y_test, model_ada, 'AdaBoost')

model_knn = KNeighborsClassifier(n_neighbors=5)
acc_knn, tt_knn, tp_knn = mostrar_desempenho(
    x_train, y_train, x_test, y_test, model_knn, 'KNN')

model_lr = LogisticRegression(max_iter=1000, random_state=42)
acc_lr, tt_lr, tp_lr = mostrar_desempenho(
    x_train, y_train, x_test, y_test, model_lr, 'LogisticRegression')

model_svm = SVC(random_state=42)
acc_svm, tt_svm, tp_svm = mostrar_desempenho(
    x_train, y_train, x_test, y_test, model_svm, 'SVM')

model_gnb = GaussianNB()
acc_gnb, tt_gnb, tp_gnb = mostrar_desempenho(
    x_train.toarray(), y_train, x_test.toarray(), y_test, model_gnb, 'GaussianNB')

model_lda = LinearDiscriminantAnalysis()
acc_lda, tt_lda, tp_lda = mostrar_desempenho(
    x_train.toarray(), y_train, x_test.toarray(), y_test, model_lda, 'LDA')

model_qda = QuadraticDiscriminantAnalysis()
acc_qda, tt_qda, tp_qda = mostrar_desempenho(
    x_train.toarray(), y_train, x_test.toarray(), y_test, model_qda, 'QDA')

algoritmos = ['GaussianNB', 'MLP', 'DecisionTree', 'KNN',
              'LogReg', 'LDA', 'SVM', 'RandomForest', 'AdaBoost', 'QDA']
accs = [acc_gnb, acc_mlp, acc_dt, acc_knn, acc_lr,
        acc_lda, acc_svm, acc_rf, acc_ada, acc_qda]

plt.figure(figsize=(10, 6))
plt.bar(algoritmos, accs)
plt.title('Acurácia dos Modelos')
plt.ylabel('Acurácia')
plt.xticks(rotation=45)
plt.show()

tts = [tt_gnb, tt_mlp, tt_dt, tt_knn, tt_lr,
       tt_lda, tt_svm, tt_rf, tt_ada, tt_qda]
plt.figure(figsize=(10, 6))
plt.bar(algoritmos, tts)
plt.title('Tempo de Treinamento (ms)')
plt.ylabel('Tempo (ms)')
plt.xticks(rotation=45)
plt.show()

tps = [tp_gnb, tp_mlp, tp_dt, tp_knn, tp_lr,
       tp_lda, tp_svm, tp_rf, tp_ada, tp_qda]
plt.figure(figsize=(10, 6))
plt.bar(algoritmos, tps)
plt.title('Tempo de Previsão (ms)')
plt.ylabel('Tempo (ms)')
plt.xticks(rotation=45)