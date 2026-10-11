import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
def get_arguments():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de modelo de mantenimiento predictivo"
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Parametro de regularizacion de Logistic Regression"
    )

    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Numero maximo de iteraciones"
    )

    parser.add_argument(
        "--solver",
        type=str,
        default="liblinear",
        choices=["liblinear", "lbfgs"],
        help="Algoritmo utilizado para optimizar el modelo"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria"
    )

    return parser.parse_args()
def load_and_prepare_data(file_path):
    # Cargar dataset
    df = pd.read_csv(file_path)

    # Eliminar identificadores y variables que pueden generar data leakage
    df_model = df.drop(
        columns=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
    )

    # Convertir la variable Type a valores numericos
    df_model = pd.get_dummies(
        df_model,
        columns=["Type"],
        drop_first=True,
        dtype=int
    )

    # Separar variables predictoras y variable objetivo
    X = df_model.drop("Machine failure", axis=1)
    y = df_model["Machine failure"]

    return X, y
def train_model(X, y, C, max_iter, solver, seed):

    # Separar datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed,
        stratify=y
    )

    # Crear el modelo
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver=solver,
        random_state=seed
    )

    # Entrenar
    model.fit(X_train, y_train)

    return model, X_test, y_test
def evaluate_model(model, X_test, y_test):

    # Realizar predicciones
    y_pred = model.predict(X_test)

    # Calcular metricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # Crear matriz de confusion
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Failure", "Failure"]
    )

    disp.plot()
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Guardar la matriz como imagen
    confusion_matrix_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    plt.close()

    return accuracy, precision, recall, f1, confusion_matrix_path
def main():

    # Leer argumentos de la terminal
    args = get_arguments()

    # Nombre del experimento en MLflow
    mlflow.set_experiment("Mantenimiento_Predictivo")

    # Cargar y preparar datos
    X, y = load_and_prepare_data("ai4i2020.csv")

    # Iniciar una corrida en MLflow
    with mlflow.start_run():

        # Entrenar modelo
        model, X_test, y_test = train_model(
            X,
            y,
            args.C,
            args.max_iter,
            args.solver,
            args.seed
        )

        # Evaluar modelo
        accuracy, precision, recall, f1, cm_path = evaluate_model(
            model,
            X_test,
            y_test
        )

        # Guardar hiperparametros en MLflow
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("solver", args.solver)
        mlflow.log_param("seed", args.seed)

        # Guardar metricas en MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Guardar matriz de confusion
        mlflow.log_artifact(cm_path)

        # Guardar modelo
        mlflow.sklearn.log_model(model, "model")

        # Mostrar resultados en terminal
        print("Entrenamiento terminado")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
if __name__ == "__main__":
    main()