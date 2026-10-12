# Mantenimiento Predictivo con MLflow

Proyecto de Machine Learning desarrollado para la materia TC5061, enfocado en la implementación de un modelo de mantenimiento predictivo utilizando MLflow para el seguimiento y comparación de experimentos.

## Objetivo del proyecto

El objetivo de este proyecto es predecir fallas en maquinaria utilizando datos de condiciones operativas y sensores.

Para ello, se utiliza el dataset **AI4I 2020 Predictive Maintenance** y un modelo de clasificación basado en **Regresión Logística**.

## Dataset

Se utiliza el dataset:

**AI4I 2020 Predictive Maintenance Dataset**

El conjunto de datos contiene información relacionada con las condiciones de operación de una máquina, incluyendo variables como:

- Temperatura del aire
- Temperatura del proceso
- Velocidad de rotación
- Torque
- Desgaste de herramienta
- Indicadores de falla de máquina

## Modelo de Machine Learning

El algoritmo utilizado para la clasificación es **Regresión Logística**.

El programa permite modificar diferentes hiperparámetros desde la terminal:

- `C`
- `max_iter`
- `solver`
- `seed`

Ejemplo de ejecución:

python train.py --C 1 --max_iter 1000 --solver liblinear --seed 42

## Seguimiento de experimentos con MLflow

Se utiliza **MLflow** para registrar y comparar los diferentes experimentos realizados durante el entrenamiento del modelo.

Cada ejecución registra la siguiente información:

### Parámetros

- C
- max_iter
- solver
- seed

### Métricas

- Accuracy
- Precision
- Recall
- F1-score

### Artefactos

- Matriz de confusión
- Modelo entrenado

Esto permite comparar diferentes configuraciones del modelo directamente desde la interfaz de MLflow.

## Resultados obtenidos

Una de las configuraciones evaluadas obtuvo aproximadamente los siguientes resultados:

- Accuracy: 0.9685
- Precision: 0.6667
- Recall: 0.1471
- F1-score: 0.2410

Aunque el modelo presenta un **accuracy alto**, el recall y el F1-score son considerablemente menores.

Esto indica que el conjunto de datos presenta un desbalance entre clases y demuestra que utilizar únicamente accuracy puede generar una interpretación incompleta del desempeño del modelo.

Por esta razón, se registraron también las métricas de precision, recall y F1-score mediante MLflow.

## Comparación de experimentos

Se realizaron diferentes ejecuciones modificando los hiperparámetros del modelo, principalmente:

- Valor de `C`
- Número máximo de iteraciones
- Solver utilizado

MLflow permite visualizar y comparar los resultados de estas ejecuciones para analizar el efecto de los hiperparámetros sobre el desempeño del modelo.

## Estructura del proyecto

TC5061_MLflow/

- `train.py` - Entrenamiento del modelo y registro de experimentos en MLflow
- `ai4i2020.csv` - Dataset de mantenimiento predictivo
- `confusion_matrix.png` - Matriz de confusión generada durante la evaluación
- `README.md` - Documentación del proyecto
- `.gitignore` - Archivos excluidos del seguimiento de Git

## Ejecución del proyecto

Para entrenar el modelo:

python train.py --C 1 --max_iter 1000 --solver liblinear --seed 42

Para iniciar la interfaz de MLflow:

python -m mlflow ui

Posteriormente, abrir en el navegador:

http://127.0.0.1:5000

Desde la interfaz de MLflow es posible consultar los experimentos, parámetros, métricas, artefactos y modelos registrados.

## Tecnologías utilizadas

- Python
- Scikit-learn
- Pandas
- MLflow
- Matplotlib
- Git
- GitHub

## Autor

**Kevin Osmany Jonguitud Zumaya A01273017**