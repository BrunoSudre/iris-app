from fastapi import FastAPI, Query
import pickle
import pandas as pd

app = FastAPI(title="IRIS Classifier API")


@app.get("/")
def homepage():
    return {"message": "Welcome to IRIS Classifier API"}


@app.get("/iris")
def iris_classifier(sepal_length: float = Query(description="Sepal Length", ge=4.3, le=7.9, default=4.3),
                    sepal_width: float = Query(description="Sepal Width", ge=2.0, le=4.4, default=2.0),
                    petal_length: float = Query(description="Petal Length", ge=1.0, le=6.9, default=1.0),
                    petal_width: float = Query(description="Petal Width", ge=0.1, le=2.5, default=0.1),
                    ):
    column_names = ["sepal length (cm)", "sepal width (cm)",
                    "petal length (cm)", "petal width (cm)"]
    data = [[sepal_length, sepal_width,
             petal_length, petal_width]]
    df = pd.DataFrame(data=data, columns=column_names)

    iris_pipe = pickle.load(open("objects/iris-pipe.pkl", "rb"))

    predictions = iris_pipe.predict(df)
    return {"prediction": predictions[0]}
