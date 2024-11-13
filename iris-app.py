import streamlit as st
import requests

url = "https://iris-app-s8si.onrender.com/iris"


def main():
    st.title('Iris Classifier')
    st.sidebar.title('Parameters')

    sepal_length = st.sidebar.slider('Sepal Length (cm)', min_value=4.3, max_value=7.9, step=0.1)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', min_value=2.0, max_value=4.4, step=0.1)
    petal_length = st.sidebar.slider('Petal Length (cm)', min_value=1.0, max_value=6.9, step=0.1)
    petal_width = st.sidebar.slider('Petal Width (cm)', min_value=0.1, max_value=2.5, step=0.1)

    query_str = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width,
    }

    if st.sidebar.button("Predict"):
        response = requests.get(url, params=query_str)
        results = response.json()

        status_code = response.status_code
        if status_code == 200:
            print(results)
            prediction = results["prediction"]
            st.write(f"**Predicted class**: :blue[**{prediction}**]")
            st.image(f'{prediction}.jpeg')
            st.balloons()
        else:
            print(f"Error: {response.status_code}")


if __name__ == '__main__':
    st.set_page_config(
        page_title="Iris Classifier",
        page_icon=":male-technologist:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.collegelasalle.com',
            'Report a bug': "https://www.collegelasalle.com",
            'About': "# IRIS Classifier. A Neural Network classifier on IRIS dataset"
        }
    )
    main()
