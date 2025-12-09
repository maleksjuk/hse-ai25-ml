import streamlit as st
import pandas as pd
import pickle
from pathlib import Path
import os
from sklearn.linear_model import Ridge, ElasticNet, LinearRegression, Lasso
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

FILE_FORMAT = '.pkl'
MODEL_DIR = Path(__file__).resolve().parent / "models"
FEATURE_NAMES_SUFFIX = "_feature_names"
SCALER_SUFFIX = "_scaler"
DEFAULT_MODEL_NAME = "ElasticNet_alpha_0.70_l1_ratio_0.80"

@st.cache_resource
def load_model(model_name):

    base_models = {
        'elastic': ElasticNet(),
        'lasso': Lasso(),
        'ridge': Ridge(),
        'linear': LinearRegression()
    }
    for base_model_name, base_model in base_models.items():
        if model_name.lower().find(base_model_name) >= 0:
            model = base_model
            break
    scaler = StandardScaler()

    model_path = MODEL_DIR / f'{model_name}{FILE_FORMAT}'
    feature_path = MODEL_DIR / f'{model_name}{FEATURE_NAMES_SUFFIX}{FILE_FORMAT}'
    scaler_path = MODEL_DIR / f'{model_name}{SCALER_SUFFIX}{FILE_FORMAT}'

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(feature_path, 'rb') as f:
        feature_names = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    print(feature_names)

    return model, feature_names, scaler


def prepare_features(df, feature_names):
    # return pd.DataFrame(scaler.transform(df[feature_names]), columns=feature_names)
    return df[feature_names]


if __name__ == "__main__":
    st.set_page_config(page_title="Homework 1")
    st.title("Домашнее задание 1. Предсказание стоимости автомобилей")

    model_filenames = [filename.replace(FILE_FORMAT, '') for filename in os.listdir(MODEL_DIR) if filename.find(FEATURE_NAMES_SUFFIX) == -1 and filename.find(SCALER_SUFFIX) == -1]
    try:
        default_index = model_filenames.index(f'{DEFAULT_MODEL_NAME}')
    except:
        default_index = None
    # model_name = st.selectbox("Model", model_filenames, default_index)
    model_name = DEFAULT_MODEL_NAME
    st.text(f'Модель: {model_name}')

    try:
        MODEL, FEATURE_NAMES, SCALER = load_model(model_name)
    except Exception as e:
        st.error(f'Ошибка загрузки модели: {e}')
        st.stop()


    # weights
    st.subheader("Параметры модели")
    df_weight = pd.DataFrame(MODEL.coef_, FEATURE_NAMES).T
    st.table(df_weight)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(FEATURE_NAMES, MODEL.coef_)
    ax.axhline(0)
    ax.set_title('Веса обученной модели')
    ax.set_xlabel('Параметр')
    ax.set_ylabel('Вес')
    ax.grid(axis='y')
    st.pyplot(fig)


    # EDA
    st.subheader("Графики EDA")
    st.image('images/corr.png', 'Матрица корреляций по тренировочным данным')
    st.image('images/pairplot.png', 'Матрица попарных зависимостей по тренировочным данным')


    # input data
    st.subheader("Ввод данных")

    input_method = st.radio('input', ['CSV', 'manual'])
    df = None

    if input_method == 'CSV':
        uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])
        if uploaded_file is None:
            st.info("Загрузите CSV файл для начала работы")
            st.stop()
        df = pd.read_csv(uploaded_file).drop('Unnamed: 0', axis=1)
    elif input_method == 'manual':
        with st.form("prediction_form"):
            manual_data = {key: [st.number_input(key, step=1.,
                                                 format="%.0f" if key in ['year', 'seats'] else None)]
                            for key in FEATURE_NAMES}
            if st.form_submit_button('Run'):
                print(manual_data)
                df = pd.DataFrame(manual_data)


    # apply model
    try:
        if df is not None:
            features = prepare_features(df, FEATURE_NAMES)
            features_scaled = pd.DataFrame(SCALER.transform(features), columns=FEATURE_NAMES)
            predictions = MODEL.predict(features_scaled)
            df['prediction'] = predictions
            st.badge("Success", icon=":material/check:", color="green")
    except Exception as e:
        st.error(f"Ошибка при обработке данных: {e}")
        st.stop()


    # results
    if df is not None:
        st.subheader("Результаты")
        st.dataframe(df)
