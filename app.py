import streamlit as st
from model import load_model_and_predict
import io


def mainpage():
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Pafnutij_AI",
    )
    st.title('Pafnutij_AI')
    st.text("Сверточная нейросеть, предсказывает наличие опухоли по МРТ снимкам головного мозга")
    st.subheader('Ваши данные')
    side_bar()
    file_upload()


def file_upload():
    uploaded_file = st.file_uploader("Отправьте МРТ-снимок", ["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        answer(io.BytesIO(bytearray(uploaded_file.read())))


def answer(picture):
    st.subheader('Предсказание модели')
    st.image(picture)
    st.text(load_model_and_predict(picture))


def side_bar():
    st.sidebar.header('Предыстория проекта:')
    st.sidebar.text(
        """
        Проект был выполнен 
        в рамках программы bonus_track 
        университета ИТМО
        """)
    st.sidebar.header('Команда проекта:')
    st.sidebar.text(
        """
        Богодист Всеволод
        Шварцмен Анастасия
        Орехова Дарья
        """)
    st.sidebar.subheader('Контакты:')
    st.sidebar.text(
        """
        мяу мяу мяу
        у меня лапки
        """)


mainpage()
