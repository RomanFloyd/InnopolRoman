import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import seaborn as sns
import sqlite3


def create_table(conn, path_to_df):
    df = pd.read_csv(path_to_df)
    df.to_sql('laptop_price', conn, if_exists='replace', index=False)


def check_tables(path):
    path_to_df = '../dataset/laptopPrice.csv'
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print(cursor.fetchall())

    if len(cursor.fetchall()) == 0:
        create_table(conn, path_to_df)


def load_data(path):
    conn = sqlite3.connect(path)
    query = "SELECT * FROM laptop_price"
    df = pd.read_sql_query(query, conn)
    return df


def plot_hist(df):
    for col in df.select_dtypes(include=[np.number]).columns:
        fig = px.histogram(df, x=col, marginal="box", nbins=50, title=f'Распределение {col}')
        st.plotly_chart(fig)


def plot_pairplot(df):
    # Используем seaborn для создания pairplot
    pairplot_fig = sns.pairplot(df.select_dtypes(include=[np.number])).fig
    st.pyplot(pairplot_fig)


def show_scatterplot(df):
    for col1 in df.select_dtypes(include=[np.number]).columns:
        for col2 in df.select_dtypes(include=[np.number]).columns:
            fig = px.scatter(df, x=col1, y=col2, title=f'Диаграмма рассеяния: {col1} vs {col2}')
            st.plotly_chart(fig)


def show_heatmap(df):
    df_numeric = df.select_dtypes(include=[np.number])
    corr = df_numeric.corr()
    fig = px.imshow(corr, x=corr.columns, y=corr.columns, title='Карта корреляции', labels=dict(color="Корреляция"))
    st.plotly_chart(fig)


def preprocess_data(df):
    labelencoder = LabelEncoder()
    df_categorical = df.select_dtypes(include=[object])
    df_categorical = df_categorical.apply(labelencoder.fit_transform)
    df_numerical = df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    df_numerical_scaled = scaler.fit_transform(df_numerical)
    df_numerical_scaled = pd.DataFrame(df_numerical_scaled, columns=df_numerical.columns)
    df_preprocessed = pd.concat([df_numerical_scaled, df_categorical], axis=1)
    return df_preprocessed


def plot_pca(df):
    df_preprocessed = preprocess_data(df)
    pca = PCA()
    pca.fit(df_preprocessed)
    fig = px.line(x=range(1, len(pca.explained_variance_ratio_) + 1), y=pca.explained_variance_ratio_,
                  labels={'x': 'Главная компонента', 'y': 'Доля объясненной дисперсии'},
                  title='Доля объясненной дисперсии PCA')
    st.plotly_chart(fig)


def train_model(df):
    df = pd.get_dummies(df, drop_first=True)
    y = df['Price']
    X = df.drop('Price', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    predictions_lr = model_lr.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(y_test, predictions_lr))
    r2_lr = r2_score(y_test, predictions_lr)

    model_rf = RandomForestRegressor()
    model_rf.fit(X_train, y_train)
    predictions_rf = model_rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(y_test, predictions_rf))
    r2_rf = r2_score(y_test, predictions_rf)

    return {'Линейная регрессия': [rmse_lr, r2_lr], 'Случайный лес': [rmse_rf, r2_rf]}


def main():
    check_tables('../database/laptopPrice.db')
    df = load_data('../database/laptopPrice.db')

    st.sidebar.title("Анализ набора данных о ценах на ноутбуки")
    st.sidebar.markdown("Это приложение позволяет исследовать набор данных о ценах на ноутбуки.")

    st.title("Набор данных о ценах на ноутбуки")

    st.markdown("""
    ## Описание приложения

    Это интерактивное веб-приложение предназначено для анализа набора данных о ценах на ноутбуки. 

    В данном приложении используются различные методы визуализации данных и машинного обучения для анализа и моделирования данных.

    - **Показать таблицу данных**: отображает весь набор данных в виде таблицы.
    - **Показать гистограммы**: отображает гистограммы всех числовых переменных.
    - **Показать парный график**: отображает парный график всех числовых переменных.
    - **Показать диаграмму рассеяния**: отображает диаграммы рассеяния для каждой пары числовых переменных.
    - **Показать тепловую карту**: отображает тепловую карту корреляции всех числовых переменных.
    - **Анализ PCA**: отображает долю объясненной дисперсии для каждой главной компоненты после выполнения анализа главных компонент.
    - **Обучить модель**: обучает модель линейной регрессии и случайного леса, а затем отображает среднеквадратическое отклонение и R2-оценку для каждой модели.
    """)

    if st.sidebar.button("Показать таблицу данных"):
        st.dataframe(df)

    if st.sidebar.button("Показать гистограммы"):
        plot_hist(df)

    if st.sidebar.button("Показать парный график"):
        plot_pairplot(df)

    if st.sidebar.button("Показать диаграмму рассеяния"):
        show_scatterplot(df)

    if st.sidebar.button("Показать корелляционную карту"):
        show_heatmap(df)

    if st.sidebar.button("Анализ PCA"):
        plot_pca(df)

    if st.sidebar.button("Обучить модель"):
        model_metrics = train_model(df)
        for model, metrics in model_metrics.items():
            st.sidebar.write(f'Метрики для {model}:')
            st.sidebar.write('Среднеквадратическое отклонение: ', metrics[0])
            st.sidebar.write('R2-оценка: ', metrics[1])

    if st.sidebar.button("Документация"):
        st.markdown("""
        ## Документация по приложению

        ### Описание приложения
        Это интерактивное веб-приложение предназначено для анализа набора данных о ценах на ноутбуки. Приложение разработано с использованием библиотеки Streamlit и предоставляет функционал по визуализации данных, анализу и обучению моделей машинного обучения.

        ### Установка и запуск
        Для запуска приложения необходимо установить следующие библиотеки:
        ```
        pip install streamlit pandas plotly scikit-learn seaborn sqlite3
        ```

        После установки библиотек выполните следующую команду в терминале для запуска приложения:
        ```
        streamlit run <имя_файла>.py
        ```

        ### Интерфейс приложения
        После запуска приложения откроется локальный сервер, и вы увидите интерфейс пользователя. В левой части окна находится боковая панель с кнопками для различных действий. Справа отображается содержимое выбранной панели.

        ### Функциональность
        Приложение предоставляет следующую функциональность:

        #### Показать таблицу данных
        Кнопка "Показать таблицу данных" отображает набор данных о ценах на ноутбуки в виде таблицы.

        #### Показать гистограммы
        Кнопка "Показать гистограммы" отображает гистограммы всех числовых переменных из набора данных. Гистограммы позволяют оценить распределение значений каждой переменной.

        #### Показать парный график
        Кнопка "Показать парный график" отображает парный график всех числовых переменных из набора данных. Парный график позволяет визуально оценить взаимосвязи между переменными.

        #### Показать диаграмму рассеяния
        Кнопка "Показать диаграмму рассеяния" отображает диаграммы рассеяния для каждой пары числовых переменных из набора данных. Диаграммы рассеяния позволяют оценить корреляцию между двумя переменными.

        #### Показать корреляционную карту
        Кнопка "Показать корреляционную карту" отображает тепловую карту корреляции всех числовых переменных из набора данных. Карта корреляции позволяет оценить силу и направление связей между переменными.

        #### Анализ PCA
        Кнопка "Анализ PCA" выполняет анализ главных компонент на наборе данных. Результатом анализа является график, отображающий долю объясненной дисперсии для каждой главной компоненты.

        #### Обучить модель
        Кнопка "Обучить модель" выполняет обучение моделей линейной регрессии и случайного леса на наборе данных о ценах на ноутбуки. После обучения моделей выводятся метрики качества, такие как среднеквадратическое отклонение (RMSE) и коэффициент детерминации (R2).

        ### Использование базы данных
        Приложение загружает данные о ценах на ноутбуки из базы данных SQLite. Файл базы данных `laptopPrice.db` должен находиться в подкаталоге `database` относительно расположения файла приложения.

        ### Заключение
        Данная документация описывает функциональность и использование приложения для анализа набора данных о ценах на ноутбуки. Приложение предоставляет удобный интерфейс для визуализации данных, анализа и обучения моделей машинного обучения. Вы можете использовать это приложение для изучения и анализа данных о ценах на ноутбуки.
        """)


if __name__ == "__main__":
    main()
