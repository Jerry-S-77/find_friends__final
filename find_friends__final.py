import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore

# Podmieniam model, dane i nazwy klastrów z pierwotnej wersji v1 na v2
MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'

DATA = 'welcome_survey_simple_v2.csv'

CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

# Dodaję kolejność wieku do pierwotnej wersji, bo alfabetycznie "<18" jest po "55-64"
AGE_ORDER = ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65', 'unknown']


@st.cache_data
def get_model():
    return load_model(MODEL_NAME)

@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())

@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)

    return df_with_clusters

# Dodaję tytuł i krótki opis do pierwotnej wersji
st.title(':handshake: Znajdź znajomych')
st.write("W tej aplikacji można sprawdzić, do której grupy uczestników kursu jest Ci najbliżej.")

with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", AGE_ORDER)
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty', 'Inne', 'Koty i Psy'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach', 'Inne'])
    gender = st.radio("Płeć", ['Mężczyzna', 'Kobieta'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender,
        }
    ])

model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions[predicted_cluster_id]

st.header(f"Najbliżej Ci do grupy {predicted_cluster_data['name']}")
st.markdown(predicted_cluster_data['description'])
same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
# Dodaję liczbę mężczyzn i kobiet w grupie do pierwotnej wersji
c0, c1, c2 = st.columns(3)
with c0:
    st.metric("Liczba twoich znajomych", len(same_cluster_df), border=True)

with c1:
    st.metric("Liczba mężczyzn", len(same_cluster_df[same_cluster_df["gender"] == "Mężczyzna"]), border=True)

with c2:
    st.metric("Liczba kobiet", len(same_cluster_df[same_cluster_df["gender"] == "Kobieta"]), border=True)

st.header("Osoby z grupy")
# Dodaję zakładki do pierwotnej wersji, żeby wykresy nie były jeden pod drugim
tab0, tab1, tab2, tab3, tab4 = st.tabs(["Wiek", "Wykształcenie", "Ulubione zwierzęta", "Ulubione miejsce", "Płeć"])

with tab0:
    # Dodaję kolejność wieku na wykresie zamiast sortowania z pierwotnej wersji
    fig = px.histogram(same_cluster_df, x="age", category_orders={"age": AGE_ORDER})
    fig.update_layout(
        title="Rozkład wieku w grupie",
        xaxis_title="Wiek",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)

with tab1:
    fig = px.histogram(same_cluster_df, x="edu_level")
    fig.update_layout(
        title="Rozkład wykształcenia w grupie",
        xaxis_title="Wykształcenie",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)

with tab2:
    fig = px.histogram(same_cluster_df, x="fav_animals")
    fig.update_layout(
        title="Rozkład ulubionych zwierząt w grupie",
        xaxis_title="Ulubione zwierzęta",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)

with tab3:
    fig = px.histogram(same_cluster_df, x="fav_place")
    fig.update_layout(
        title="Rozkład ulubionych miejsc w grupie",
        xaxis_title="Ulubione miejsce",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)

with tab4:
    fig = px.histogram(same_cluster_df, x="gender")
    fig.update_layout(
        title="Rozkład płci w grupie",
        xaxis_title="Płeć",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)
