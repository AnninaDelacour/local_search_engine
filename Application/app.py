import pandas as pd
import streamlit as st
from utils import build_inverted_index, calculate_tfidf, search_document, return_found_information

st.set_page_config(page_title="Local Search Engine", page_icon="🔎", layout="wide")
st.title("Local Search Engine")
text_search = st.text_input("Search Term", value="")

spielerinnen_df = pd.read_csv('../Spielerinnendaten.csv')
player_documents = spielerinnen_df[['Dressnummer', 'Name', 'Position', 'Nationalitaet', 'Groesse']].values.tolist()
player_inverted_index = build_inverted_index(player_documents)
player_tfidf_df = calculate_tfidf(player_documents, player_inverted_index)
ranked_docs = search_document(player_tfidf_df, text_search)

st.write(return_found_information(spielerinnen_df, ranked_docs, text_search))