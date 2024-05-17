import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import json
from wordcloud import WordCloud
from nltk import casual_tokenize
# Changing the Layout
st.set_page_config( layout="wide")


with open('config/filepaths.json') as f:
    FPATHS = json.load(f)

st.title("Exploratory Data Analysis of Movie Reviews")

st.divider()
st.subheader("ScatterText:")
@st.cache_data
def load_scattertext(fpath):
    with open(fpath) as f:
        explorer = f.read()
        return explorer


# Load scattertext from filepaths
html_to_show = load_scattertext(FPATHS['eda']['scattertext'])

# Checkbox to trigger display of scattertext
checkbox_scatter = st.checkbox("Show Scattertext Explorer",value=False)


if checkbox_scatter:     
    components.html(html_to_show, width=1200, height=800, scrolling=True)
else:
    st.empty()

