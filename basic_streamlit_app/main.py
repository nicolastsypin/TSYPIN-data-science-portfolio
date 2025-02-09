import streamlit as st
import pandas as pd

st.title("Let's look at penguins today!")
st.subheader("In this streamlit app we will explore a fun penguin data set!")

penguins = pd.read_csv("basic_streamlit_app/data/penguins.csv")

st.write("Here are our penguins: ")
st.dataframe(penguins)

species = st.selectbox("Select a species:", penguins['species'].unique())
filtered_penguins = penguins[penguins['species'] == species]
st.write("Penguins of this species: ")
st.dataframe(filtered_penguins)

salary = st.slider("Choose a body mass (in grams) range:", int(penguins['body_mass_g'].min()), int(penguins['body_mass_g'].max()))
st.write(f"Penguins with body mass <= {salary} grams: ")
st.dataframe(penguins[penguins['body_mass_g'] <= salary])