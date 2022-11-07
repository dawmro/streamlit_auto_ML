import streamlit as st
import pandas as pd
import os

# import profiling capability
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

# ML stuff
from pycaret.classification import setup, compare_models, pull, save_model

with st.sidebar:
	st.image("https://previews.123rf.com/images/kirillm/kirillm1606/kirillm160600012/60317004-retro-robot-reading-books-isolated-3d-illustration-.jpg")
	st.title("AutoStreamML")
	choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
	st.info("This app has been designed to help you build automated ML pipeline using Streamlit, Pandas Profiling and PyCaret.")

if os.path.exists("source_data.csv"):
	df = pd.read_csv("source_data.csv", index_col = None)

if choice == "Upload":
	st.title("Upload Data for Modelling")
	file = st.file_uploader("Upload Your Dataset")
	if file:
		df = pd.read_csv(file, index_col = None)
		st.dataframe(df)
		df.to_csv("source_data.csv", index = None)
	
if choice == "Profiling":
	st.title("Automated Exploratory Data Analysis")
	profile_df = df.profile_report()
	st_profile_report(profile_df)

if choice == "ML":
	st.title("Machine Learning")
	chosen_target = st.selectbox("Select Your Target", df.columns)
	if st.button('Run Modelling'): 
		setup(df, target=chosen_target, silent=True)
		setup_df = pull()
		st.info("Experiment ML Settings")
		st.dataframe(setup_df)
		best_model = compare_models()
		compare_df = pull()
		st.info("ML Model")
		st.dataframe(compare_df)
		save_model(best_model, 'best_model')


if choice == "Download":
	with open('best_model.pkl', 'rb') as f: 
		st.download_button('Download Model', f, file_name="best_model.pkl")

