import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from deep_translator import GoogleTranslator
import os

from langchain.chains import LLMChain

from langchain.prompts.prompt import PromptTemplate

from langchain.llms import OpenAI

import openai

 

os.environ["OPENAI_API_KEY"] = "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"

openai.organization = "org-27oOLigIQ3RXF0GiWNMAKeXN"
st.set_page_config(layout="wide")

@st.cache_data()
def load_data():
    data = pd.read_csv("complete_results.csv")
    return data

data = load_data()

@st.cache_data
def translate_text(text, target_language):
    prompt = '''Translate the following text to German:\n\n/"{text}/"'''


    DOCUMENT_DESCRIPTION_PROMPT = PromptTemplate(template=prompt, input_variables=["text"])

    llm = OpenAI()

    xx = LLMChain(llm=llm, prompt=DOCUMENT_DESCRIPTION_PROMPT)

    # translator =  GoogleTranslator(source='auto', target=target_language)
    # translated = translator.translate(text)
    translated = xx.run({'text':  text})
    print(translated)
    return translated   
@st.cache_data
def translate_dataframe(df, target_language):
    df_copy = df.copy()
   
    for col in df_copy.columns:
        if df_copy[col].dtype == "object":
            # df_copy[col] = df_copy[col].apply(lambda x: )
            df_copy[col] = df_copy[col].apply(lambda x: translate_text(x, target_language))
    return df_copy

@st.cache_data
def save_data(df):
    df.to_csv("sample.csv", index=False)

gb = GridOptionsBuilder()
# Rest of the configuration code...
gb.configure_default_column(
    resizable=True,
    filterable=True,
    sortable=True,
    editable=False,
    wrapText=True,
    autoHeight=True,
)
# print("data")

gb.configure_column(
    
    field="Unnamed: 0",  # Use the first column name as the index column
    header_name="#",  # Use the first column name as the header
    width=50,
    resizable=True,
    tooltipField="Unnamed: 0",
    lockVisible=True,
)


gb.configure_column(
    field="Question",
    header_name="Question",
    width=300,
    tooltipField="Question",
    enableFilter=True,
    editable=True,
)

# Make sure the column names match your DataFrame exactly
gb.configure_column(
    field="True_Answer",  # Check the correct column name in your DataFrame
    header_name="True Answer",
    width=300,
    tooltipField="true_answer",
    enableFilter=True,
    editable=True,
)
gb.configure_column(
    field="Predicted_Answer",  # Check the correct column name in your DataFrame
    header_name="Predicted Answer",
    width=300,
    tooltipField="predicted_answer",
    enableFilter=True,
    editable=True,

)
gb.configure_column(
    field="Grade",
    header_name="Grade",
    width=200,
    valueFormatter="value != undefined ? value : ''",
    pivot=False,
    enableFilter=True,
    editable=True,
)
gb.configure_column(
    field="Comments",
    header_name="Comments",
    width=200,
    enableFilter=True,
    editable=True,
)

gb.configure_grid_options(
    tooltipShowDelay=0,
    
)


# Build the grid options
go = gb.build()

# Display the ag-Grid and capture user edits
if st.button("Translate"):
    translated_data = translate_dataframe(data, target_language='de')  # Translate the original DataFrame
    data = AgGrid(data=translated_data, gridOptions=go, key='df', autoHeight=True)  # Display the translated DataFrame
    # df = data["data"]
    # save_data(translated_data)
     
else:
    new_data = AgGrid(data=load_data(), gridOptions=go, key='df', autoHeight=True)  # Display the translated DataFrame
    df = new_data["data"]
    # df.to_csv('2.csv', index=False) 
    # data = load_data()
data = pd.read_csv("complete_results.csv")
# Now you can use the updated DataFrame 'translated_data' for further processing
# For example, you can save the translated DataFrame back to the CSV file after user edits


# Check if the CSV is updated
df = pd.read_csv('complete_results.csv')
# st.write(df)
