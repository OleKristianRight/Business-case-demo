import os
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
import pandas as pd
from data_cleaning import clean_data

# Last inn miljøvariabler
load_dotenv()
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
TARGET_URL       = os.getenv("target_url")        # https://<resource>.openai.azure.com/
DEPLOYMENT_NAME  = os.getenv("DEPLOYMENT_NAME")   # f.eks. "gpt-4o-mini"

# Sett opp AzureOpenAI-klienten
client = AzureOpenAI(
    api_key        = OPENAI_API_KEY,
    azure_endpoint = TARGET_URL,
    api_version    = "2024-02-15-preview"
)

# Initialiser historikk i session state hvis den ikke finnes
if 'historikk' not in st.session_state:
    st.session_state.historikk = []

st.set_page_config(page_title="Prompt til Output – Coast Seafood", layout="centered")

# Opprett to kolonner
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
        <h1 style='text-align: center; font-size: 3em;'>🧠 Prompt til Output –<br>Coast Seafood</h1>
    """, unsafe_allow_html=True)

    # Legg til Excel-opplastingsfelt
    uploaded_file = st.file_uploader("Last opp Excel-fil", type=['xlsx', 'xls'])
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"Filen '{uploaded_file.name}' er lastet opp!")
            st.dataframe(df)
            
            # Legg til data cleaning-knapp
            if st.button("Rens data"):
                with st.spinner("Renser data..."):
                    df_cleaned = clean_data(df)
                    st.success("Data er nå renset! (Duplikater og NaN-verdier er fjernet)")
                    st.dataframe(df_cleaned)
                    
                    # Legg til nedlastingsmulighet for renset data
                    csv = df_cleaned.to_csv(index=False)
                    st.download_button(
                        label="Last ned renset data som CSV",
                        data=csv,
                        file_name="renset_data.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Det oppstod en feil ved lesing av Excel-filen: {e}")

    prompt = st.text_input("Skriv inn din prompt her:", placeholder="F.eks, Hva er hovedstaden i Norge")

    st.markdown("<b>Innstillinger</b>", unsafe_allow_html=True)
    temperatur = st.slider("Temperatur", 0.0, 1.0, 0.0, 0.01)
    st.caption("Styrer hvor kreative svarene skal være")
    
    modell = st.selectbox(
        "Velg modell",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0,
        help="Velg hvilken modell som skal brukes"
    )

    st.markdown("<b>Output</b>", unsafe_allow_html=True)
    output = ""

    # Legg til kjør-knapp
    if st.button("Kjør", type="primary"):
        if prompt:
            try:
                # Vis spinner mens vi venter på svar
                with st.spinner("Henter svar..."):
                    # Bruk nye API-kallet med 'model' i stedet for engine/deployment_id
                    response = client.chat.completions.create(
                        model       = modell,
                        messages    = [{"role": "user", "content": prompt}],
                        temperature = temperatur
                    )
                    output = response.choices[0].message.content.strip()
                    # Legg til i historikk
                    st.session_state.historikk.append({
                        "prompt": prompt,
                        "svar": output,
                        "modell": modell,
                        "temperatur": temperatur
                    })
            except Exception as e:
                output = f"Det oppstod en feil: {e}"
        else:
            output = "Vennligst skriv inn en prompt først"

    st.text_area("", value=output, height=120, disabled=True)

with col2:
    with st.expander("📚 Historikk", expanded=True):
        if st.session_state.historikk:
            for i, item in enumerate(reversed(st.session_state.historikk)):
                st.markdown(f"**Prompt {len(st.session_state.historikk)-i}:**")
                st.markdown(f"*{item['prompt']}*")
                st.markdown(f"**Svar:**")
                st.markdown(item['svar'])
                st.markdown(f"*Modell: {item['modell']}, Temp: {item['temperatur']}*")
                st.markdown("---")
        else:
            st.markdown("*Ingen historikk ennå*") 