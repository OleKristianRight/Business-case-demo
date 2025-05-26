import os
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
import pandas as pd
from data_cleaning import clean_data
from embeddings_utils import create_embeddings_from_df, query_data

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
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

st.set_page_config(page_title="Prompt til Output – Coast Seafood", layout="centered")

# Opprett to kolonner
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
        <h1 style='text-align: center; font-size: 3em;'>🧠 Prompt til Output –<br>Coast Seafood</h1>
    """, unsafe_allow_html=True)

    # Legg til Excel og CSV-opplastingsfelt
    uploaded_file = st.file_uploader("Last opp Excel eller CSV-fil", type=['xlsx', 'xls', 'csv'])
    
    # Flytt modellvalg høyere opp så det er tilgjengelig for alle funksjoner
    st.markdown("<b>Innstillinger</b>", unsafe_allow_html=True)
    modell = st.selectbox(
        "Velg modell",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0,
        help="Velg hvilken modell som skal brukes"
    )
    
    if uploaded_file is not None:
        try:
            # Sjekk filtype og les filen tilsvarende
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:  # Excel-filer (.xlsx, .xls)
                df = pd.read_excel(uploaded_file)
                
            st.success(f"Filen '{uploaded_file.name}' er lastet opp!")
            st.dataframe(df)
            
            # Legg til data cleaning-knapp
            if st.button("Rens data"):
                with st.spinner("Renser data..."):
                    df_cleaned = clean_data(df)
                    st.success("Data er nå renset! (Duplikater og NaN-verdier er fjernet)")
                    st.dataframe(df_cleaned)
                    
                    # Lagre renset data i session state
                    st.session_state.df_cleaned = df_cleaned
                    
                    # Legg til nedlastingsmulighet for renset data
                    csv = df_cleaned.to_csv(index=False)
                    st.download_button(
                        label="Last ned renset data som CSV",
                        data=csv,
                        file_name="renset_data.csv",
                        mime="text/csv"
                    )
            
            # Legg til knapp for å aktivere LLM-funksjonalitet
            if 'df_cleaned' in st.session_state:
                if st.button("Bruk opplastet data for LLM-modellen"):
                    with st.spinner("Oppretter embeddings og indeks..."):
                        st.session_state.vectorstore = create_embeddings_from_df(st.session_state.df_cleaned)
                        st.session_state.llm_ready = True
                        st.success("Data er nå klar for LLM-spørsmål!")
        except Exception as e:
            st.error(f"Det oppstod en feil ved lesing av filen: {e}")

    # Legg til spørsmål om data - kun hvis LLM er aktivert
    if st.session_state.get('llm_ready', False):
        st.markdown("### Spør om dataen")
        data_query = st.text_input("Still et spørsmål om dataen:", placeholder="F.eks, Hvilken matvare har lavest prisPerEnhet for hver butikk?")
        if data_query:
            with st.spinner("Analyserer data og genererer svar..."):
                # Hent relevante dokumenter
                relevant_docs = query_data(st.session_state.vectorstore, data_query)
                
                # Lag kontekst fra relevante dokumenter
                context = "\n".join([doc.page_content for doc in relevant_docs])
                
                # Lag et sammendrag av dataen for bedre kontekst
                df = st.session_state.df_cleaned
                data_summary = f"""
DATASETT OVERSIKT:
- Totalt antall rader: {len(df)}
- Kolonner: {', '.join(df.columns.tolist())}
- Unike butikker: {df['Butikk'].unique().tolist() if 'Butikk' in df.columns else 'Ikke funnet'}
- Prisområde: {df['PrisPerEnhet'].min():.2f} - {df['PrisPerEnhet'].max():.2f} if 'PrisPerEnhet' in df.columns else 'Ikke funnet'
"""
                
                # Lag prompt for LLM
                llm_prompt = f"""
Du er en ekspert dataanalytiker. Analyser datasettet grundig og svar på spørsmålet.

{data_summary}

Spørsmål: "{data_query}"

RELEVANTE DATA:
{context}

INSTRUKSJONER:
1. Hvis spørsmålet handler om "lavest/høyest per butikk", må du:
   - Identifisere alle unike butikker i dataen
   - For hver butikk, finn produktet med lavest/høyest verdi
   - Presenter resultatet for HVER butikk

2. Hvis spørsmålet handler om "lavest/høyest totalt", finn den absolutt laveste/høyeste verdien

3. Inkluder alltid:
   - Produktnavn
   - ProduktID
   - Butikknavn
   - Eksakt pris/verdi

4. Hvis du ikke ser data for alle butikker, si det tydelig

Svar basert på dataanalysen:
"""
                
                # Send til LLM
                try:
                    response = client.chat.completions.create(
                        model=modell,
                        messages=[{"role": "user", "content": llm_prompt}],
                        temperature=0.0  # Helt deterministisk for presise svar
                    )
                    llm_answer = response.choices[0].message.content.strip()
                    
                    st.markdown("### Svar:")
                    st.markdown(llm_answer)
                    
                    # Vis også antall dokumenter som ble analysert
                    st.caption(f"Analyserte {len(relevant_docs)} relevante datapunkter")
                    
                except Exception as e:
                    st.error(f"Det oppstod en feil ved generering av svar: {e}")

    prompt = st.text_input("Skriv inn din prompt her:", placeholder="F.eks, Hva er hovedstaden i Norge")

    temperatur = st.slider("Temperatur", 0.0, 1.0, 0.0, 0.01)
    st.caption("Styrer hvor kreative svarene skal være")

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