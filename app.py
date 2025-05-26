import os
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
import pandas as pd
from data_cleaning import clean_data
from embeddings_utils import create_embeddings_from_df, query_data
import uuid
import streamlit.components.v1 as components

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

    # Legg til dropdown med forhåndsdefinerte prompter
    st.markdown("### Generell AI-assistent")
    
    # Forhåndsdefinerte prompter
    predefined_prompts = {
        "Velg en forhåndsdefinert prompt...": "",
        "Hva er hovedstaden i Sverige?": "Hva er hovedstaden i Sverige?",
        "Gi meg en liste over hvordan vurdere markedsrisiko?": "Gi meg en liste over hvordan vurdere markedsrisiko?",
        "Hvor mange fylker har vi i Norge?": "Hvor mange fylker har vi i Norge?",
        "Hvordan skrive en selskapsrapport?": "Hvordan skrive en selskapsrapport?"
    }
    
    selected_prompt = st.selectbox(
        "Velg en forhåndsdefinert prompt:",
        options=list(predefined_prompts.keys()),
        index=0
    )

    prompt = st.text_input("Skriv inn din prompt her:", 
                          value=predefined_prompts[selected_prompt] if selected_prompt != "Velg en forhåndsdefinert prompt..." else "",
                          placeholder="F.eks, Hva er hovedstaden i Norge")

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

    # Beregn høyde basert på innhold
    if output:
        # Estimer antall linjer basert på lengde og linjeskift
        lines = output.count('\n') + len(output) // 80 + 1
        height = max(200, min(lines * 25, 800))  # Økt minimum til 200px, maksimum til 800px
    else:
        height = 200  # Økt standard høyde

    # Text area med copy-funksjonalitet
    col_output, col_copy = st.columns([4, 1])
    
    with col_output:
        st.text_area("", value=output, height=height, disabled=True, key="output_area")
    
    with col_copy:
        if output and output != "Vennligst skriv inn en prompt først":
            # Legg til litt spacing for å justere knappen
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Generer unik ID for denne instansen
            unique_id = str(uuid.uuid4()).replace('-', '')
            
            # HTML/JavaScript copy-knapp
            components.html(f"""
                <div id="copy-container-{unique_id}">
                    <button onclick="copyToClipboard_{unique_id}()" style="
                        background-color: #ff4b4b;
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 14px;
                        margin-top: 5px;">
                        Copy
                    </button>
                    <div id="feedback-{unique_id}" style="
                        color: green;
                        font-size: 12px;
                        margin-top: 5px;
                        display: none;">
                        Copied!
                    </div>
                </div>

                <script>
                function copyToClipboard_{unique_id}() {{
                    const textToCopy = `{output.replace('`', '\\`').replace('\\', '\\\\').replace('"', '\\"')}`;
                    
                    if (navigator.clipboard && window.isSecureContext) {{
                        // Moderne clipboard API
                        navigator.clipboard.writeText(textToCopy).then(function() {{
                            showFeedback_{unique_id}();
                        }}).catch(function(err) {{
                            console.error('Clipboard API failed: ', err);
                            fallbackCopy_{unique_id}(textToCopy);
                        }});
                    }} else {{
                        // Fallback for eldre nettlesere
                        fallbackCopy_{unique_id}(textToCopy);
                    }}
                }}

                function fallbackCopy_{unique_id}(text) {{
                    const textArea = document.createElement('textarea');
                    textArea.value = text;
                    textArea.style.position = 'fixed';
                    textArea.style.left = '-999999px';
                    textArea.style.top = '-999999px';
                    document.body.appendChild(textArea);
                    textArea.focus();
                    textArea.select();
                    
                    try {{
                        document.execCommand('copy');
                        showFeedback_{unique_id}();
                    }} catch (err) {{
                        console.error('Fallback copy failed: ', err);
                    }}
                    
                    document.body.removeChild(textArea);
                }}

                function showFeedback_{unique_id}() {{
                    const feedback = document.getElementById('feedback-{unique_id}');
                    feedback.style.display = 'block';
                    setTimeout(() => {{
                        feedback.style.display = 'none';
                    }}, 2000);
                }}
                </script>
            """, height=100)

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