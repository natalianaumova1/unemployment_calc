import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb


st.set_page_config(
    page_title="Unemployment Risk Calculator",
    page_icon="📊",
    layout="centered"
)

st.title("Unemployment Risk Calculator")

st.markdown(
    "Estimate your risk of unemployment based on socio-demographic factors. "
    "Answer a few questions about your background to get a personalized statistical assessment."
)


with st.expander("ℹ️ About this tool", expanded=False):
    st.markdown(
        """
        ### What is this tool?
        This calculator uses a machine learning model to estimate an individual's risk of involuntary unemployment. 
        By answering a few questions about your background, the tool provides a statistical probability of being unemployed 
        based on your socio-demographic profile.

        The model is trained on data from the **European Social Survey (ESS)**, 
        a comprehensive cross-national survey of diverse populations across Europe.

        ---

        ### Highlighting Structural Inequality
        Unemployment is often viewed through the lens of individual effort and personal choice. 
        However, this perspective overlooks the powerful influence of structural factors—the social and economic systems 
        that create advantages for some and barriers for others.

        This calculator aims to demonstrate how characteristics largely outside of your control can 
        significantly impact your employment prospects. These include:

        - **Demographics:** Age, gender, country of residence  
        - **Migration Background:** Whether you or your parents were born in your country of residence  
        - **Socio-economic Background:** Parents' education level  
        - **Personal Profile:** Education, health status, marital status  
        - **Geographic Location:** Specific region where you live  

        By analyzing these variables, this tool shows that the risk of unemployment is **not equally distributed** 
        — it reflects **structural inequality** in the labor market.

        ---

        ### How to Use It & Disclaimer
        Fill out the form below with your information to get a personalized risk assessment.  

        ⚠️ **Disclaimer**: The result is a statistical probability, not a definitive prediction of your future.  
        This tool is for **educational purposes only** — to encourage reflection on the complex factors 
        shaping economic outcomes in society.
        """
    )

# --- Load models
best_xgb = joblib.load("best_xgboost.pkl")

st.markdown("Complete the questionnaire to see your estimated unemployment risk. All fields are required.")

# --- Mapping functions
def map_yes_no(value):
    return {"Yes": "1", "No": "2", "Other/No answer": "9"}.get(value, "9")

def map_gender(value):
    return {"Male": "1", "Female": "2", "Other/No answer": "9"}.get(value, "9")

map_eisced_dict = {
    "Less than lower secondary": "1",
    "Lower secondary": "2",
    "Upper secondary (lower tier)": "3",
    "Upper secondary (upper tier)": "4",
    "Advanced vocational/sub-degree": "5",
    "Bachelor": "6",
    "Master": "7",
    "Other/No answer": "55"
}

nacer2_options = {
    "Agriculture, forestry, fishing": "1",
    "Mining and quarrying": "2",
    "Manufacturing": "3",
    "Energy, water supply, waste": "4",
    "Construction": "5",
    "Trade and transport": "6",
    "Accommodation and food service": "7",
    "Information and communication": "8",
    "Financial and insurance": "9",
    "Real estate": "10",
    "Professional, scientific, technical services": "11",
    "Administrative and support services": "12",
    "Public administration, education, health": "13",
    "Arts, entertainment, recreation": "14",
    "Other services": "15",
}

tporgwk_options = {
    "Central or local government": "1",
    "Other public sector (education, health)": "2",
    "State-owned enterprise": "3",
    "Private firm": "4",
    "Self-employed": "5",
    "Other": "6",
}

hlthhmp_options = {
    "Yes, a lot": "1",
    "Yes, to some extent": "2",
    "No": "3",
    "Other/No answer": "9"
}

domicil_options = {
    "Big city": "1",
    "Suburb/outskirts": "2",
    "Town/small city": "3",
    "Village/countryside": "4",
    "Farm/home in countryside": "5",
    "Other/No answer": "9"
}

dscrgrp_options = {
    "Yes": "1",
    "No": "2",
    "Other/No answer": "9"
}

maritalb_options = {
    "Married": "1",
    "Registered civil union": "2",
    "Separated": "3",
    "Divorced": "4",
    "Widowed": "5",
    "Never married": "6",
    "Other/No answer": "9"
}

mbtru_options = {
    "Yes, currently": "1",
    "Yes, previously": "2",
    "No": "3",
    "Other/No answer": "9"
}

# --- Countries and regions
country_regions = {
    "Austria": ["-- Select --","Burgenland","Niederösterreich","Wien","Kärnten","Steiermark","Oberösterreich","Salzburg","Tirol","Vorarlberg"],
    "Belgium": ["-- Select --","Région de Bruxelles-Capitale/Brussels Hoofdstedelijk Gewest","Vlaams Gewest","Région wallonne"],
    "Bulgaria": ["-- Select --","Северна и Югоизточна България","Югозападна и Южна централна България"],
    "Croatia": ["-- Select --","Panonska Hrvatska","Jadranska Hrvatska","Grad Zagreb","Sjeverna Hrvatska"],
    "Cyprus": ["-- Select --","Κύπρος"],
    "Finland": ["-- Select --","Länsi-Suomi","Helsinki-Uusimaa","Etelä-Suomi","Pohjois- ja Itä-Suomi","Åland"],
    "France": ["-- Select --","Ile-de-France","Centre — Val de Loire","Bourgogne","Franche-Comté","Basse-Normandie","Haute-Normandie","Nord-Pas de Calais","Picardie","Alsace","Champagne-Ardenne","Lorraine","Pays de la Loire","Bretagne","Aquitaine","Limousin","Poitou-Charentes","Languedoc-Roussillon","Midi-Pyrénées","Auvergne","Rhône-Alpes","Provence-Alpes-Côte d’Azur","Corse","Guadeloupe","Martinique","Guyane","La Réunion","Mayotte"],
    "Georgia": ["-- Select --","Capital","Western Georgia","Eastern Georgia"],
    "Germany": ["-- Select --","Baden-Württemberg","Bayern","Berlin","Brandenburg","Bremen","Hamburg","Hessen","Mecklenburg-Vorpommern","Niedersachsen","Nordrhein-Westfalen","Rheinland-Pfalz","Saarland","Sachsen","Sachsen-Anhalt","Schleswig-Holstein","Thüringen"],
    "Greece": ["-- Select --","Αττική","Νησιά Αιγαίου, Κρήτη","Βόρεια Ελλάδα","Κεντρική Ελλάδα"],
    "Hungary": ["-- Select --","Közép-Magyarország","Dunántúl","Alföld és Észak"],
    "Ireland": ["-- Select --","Northern and Western","Southern","Eastern and Midland"],
    "Iceland": ["-- Select --","Ísland"],
    "Italy": ["-- Select --","Nord-Ovest","Sud","Isole","Nord-Est","Centro"],
    "Lithuania": ["-- Select --","Sostinės regionas","Vidurio ir vakarų Lietuvos regionas"],
    "Latvia": ["-- Select --","Latvija"],
    "Montenegro": ["-- Select --","Црна Гора"],
    "Netherlands": ["-- Select --","Noord-Nederland","Oost-Nederland","West-Nederland","Zuid-Nederland"],
    "Norway": ["-- Select --","Norge","Innlandet","Trøndelag","Nord-Norge","Oslo og Viken","Agder og Sør-Østlandet"],
    "Poland": ["-- Select --","Makroregion południowy","Makroregion północno-zachodni","Makroregion południowo-zachodni","Makroregion północny","Makroregion centralny","Makroregion wschodni","Makroregion województwo mazowieckie"],
    "Portugal": ["-- Select --","Continente","Região Autónoma dos Açores","Região Autónoma da Madeira"],
    "Serbia": ["-- Select --","Србија - север","Србија - југ"],
    "Slovenia": ["-- Select --","Slovenija","Vzhodna Slovenija","Zahodna Slovenija"],
    "Slovakia": ["-- Select --","Slovensko","Bratislavský kraj","Západné Slovensko","Stredné Slovensko","Východné Slovensko"],
    "Spain": ["-- Select --","Galicia","Principado de Asturias","Cantabria","País Vasco","Comunidad Foral de Navarra","La Rioja","Aragón","Comunidad de Madrid","Castilla y León","Castilla-La Mancha","Extremadura","Cataluña","Comunitat Valenciana","Illes Balears","Andalucía","Región de Murcia","Ciudad de Ceuta","Ciudad de Melilla","Canarias"],
    "Sweden": ["-- Select --","Östra Sverige","Södra Sverige","Norra Sverige"],
    "Switzerland": ["-- Select --","Région lémanique","Espace Mittelland","Nordwestschweiz","Zürich","Ostschweiz","Zentralschweiz","Ticino"],
    "United Kingdom": ["-- Select --","North East (England)","North West (England)","Yorkshire and the Humber","East Midlands (England)","West Midlands (England)","East of England","London","South East (England)","South West (England)","Wales","Scotland","Northern Ireland"]
}

# --- Initialize session_state for regions
if "region_options" not in st.session_state:
    st.session_state.region_options = ["-- Select --"]

# --- Country select (outside form to update regions)
selected_country = st.selectbox(
    "Country",
    ["-- Select --"] + list(country_regions.keys())
)

# Update regions based on selected country
if selected_country != "-- Select --":
    st.session_state.region_options = country_regions[selected_country]
else:
    st.session_state.region_options = ["-- Select --"]

# --- User form
with st.form("user_input_form"):
    gndr = st.selectbox("Gender", ["-- Select --","Male","Female","Other/No answer"])
    agea = st.number_input("Age", min_value=15, max_value=100, value=15)
    maritalb = st.selectbox("Marital status", ["-- Select --"] + list(maritalb_options.keys()))
    hhmmb = st.number_input("Number of people in household", min_value=1, max_value=20, value=1)

    region = st.selectbox("Region", st.session_state.region_options)

    brncntr = st.selectbox("Were you born in this country?", ["-- Select --","Yes", "No", "Other/No answer"])
    facntr = st.selectbox("Was your father born in this country?", ["-- Select --","Yes","No","Other/No answer"])
    mocntr = st.selectbox("Was your mother born in this country?", ["-- Select --","Yes","No","Other/No answer"])

    domicil = st.selectbox("Type of area you live in", ["-- Select --"] + list(domicil_options.keys()))

    eisced = st.selectbox("Your highest level of education", ["-- Select --"] + list(map_eisced_dict.keys()))
    eiscedf = st.selectbox("Father's highest level of education", ["-- Select --"] + list(map_eisced_dict.keys()))
    eiscedm = st.selectbox("Mother's highest level of education", ["-- Select --"] + list(map_eisced_dict.keys()))

    isco08 = st.selectbox("Main occupation", ["-- Select --","Armed forces","Managers","Professionals","Technicians","Clerical support","Service and sales","Skilled agricultural","Craft and trades","Plant and machine operators","Elementary occupations"])
    nacer2 = st.selectbox("Industry of your main job", ["-- Select --"] + list(nacer2_options.keys()))
    tporgwk = st.selectbox("Type of organization you work/worked for", ["-- Select --"] + list(tporgwk_options.keys()))
    mbtru = st.selectbox("Are/were you a member of a trade union?", ["-- Select --"] + list(mbtru_options.keys()))

    hlthhmp = st.selectbox("Are you hampered in daily activities by illness/disability?", ["-- Select --"] + list(hlthhmp_options.keys()))
    dscrgrp = st.selectbox("Are you member of a discriminated group?", ["-- Select --"] + list(dscrgrp_options.keys()))

    submitted = st.form_submit_button("Predict risk")

# --- Prediction
if submitted:
    missing = []
    required_fields = {
        "Gender": gndr,
        "Region": region,
        "Your education": eisced,
        "Father's education": eiscedf,
        "Mother's education": eiscedm,
        "Industry": nacer2,
        "Type of organization": tporgwk,
        "Trade union membership": mbtru,
        "Health limitation": hlthhmp,
        "Discrimination group": dscrgrp
    }
    for k,v in required_fields.items():
        if v == "-- Select --":
            missing.append(k)

    if missing:
        st.error(f"Please fill in all required fields: {', '.join(missing)}")
        st.stop() 
    else:
        user_inputs = {
            "gndr": map_gender(gndr),
            "agea": agea,
            "maritalb": maritalb_options[maritalb],
            "hhmmb": hhmmb,
            "cntry": selected_country,
            "region": region,
            "brncntr": map_yes_no(brncntr),
            "facntr": map_yes_no(facntr),
            "mocntr": map_yes_no(mocntr),
            "domicil": domicil_options[domicil],
            "eisced": map_eisced_dict[eisced],
            "eiscedf": map_eisced_dict[eiscedf],
            "eiscedm": map_eisced_dict[eiscedm],
            "isco08": isco08,
            "nacer2": nacer2_options[nacer2],
            "tporgwk": tporgwk_options[tporgwk],
            "mbtru": mbtru_options[mbtru],
            "hlthhmp": hlthhmp_options[hlthhmp],
            "dscrgrp": dscrgrp_options[dscrgrp],
        }


        df_user = pd.DataFrame([user_inputs])

        # --- Prediction
        xgb_clf = best_xgb.named_steps['xgb_clf']
        preproc = best_xgb.named_steps['preproc']

        # getting feature names
        X_user_array = preproc.transform(df_user)
        feat_names = preproc.get_feature_names_out()
        X_user = pd.DataFrame(X_user_array, columns=feat_names)

        proba = xgb_clf.predict_proba(X_user)[:, 1]
        st.success(f"Predicted unemployment risk: {proba[0]:.2%}")

        # --- Individual feature contributions
        dmatrix = xgb.DMatrix(X_user, feature_names=list(X_user.columns))

        contribs = xgb_clf.get_booster().predict(dmatrix, pred_contribs=True)[0]

        contribs = contribs[:-1]

        # --- Map back to original variables
        df = pd.DataFrame({"feature": X_user.columns, "contribution": contribs})

        def original_name(encoded_feature):
            parts = encoded_feature.split('__')
            if len(parts) >= 2:
                rest = parts[1]
                return rest.split('_')[0]
            return encoded_feature

        friendly_names = {
            "gndr": "Gender",
            "agea": "Age",
            "maritalb": "Marital status",
            "hhmmb": "Household size",
            "cntry": "Country",
            "region": "Region",
            "brncntr": "Born in country",
            "facntr": "Father born in country",
            "mocntr": "Mother born in country",
            "domicil": "Type of area",
            "eisced": "Your education",
            "eiscedf": "Father's education",
            "eiscedm": "Mother's education",
            "isco08": "Occupation",
            "nacer2": "Industry",
            "tporgwk": "Type of organization",
            "mbtru": "Membership in trade union",
            "hlthhmp": "Health limitation",
            "dscrgrp": "Discrimination group"
        }

        df['orig'] = df['feature'].apply(original_name)
        df['friendly'] = df['orig'].map(friendly_names).fillna(df['orig'])

        # --- Aggregate contributions by original feature
        agg = df.groupby('friendly')['contribution'].sum().reset_index()
        agg = agg.sort_values('contribution', ascending=True)

        st.markdown("""
        **How to read this chart:**  
        - Each bar represents how much a specific factor contributes to your predicted unemployment risk.  
        - Bars pointing to the right (positive values) **increase your risk**, while bars pointing to the left (negative values) **decrease your risk**.  
        - The longer the bar, the stronger the effect of that factor.  
        """)

        # --- Horizontal bar chart
        st.subheader("Factors Affecting Your Risk")
        fig, ax = plt.subplots(figsize=(8,6))
        ax.barh(agg['friendly'], agg['contribution'], color='skyblue')
        ax.set_xlabel("Contribution to Predicted Risk")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        st.pyplot(fig)