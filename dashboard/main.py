from IPython import display  # needed to call init.js on shap
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import requests
import json
from typing import Union, Optional

# Type checking
from api_models import ClientExplainResponse, ClientPredictResponse, ErrorResponse
from gauge import plot_gauge
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
# -------------------------------------------
# SECRETS
# stored locally in .streamlit/secrets.toml
# pasted into configuration in streamlit console when deploying
# -------------------------------------------
# Everything is accessible via the st.secrets dict (subsection [config]) :

API_URL = st.secrets["config"]["API_URL"]
default_threshold = st.secrets["config"]["THRESHOLD"]

# -------------------------------------------
#  PAGE LAYOUT
# Example page icons 💶💰💸💳🪙🤑💲
st.set_page_config(
    page_title="Scoring Dashboard",
    page_icon="💸",
    initial_sidebar_state="expanded",
    layout="wide",
)


# -----------------------------------------------------
# Header
def show_header():
    html_header = """
        <head>
            <meta charset="utf-8">
            <meta name="author" content="Ben Abdallah Helmi">
            <meta name="viewport" content="width=device-width, initial-scale=1">
        </head>             
    """
    st.write(html_header, unsafe_allow_html=True)
    # --------------------------------------------------------------------
    # Logo + Title
    t1, t2 = st.columns((1, 5))
    t1.image("dashboard/logo_projet_fintech.png", width=120)
    c2 = t2.container()
    c2.title("Scoring Dashboard")
    list_clients = get_list_clients()
    # c2.subheader(f"Client ID : {list(list_clients)[0]}")
    show_select_client(c2)
    mc = st.container()
    show_threshold_slider(mc)
    show_metrics(mc)


# ----------------------------------------------------
# Add custom styles
st.markdown(
    """<style>
    .streamlit-expanderHeader {font-size: x-large;background:#DDDDDD;margin-bottom:10px;}
    </style>""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# load JS visualization code to notebook. Without this, the SHAP plots won't be displayed
shap.initjs()


# ---------------------------------------
def get_query_params():
    query_params = st.query_params
    id = query_params.get("id")
    threshold = query_params.get("threshold")
    if id is not None:
        st.session_state.client_id = int(id)
    if threshold is not None:
        thresh = float(threshold)
        st.session_state.threshold = thresh
        st.session_state.threshold100 = thresh * 100


def set_query_params():
    st.query_params.clear()
    params = dict(id=st.session_state.client_id, threshold=st.session_state.threshold)
    st.query_params["id"] = st.session_state.client_id
    st.query_params["threshold"] = st.session_state.threshold


# ----------------------------------------
def init_key(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def initialise_session_state():
    # init_key('client_id',None)
    init_key("proba", 0)
    # Best threshold for both lgbm and logistic model is 0.6
    init_key("threshold", float(default_threshold))
    init_key("threshold100", st.session_state.threshold * 100)
    # get_query_params()


# ----------------------------------------
# Load list of clients
@st.cache_data
def get_list_clients():
    """API - load list of clients"""
    try:
        response = requests.get(f"{API_URL}/clients/ids/")
        if response.status_code == 200:
            data = response.json()
            client_ids = data["client_ids"]
            return client_ids

        else:
            print(f"Error: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


@st.cache_data
def get_all_clients():
    """API - load list of All clients"""
    try:
        response = requests.get(f"{API_URL}/clients/")
        if response.status_code == 200:
            json_string = response.json()
            json_data = json.loads(json_string)
            df = pd.DataFrame(json_data)
            return df

        else:
            st.error(f"Error: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return None


def show_select_client(col):
    """Selectionne un client parmi la liste des clients"""
    with st.spinner("recuperation de la liste de clients"):
        list_clients = get_list_clients()
        nb_clients = len(list_clients)
        logging.info(f"Number of clients: {nb_clients}")
        # client_id= st.sidebar.selectbox(f'Choisir un client (count={nb_clients}) :',list_clients)
        init_key("client_id", list_clients[0])
        col.selectbox(
            "Selectionne un client",
            list_clients,
            key="client_id",
            on_change=on_change_client,
            help="SK_ID_CURR",
        )


def on_change_client():
    data = get_all_clients()
    data["SK_ID_CURR"] = data["SK_ID_CURR"].astype(str)
    update_client_data(data)


# ----------------------------------------------
# Predict client data
@st.cache_data
def get_client_data(id):
    response = requests.get(f"{API_URL}/client/{id}")
    data = response.json()
    if data.get("error"):
        return data
    else:
        return pd.DataFrame.from_dict(data, orient="index")


import requests
from typing import Union


@st.cache_data  # Cache data for 600 seconds (10 minutes)
def get_client_predict(
    id, threshold, return_data
) -> Union[ClientPredictResponse, None]:
    """Predict whether to give a loan or not."""
 
    try:
        response = requests.get(f"{API_URL}/predict/{id}", params={"return_data": return_data, "threshold": threshold})
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            st.error(f"Error: {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return None


def update_client_dataframe(col):
    """Mettre à jour données du client et prediction de risque"""
    data = get_all_clients()

    data["SK_ID_CURR"] = data["SK_ID_CURR"].astype(str)
    client_id = st.session_state.client_id
    client_id_str = str(client_id)
    selected_index = data[data["SK_ID_CURR"] == client_id_str].index[0]
    # Définir l'index de départ pour afficher 10 éléments

    start_index = max(selected_index - 4, 0)
    end_index = min(selected_index + 5, len(data))
    # Filtrer le DataFrame pour afficher uniquement les 10 éléments autour de l'ID sélectionné
    filtered_data = data.iloc[start_index:end_index]
    styled_data = filtered_data.style.apply(
        lambda row: highlight_rows(row, client_id_str), axis=1
    )
    # Display the styled DataFrame in Streamlit
    col.dataframe(styled_data)


def update_client_data(data):
    """Mettre à jour données du client et prediction de risque"""
    client_id = st.session_state.client_id
    threshold = st.session_state.threshold
    pred_data = get_client_predict(client_id, threshold, return_data=True)
    st.session_state.proba = pred_data.get("y_pred_proba")
    st.session_state.client_data = series_from_dictkey(pred_data, "client_data")
    # filtered_data_display.dataframe(styled_data)
    set_query_params()


# Define a function to apply background color to rows based on condition
def highlight_rows(row, client_id):
    # Define the condition for highlighting
    condition = (
        row["SK_ID_CURR"] == client_id
    )  # For example, highlight rows where Age > 30

    # Define the CSS styling for highlighted rows
    if condition:
        return ["background-color: yellow"] * len(row)
    else:
        return [""] * len(row)


def on_change_threshold():
    # print(f'on_change_threshold')
    st.session_state.threshold = st.session_state.threshold100 / 100
    set_query_params()


def show_threshold_slider(col):
    col.slider(
        "Threshold",
        0.0,
        100.0,
        1.0,
        format="%g %%",
        key="threshold100",
        on_change=on_change_threshold,
    )


def show_metrics(mc):
    """Predict, et retourner les données client"""
    accept = st.session_state.proba < st.session_state.threshold
    accept_style = "{background: rgba(0,255, 255, 0.1);color: green;}"
    refuse_style = "{background: rgba(255,0, 20, 0.1);color: red;}"
    metric_style = f'<style>div[data-testid="metric-container"] {accept_style if accept else refuse_style}</style>'
    mc.markdown(metric_style, unsafe_allow_html=True)
    m2, m3, m4 = mc.columns(3)
    m2.metric(
        label=f"Décision",
        value="accepté" if accept else "refusé",
        delta=f"threshold = {st.session_state.threshold100:.1f}",
        delta_color="inverse",
    )
    m3.metric(
        label="Niveau de Risque :",
        value=f"{st.session_state.proba*100:.1f} %",
        delta="probabilité de defaut",
        delta_color="inverse",
    )
    fig, ax = plt.subplots()
    plot_gauge(
        arrow=st.session_state.proba,
        threshold=st.session_state.threshold100 / 100,
        n_colors=50,
        title="risque",
        ax=ax,
    )
    m4.write(fig)

    # m5.write(list_clients[0])



# Define ClientExplainResponse and ErrorResponse types if not defined already
@st.cache_data
def get_client_explain(
    id: int, threshold: Optional[float] = None, return_data: bool = False
) -> Union[ClientExplainResponse, ErrorResponse]:
    """Explain whether to give a loan or not."""

    try:
        
        response = requests.get(f"{API_URL}/explain/{id}", params={"return_data": return_data, "threshold": threshold})
        data = response.json()
        if response.status_code == 200:
            # Check if the response contains an error message
            if "error" in data:
                return ErrorResponse(message=data["error"])
            # Assuming ClientExplainResponse structure, replace it with your actual structure
            return data
        else:
            # Handle non-200 status codes
            return ErrorResponse(
                message=f"API returned status code {response.status_code}"
            )
    except requests.RequestException as e:
        # Handle request exceptions
        return ErrorResponse(message=str(e))


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def st_shap2(plot, height=None):
    fig = plt.gcf() if plot is None else plot
    st.write(fig)


def show_local_explain(col):
    # Explain
    explain_data = get_client_explain(
        st.session_state.client_id, st.session_state.threshold, return_data=True
    )
    if not explain_data.get("error"):
        shap_values = series_from_dictkey(explain_data, "shap_values").to_numpy()
        expected_value = explain_data.get("expected_value")
        client_data = series_from_dictkey(explain_data, "client_data")
        feature_names = client_data.index.tolist()
        client_values = client_data.to_numpy()
        st.session_state.client_data = client_data
        exp = shap.Explanation(
            shap_values,
            base_values=expected_value,
            feature_names=feature_names,
            data=client_values,
        )
        with col:
            d1 = '<span style="color:#FFF;background-color:#FF0051; padding:5px;margin: 10px;">augmentation de risque</span>'
            d2 = '<span style="color:#FFF;background-color:#008BFB; padding:5px;margin: 10px;">reduction de risque</span>'
            v1, v2 = st.columns(2)
            v1.write(d1, unsafe_allow_html=True)
            v2.write(d2, unsafe_allow_html=True)

            # Interprétabilité locale
            st.write("Interprétabilité locale (Waterfall plot)")
            plt.close()
            st_shap2(shap.plots.waterfall(exp, max_display=10))
            st.write("Force plot")
            plt.close()
            st_shap(shap.plots.force(exp))


# -------------------------------------------------------------------------
# Explain summary (global interpretation)
@st.cache_data
def get_explain_all(nb=100) -> Union[ClientExplainResponse, ErrorResponse]:
    """explain give loan or not"""
    params = dict(nb=nb)
    response = requests.get(f"{API_URL}/explain/all", params=params)
    data = response.json()
    if data.get("error"):
        st.write(data)
    return data





def show_global_explain(col):
    # Explain (maximum de 1000 clients, car plus de 1000 c'est des très gros réponses)
    exp_data = get_explain_all(nb=300)
    if not exp_data.get("error"):
        # st.write(exp_data.keys())
        x_data: dict = exp_data.get("client_data")
        #df_data = json_to_df(x_data)
        df_data = pd.DataFrame(x_data)
        feature_names = df_data.columns
        shap_values = np.array(exp_data.get("shap_values"))
        expected_value = exp_data.get("expected_value")
        client_values = df_data.to_numpy()
        exp = shap.Explanation(
            shap_values,
            base_values=expected_value,
            feature_names=feature_names,
            data=client_values,
        )
        with col:
            d1 = '<span style="color:#FFF;background-color:#FF0051; padding:5px;margin: 10px;">augmentation de risque</span>'
            d2 = '<span style="color:#FFF;background-color:#008BFB; padding:5px;margin: 10px;">reduction de risque</span>'
            v1, v2 = st.columns(2)
            v1.write(d1, unsafe_allow_html=True)
            v2.write(d2, unsafe_allow_html=True)

            # Interprétabilité globale
            # st.write('Summary plot')
            # plt.close()
            # st_shap2(shap.plots.summary(exp, max_display=20))
            st.write("Interprétabilité globale")
            plt.close()
            st_shap2(shap.plots.beeswarm(exp, max_display=20))


def main():
    """Display data"""
    initialise_session_state()
    show_header()

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Liste Clients",
            "Variables Influentes Globalement",
            "Variables les plus influentes pour ce client",
            " position Client dans chaque variable",
        ]
    )
    filtered_data_display = tab1
    with tab1:
        st.header("Liste Clients  : ")
        data = get_all_clients()

        data["SK_ID_CURR"] = data["SK_ID_CURR"].astype(str)
        # Trouver l'index de la ligne sélectionnée
        update_client_data(data)
        expander_global1 = st.expander("Variables clients ", expanded=True)
        update_client_dataframe(expander_global1)
    with tab2:
        st.header("Features Importance")
        expander_global = st.expander("Variables influentes globalement", expanded=True)
        show_global_explain(expander_global)

    with tab3:
        st.header("Individual Features Importance")
        expander_local = st.expander(
            "Variables les plus influentes pour ce client", expanded=True
        )
        show_local_explain(expander_local)
    with tab4:
        st.header("Client position dans chaque variable")
        expander_client = st.expander(
            "Client position dans chaque variable", expanded=True
        )
        # show_variable_explain(expander_client)
        with expander_client:
            df_client = st.session_state.client_data.to_frame()
            if isinstance(df_client, pd.DataFrame):
                st.dataframe(df_client)


# ------------------------------------------------
# Utility functions
# ------------------------------------------------
def series_from_dictkey(data, key: str) -> pd.Series:
    """
    Convert a dictionary held within a dictionary key to a series.

    Most response json are dictionaries within dictionaries (hierarchical json).
    Use this method to extract a series
    returns empty series otherwise
    """
    dict_key = data.get(key, {})
    nb_keys = len(dict_key.keys())
    df = pd.DataFrame.from_dict(data.get(key, {}), orient="index")
    # st.write(f'series_from_dictkey(data,key={key}, nb = {nb_keys}, df.shape={df.shape}')
    return df.iloc[:, 0]


# ------------------------------------------------
# Requests to API server
# ------------------------------------------------


# ------------------------------------------------
# Plotting routines
# ------------------------------------------------


main()
