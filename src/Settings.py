import streamlit as st

from components.config import AppConfig, InternalAppConfig

app_config = AppConfig.load()
internal_app_config = InternalAppConfig.load()

st.set_page_config(
    page_title="yaepimet",
    page_icon="♫",
    layout="wide",
)
st.title("yaepimet")
placeholder = st.empty()

with placeholder.container():
    dict_ = {}
    for p in app_config.PARAMETERS:
        value_p = st.selectbox(p.display_name, options=p.list_of_choices, index=p.default_index)
        dict_[p.name] = value_p

    app_config = AppConfig(**{**dict_, "PATH": AppConfig.PATH, "PARAMETERS": AppConfig.PARAMETERS})

    st.button("Save", on_click=app_config.save)

    app_config = AppConfig.load()

    st.write("Current configuration:")

    metric_columns = st.columns(len(app_config.PARAMETERS))

    dict_ = app_config.to_dict()
    for j_p, p in enumerate(app_config.PARAMETERS):
        v_ = dict_[p.name]
        metric_columns[j_p].metric(p.display_name, v_)
