"""DriftGuard AML dashboard entry point: `streamlit run app/app.py`. Reads only artifacts/ (no raw data)."""

import streamlit as st

st.set_page_config(page_title="DriftGuard AML", page_icon="🛡️", layout="wide")
nav = st.navigation([
    st.Page("home.py", title="Home", icon="🛡️", default=True),
    st.Page("pages/1_Timeline.py", title="Timeline", icon="📈", url_path="timeline"),
    st.Page("pages/2_Alert_Queue.py", title="Alert Queue", icon="🚨", url_path="alerts"),
    st.Page("pages/3_Label_Budget.py", title="Label Budget", icon="🏷️", url_path="label-budget"),
    st.Page("pages/4_Methodology.py", title="Methodology & Results", icon="📚", url_path="methodology"),
])
nav.run()
