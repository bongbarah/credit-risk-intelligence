
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# Finance chart colour palette
NAVY = "#102A43"
GREEN = "#087F5B"
GOLD = "#D4A72C"
LIGHT_BLUE = "#829AB1"
CHART_COLORS = [NAVY, GREEN, GOLD, LIGHT_BLUE]

def style_chart(fig):
    fig.update_layout(
        template="plotly_white",
        font=dict(
            family="Arial, sans-serif",
            color=NAVY,
            size=13
        ),
        title_font=dict(size=18, color=NAVY),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor="#D9E2EC"
    )
    fig.update_yaxes(
        gridcolor="#E6EDF3",
        zerolinecolor="#D9E2EC"
    )
    return fig

# --------------------------------------------------
# 1. PROJECT SETTINGS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "credit_default.csv"
MODEL_PATH = PROJECT_DIR / "models" / "credit_risk_model.joblib"
IMPORTANCE_PATH = PROJECT_DIR / "feature_importance.csv"
COMPARISON_PATH = PROJECT_DIR / "model_comparison.csv"
THRESHOLD_PATH = PROJECT_DIR / "threshold_comparison.csv"

FEATURE_NAMES = [
    "credit_limit", "gender", "education", "marital_status", "age",
    "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6",
    "bill_amt1", "bill_amt2", "bill_amt3", "bill_amt4",
    "bill_amt5", "bill_amt6",
    "pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4",
    "pay_amt5", "pay_amt6"
]

TARGET = "default_next_month"

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="📊",
    layout="wide"
)

# --------------------------------------------------
# FINANCE DASHBOARD DESIGN
# --------------------------------------------------

st.markdown("""
<style>
    /* Main page */
    .stApp {
        background-color: #F1F5F9;
        color: #102A43;
    }

    [data-testid="stHeader"] {
        background-color: #F1F5F9;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* Main title and headings */
    h1 {
        color: #102A43 !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    h2, h3, h4 {
        color: #102A43 !important;
        font-weight: 700 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #102A43;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background-color: transparent;
        border-radius: 8px;
        padding: 5px;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #D9E2EC;
        border-left: 5px solid #087F5B;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 3px 10px rgba(16, 42, 67, 0.06);
    }

    [data-testid="stMetricLabel"] {
        color: #486581 !important;
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        color: #102A43 !important;
        font-weight: 800;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button,
    .stFormSubmitButton > button {
        background-color: #087F5B;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 0.55rem 1rem;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    .stFormSubmitButton > button:hover {
        background-color: #066649;
        color: #FFFFFF;
        border: 1px solid #D4A72C;
    }

    /* Tabs and selected controls */
    [data-baseweb="tab"][aria-selected="true"] {
        color: #087F5B !important;
        border-bottom-color: #087F5B !important;
    }

    /* Data tables */
    [data-testid="stDataFrame"] {
        background-color: #FFFFFF;
        border: 1px solid #D9E2EC;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Input fields */
    input, textarea {
        border-radius: 7px !important;
    }

    /* Divider */
    hr {
        border-color: #D9E2EC;
    }

    /* Informational text */
    [data-testid="stCaptionContainer"] {
        color: #627D98;
    }

    /* Links */
    a {
        color: #087F5B !important;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 2. LOAD EXISTING FILES
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = FEATURE_NAMES + [TARGET]
    df = df.drop_duplicates().copy()
    return df


@st.cache_data
def load_optional_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# --------------------------------------------------
# 3. CHECK REQUIRED FILES
# --------------------------------------------------


st.markdown("""
<div style="
    background: linear-gradient(110deg, #102A43, #087F5B);
    padding: 28px 30px;
    border-radius: 16px;
    margin-bottom: 22px;
">
    <div style="
        color: #D4A72C;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
    ">FINANCIAL ANALYTICS • CREDIT RISK</div>
    <h1 style="
        color: white !important;
        margin: 10px 0 8px 0;
        font-size: 32px;
    ">Credit Risk Intelligence</h1>
    <p style="
        color: #E6F4F1;
        margin: 0;
        font-size: 15px;
    ">Credit-default analysis, risk screening and model insights</p>
</div>
""", unsafe_allow_html=True)


if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Trained model not found: {MODEL_PATH}")
    st.info("Run model_training.py first.")
    st.stop()

try:
    df = load_data()
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the dataset or model: {error}")
    st.stop()

importance_df = load_optional_csv(IMPORTANCE_PATH)
comparison_df = load_optional_csv(COMPARISON_PATH)
threshold_df = load_optional_csv(THRESHOLD_PATH)




# --------------------------------------------------
# 4. SIDEBAR CONTROLS
# --------------------------------------------------



st.sidebar.markdown(
    "<div style='padding:8px 0 20px 0;'>"
    "<p style='color:#D4A72C;font-size:12px;"
    "font-weight:800;letter-spacing:2px;'>"
    "FINANCE ANALYTICS</p>"
    "<h2 style='color:#FFFFFF;font-size:23px;"
    "font-weight:800;line-height:1.3;margin:8px 0;'>"
    "Credit Risk Intelligence</h2>"
    "<p style='color:#BCCCDC;font-size:12px;margin-top:10px;'>"
    "Portfolio monitoring &amp; risk screening</p>"
    "</div>",
    unsafe_allow_html=True
)



st.sidebar.markdown(
    "<p style='color:#D4A72C;font-size:11px;"
    "font-weight:700;letter-spacing:1.5px;'>NAVIGATION</p>",
    unsafe_allow_html=True
)

st.sidebar.caption("Choose a dashboard section")


threshold = st.sidebar.slider(
    "Default-risk threshold",
    min_value=0.05,
    max_value=0.80,
    value=0.40,
    step=0.05,
    help=(
        "Customers at or above this predicted probability "
        "are flagged for further review."
    )
)

st.sidebar.caption(
    "Thresholds are for demonstration and should be validated "
    "before real lending use."
)


page = st.sidebar.radio(
    "Dashboard pages",
    [
        "Overview",
        "Customer Risk Analysis",
        "Model Performance",
        "Feature Importance",
        "Score a Customer"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style="font-size:12px;color:#BCCCDC;line-height:1.8;">
    <b style="color:#FFFFFF;">PROJECT DETAILS</b><br>
    Model: Random Forest<br>
    Dataset: Credit default<br>
    Purpose: Educational analysis
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="
    margin-top: 22px;
    padding-top: 12px;
    border-top: 1px solid #486581;
    color: #BCCCDC;
    font-size: 11px;
">
    CREDIT RISK INTELLIGENCE<br>
    Portfolio project • 2026
</div>
""", unsafe_allow_html=True)



# --------------------------------------------------
# 5. CREATE MODEL PREDICTIONS
# --------------------------------------------------

X = df[FEATURE_NAMES]
y = df[TARGET].astype(int)

try:
    probabilities = model.predict_proba(X)[:, 1]
except Exception as error:
    st.error(f"Unable to generate model predictions: {error}")
    st.stop()

analysis_df = df.copy()
analysis_df["predicted_default_probability"] = probabilities
analysis_df["predicted_default"] = (
    analysis_df["predicted_default_probability"] >= threshold
).astype(int)

analysis_df["risk_category"] = pd.cut(
    analysis_df["predicted_default_probability"],
    bins=[-0.001, 0.20, 0.40, 1.001],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)


# --------------------------------------------------
# 6. OVERVIEW
# --------------------------------------------------

if page == "Overview":
    st.subheader("Portfolio overview")

    total_customers = len(analysis_df)
    actual_defaults = int(y.sum())
    actual_default_rate = y.mean() * 100
    flagged_customers = int(analysis_df["predicted_default"].sum())
    flagged_rate = flagged_customers / total_customers * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Customers analysed", f"{total_customers:,}")
    col2.metric("Actual defaults in dataset", f"{actual_defaults:,}")
    col3.metric("Historical default rate", f"{actual_default_rate:.2f}%")
    col4.metric("Flagged at selected threshold", f"{flagged_rate:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.markdown("#### Actual default outcomes")
        outcome_df = pd.DataFrame({
            "Outcome": ["No default", "Default"],
            "Customers": [
                int((y == 0).sum()),
                int((y == 1).sum())
            ]
        })
        
        fig = px.pie(
            outcome_df,
            names="Outcome",
            values="Customers",
            hole=0.55,
            title="Observed outcomes in the dataset",
            color="Outcome",
            color_discrete_map={
                "No default": GREEN,
                "Default": GOLD
            }
        )
        fig = style_chart(fig)

        
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Predicted risk categories")
        risk_counts = (
            analysis_df["risk_category"]
            .value_counts()
            .reindex(["Low", "Medium", "High"], fill_value=0)
            .rename_axis("Risk category")
            .reset_index(name="Customers")
        )
        fig = px.bar(
            risk_counts,
            x="Risk category",
            y="Customers",
            title="Customers by predicted risk category"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # BUSINESS INSIGHTS

    st.divider()
    st.subheader("Business Insights")
    st.write(
        "Explore how historical default rates vary by repayment status "
        "and credit limit."
    )

    insight_left, insight_right = st.columns(2)

    with insight_left:
        repayment_df = (
            df.groupby("pay_0", observed=True)[TARGET]
            .agg(["mean", "count"])
            .reset_index()
        )
        repayment_df["default_rate_percent"] = repayment_df["mean"] * 100

        fig = px.bar(
            repayment_df,
            x="pay_0",
            y="default_rate_percent",
            hover_data=["count"],
            labels={
                "pay_0": "Most recent repayment status",
                "default_rate_percent": "Historical default rate (%)",
                "count": "Number of customers"
            },
            title="Default rate by repayment status"
        )
        st.plotly_chart(fig, use_container_width=True)

    with insight_right:
        credit_df = df.copy()
        credit_df["credit_limit_group"] = pd.cut(
            credit_df["credit_limit"],
            bins=[0, 50000, 100000, 180000, 270000, float("inf")],
            labels=[
                "10k–50k",
                "50k–100k",
                "100k–180k",
                "180k–270k",
                "270k+"
            ],
            include_lowest=True
        )

        credit_insights = (
            credit_df.groupby("credit_limit_group", observed=True)[TARGET]
            .agg(["mean", "count"])
            .reset_index()
        )
        credit_insights["default_rate_percent"] = (
            credit_insights["mean"] * 100
        )

        fig = px.bar(
            credit_insights,
            x="credit_limit_group",
            y="default_rate_percent",
            hover_data=["count"],
            labels={
                "credit_limit_group": "Credit limit group",
                "default_rate_percent": "Historical default rate (%)",
                "count": "Number of customers"
            },
            title="Default rate by credit limit"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "These charts describe historical associations in the dataset. "
        "They do not prove that repayment status or credit limit causes "
        "default."
    )
    

    st.info(
        "The dataset contains historical customer outcomes. "
        "Predictions shown here are model estimates, not guarantees "
        "of future behaviour or lending decisions."
    )


# --------------------------------------------------
# 7. CUSTOMER RISK ANALYSIS
# --------------------------------------------------

elif page == "Customer Risk Analysis":
    st.subheader("Customer Risk Analysis")

    risk_filter = st.multiselect(
        "Filter by risk category",
        options=["Low", "Medium", "High"],
        default=["Low", "Medium", "High"]
    )

    filtered_df = analysis_df[
        analysis_df["risk_category"].astype(str).isin(risk_filter)
    ]

    st.write(f"Customers displayed: {len(filtered_df):,}")

    display_columns = [
        "credit_limit",
        "age",
        "pay_0",
        "predicted_default_probability",
        "risk_category",
        TARGET
    ]

    display_df = filtered_df[display_columns].copy()
    display_df["predicted_default_probability"] = (
        display_df["predicted_default_probability"] * 100
    ).round(2)

    display_df = display_df.rename(columns={
        "credit_limit": "Credit limit",
        "age": "Age",
        "pay_0": "Most recent repayment status",
        "predicted_default_probability": "Predicted default probability (%)",
        "risk_category": "Risk category",
        TARGET: "Actual default (historical)"
    })

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered customer analysis",
        data=csv_data,
        file_name="customer_risk_analysis.csv",
        mime="text/csv"
    )

    st.caption(
        "Historical outcomes are displayed for analysis. "
        "In a real deployment, those outcomes would not be available "
        "when scoring a new customer."
    )


# --------------------------------------------------
# 8. MODEL PERFORMANCE
# --------------------------------------------------

elif page == "Model Performance":
    st.subheader("Model Performance")

    st.markdown(
        "These are results from the held-out test set recorded during "
        "model training, not metrics recalculated on this full dataset."
    )

    if not comparison_df.empty:
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    else:
        st.warning("model_comparison.csv was not found.")

    if not threshold_df.empty:
        st.markdown("#### Threshold comparison")
        st.dataframe(threshold_df, use_container_width=True, hide_index=True)

        metric_choice = st.selectbox(
            "Select a metric to compare",
            ["precision", "recall", "f1_score"]
        )

        if {"threshold", metric_choice}.issubset(threshold_df.columns):
            fig = px.line(
                threshold_df,
                x="threshold",
                y=metric_choice,
                markers=True,
                title=f"{metric_choice.replace('_', ' ').title()} by threshold"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("threshold_comparison.csv was not found.")


# --------------------------------------------------
# 9. FEATURE IMPORTANCE
# --------------------------------------------------

elif page == "Feature Importance":
    st.subheader("What influenced the Random Forest model?")

    if importance_df.empty:
        st.warning("feature_importance.csv was not found.")
    else:
        feature_col = next(
            (c for c in ["feature", "Feature", "feature_name"]
             if c in importance_df.columns),
            None
        )
        importance_col = next(
            (c for c in ["importance_percent", "importance", "Importance"]
             if c in importance_df.columns),
            None
        )

        if feature_col and importance_col:
            plot_df = importance_df.sort_values(
                importance_col, ascending=False
            ).head(15).sort_values(importance_col)

            fig = px.bar(
                plot_df,
                x=importance_col,
                y=feature_col,
                orientation="h",
                title="Top 15 model features"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                importance_df.sort_values(
                    importance_col, ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.error(
                "The feature importance CSV does not contain the "
                "expected feature and importance columns."
            )

    st.caption(
        "Feature importance describes how the model uses features. "
        "It does not establish causation or explain the direction "
        "of each feature's relationship with default."
    )


# --------------------------------------------------
# 10. SCORE ONE CUSTOMER
# --------------------------------------------------

elif page == "Score a Customer":
    st.subheader("Estimate an individual customer's risk")
    st.write(
        "Enter example customer information to obtain a model-estimated "
        "default probability. Use valid values for each field."
    )

    with st.form("customer_scoring_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            credit_limit = st.number_input(
                "Credit limit",
                min_value=1000,
                max_value=1000000,
                value=50000,
                step=1000
            )
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=35
            )
            gender = st.selectbox("Gender code", [1, 2], index=0)
            education = st.selectbox(
                "Education code", [1, 2, 3, 4, 5, 6, 0], index=1
            )
            marital_status = st.selectbox(
                "Marital status code", [1, 2, 3, 0], index=0
            )

        with c2:
            st.markdown("**Repayment status**")
            pay_0 = st.selectbox("Most recent month", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
            pay_2 = st.selectbox("2 months prior", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
            pay_3 = st.selectbox("3 months prior", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
            pay_4 = st.selectbox("4 months prior", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
            pay_5 = st.selectbox("5 months prior", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)
            pay_6 = st.selectbox("6 months prior", [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], index=2)

        with c3:
            st.markdown("**Bill amounts**")
            bill_amt1 = st.number_input("Latest bill amount", value=20000, step=1000)
            bill_amt2 = st.number_input("Bill amount 2 months prior", value=18000, step=1000)
            bill_amt3 = st.number_input("Bill amount 3 months prior", value=15000, step=1000)
            bill_amt4 = st.number_input("Bill amount 4 months prior", value=12000, step=1000)
            bill_amt5 = st.number_input("Bill amount 5 months prior", value=10000, step=1000)
            bill_amt6 = st.number_input("Bill amount 6 months prior", value=8000, step=1000)

        st.markdown("**Payment amounts**")
        p1, p2, p3 = st.columns(3)

        with p1:
            pay_amt1 = st.number_input("Latest payment", value=3000, step=500)
            pay_amt2 = st.number_input("Payment 2 months prior", value=2500, step=500)

        with p2:
            pay_amt3 = st.number_input("Payment 3 months prior", value=2000, step=500)
            pay_amt4 = st.number_input("Payment 4 months prior", value=1500, step=500)

        with p3:
            pay_amt5 = st.number_input("Payment 5 months prior", value=1000, step=500)
            pay_amt6 = st.number_input("Payment 6 months prior", value=1000, step=500)

        submitted = st.form_submit_button("Estimate risk")

    if submitted:
        customer = pd.DataFrame([{
            "credit_limit": credit_limit,
            "gender": gender,
            "education": education,
            "marital_status": marital_status,
            "age": age,
            "pay_0": pay_0,
            "pay_2": pay_2,
            "pay_3": pay_3,
            "pay_4": pay_4,
            "pay_5": pay_5,
            "pay_6": pay_6,
            "bill_amt1": bill_amt1,
            "bill_amt2": bill_amt2,
            "bill_amt3": bill_amt3,
            "bill_amt4": bill_amt4,
            "bill_amt5": bill_amt5,
            "bill_amt6": bill_amt6,
            "pay_amt1": pay_amt1,
            "pay_amt2": pay_amt2,
            "pay_amt3": pay_amt3,
            "pay_amt4": pay_amt4,
            "pay_amt5": pay_amt5,
            "pay_amt6": pay_amt6
        }])[FEATURE_NAMES]

        probability = float(model.predict_proba(customer)[0, 1])
        category = (
            "High" if probability >= 0.40
            else "Medium" if probability >= 0.20
            else "Low"
        )

        st.metric(
            "Estimated probability of default",
            f"{probability:.1%}"
        )
        st.write(f"**Demonstration risk category:** {category}")

        if probability >= threshold:
            st.warning(
                "This example meets or exceeds the selected threshold "
                "and should be flagged for further review in this demo."
            )
        else:
            st.success(
                "This example is below the selected threshold in this demo."
            )

        st.caption(
            "This is an educational model output, not financial advice, "
            "a credit decision, or a guarantee of future behaviour."
        )


# --------------------------------------------------
# 11. FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "Portfolio project | UCI Default of Credit Card Clients dataset | "
    "Educational use only — not for real lending decisions."
)
