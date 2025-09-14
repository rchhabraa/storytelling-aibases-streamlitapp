import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from fpdf import FPDF
import google.generativeai as genai

# -------------------------------
# Gemini API Key 
# -------------------------------
GEMINI_API_KEY = "AIzaSyByFCK9ZmN1xudTarr7jZq1GVM2BHZztOg"   


# -------------------------------
# Gemini Loader
# -------------------------------
def load_gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")


# -------------------------------
# Dataset summary generator
# -------------------------------
def generate_dataset_summary(df):
    buffer = []
    buffer.append("Dataset Shape: " + str(df.shape))
    buffer.append("\nColumn Types:\n" + str(df.dtypes))
    buffer.append("\nMissing Values:\n" + str(df.isnull().sum()))
    buffer.append("\nBasic Statistics:\n" + str(df.describe(include="all").transpose()))
    return "\n".join(buffer)


# -------------------------------
# Get EDA insights from Gemini
# -------------------------------
def get_gemini_eda_insights(df, gemini_model):
    summary = generate_dataset_summary(df)
    prompt = f"""
You are a data scientist. Perform Exploratory Data Analysis (EDA) 
on the following dataset summary. 
Provide 8–10 bullet points covering:
- Missing values
- Data distributions
- Potential outliers
- Correlations
- Interesting patterns
- Data quality issues
Keep the explanation clear and useful.

Dataset Summary:
{summary}
"""
    response = gemini_model.generate_content(prompt)
    return response.text


# -------------------------------
# Save Report to PDF
# -------------------------------
def save_report_pdf(heuristic_text, imgs, llm_text=None):
    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    pdf = FPDF()
    pdf.add_page()

    # Executive Summary
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Executive Summary", ln=True, align="C")
    pdf.ln(6)

    # Heuristic Insights
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Heuristic Insights", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 6, heuristic_text if heuristic_text else "No heuristic insights available.")
    pdf.ln(6)

    # LLM Insights
    if llm_text:
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "AI (Gemini) Insights", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 6, llm_text)
        pdf.ln(6)

    # Plots
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Visualizations", ln=True)
    for im in imgs:
        pdf.add_page()
        pdf.image(im, x=10, w=190)

    pdf.output(pdf_path)
    return pdf_path


# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config("AI Data Storyteller", layout="wide")
st.title("📊 AI Data Storyteller Dashboard")

# Sidebar Input: only file upload (no API key needed now)
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    # Load data
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        df = pd.read_csv(uploaded_file, encoding="latin1")

    # -------------------------------
    # Dataset Preview + Validation
    # -------------------------------
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", list(df.columns))

    # Validation checks
    msgs = []
    if df.shape[0] == 0:
        msgs.append("⚠️ Empty dataset")
    if df.columns.duplicated().any():
        msgs.append("⚠️ Duplicate column names")
    missing = (df.isna().mean() * 100).round(1)
    if (missing > 50).any():
        msgs.append("⚠️ Some columns have >50% missing values")
    if msgs:
        for m in msgs:
            st.warning(m)

    # -------------------------------
    # Data Preprocessing Section
    # -------------------------------
    st.subheader("🧹 Data Preprocessing")

    # Missing values table
    missing_table = pd.DataFrame({
        "Missing Values": df.isnull().sum(),
        "Percentage (%)": (df.isnull().mean() * 100).round(2)
    })
    missing_table = missing_table[missing_table["Missing Values"] > 0].sort_values("Missing Values", ascending=False)

    if not missing_table.empty:
        st.write("### Missing Values Summary")
        st.dataframe(missing_table)

        # Highlight if critical missing
        critical_cols = missing_table[missing_table["Percentage (%)"] > 50]
        if not critical_cols.empty:
            st.warning(f"⚠️ Columns with >50% missing values: {', '.join(critical_cols.index.tolist())}")
    else:
        st.success("✅ No missing values found in the dataset.")

    # -------------------------------
    # Automated EDA (Summary + Heuristics)
    # -------------------------------
    st.subheader("📈 Automated EDA (Heuristic Insights)")
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    insights = []
    if len(num_cols) > 0:
        high_var = df[num_cols].var().sort_values(ascending=False)
        insights.append(
            f"Highest variance numeric cols: {', '.join(high_var.head(3).index.tolist())}"
        )
    if len(cat_cols) > 0:
        top_cat = df[cat_cols[0]].value_counts().head(3).to_dict()
        insights.append(f"Top categories in {cat_cols[0]}: {top_cat}")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        c = corr.abs().unstack().sort_values(ascending=False)
        c = c[c < 1]
        if not c.empty:
            top_pair = c.drop_duplicates().index[0]
            insights.append(
                f"Top numeric correlation: {top_pair} = {corr.loc[top_pair[0], top_pair[1]]:.2f}"
            )

    heuristic_text = "\n".join(["- " + it for it in insights])
    st.markdown(heuristic_text if heuristic_text else "No heuristic insights available.")

    # -------------------------------
    # AI Insights via Gemini
    # -------------------------------
    st.subheader("🤖 Gemini LLM Insights")
    try:
        model = load_gemini()
        with st.spinner("Calling Gemini LLM for insights..."):
            llm_text = get_gemini_eda_insights(df, model)
        st.success("LLM analysis complete.")
        st.markdown(llm_text)
    except Exception as e:
        st.error("Gemini LLM call failed: " + str(e))

    # -------------------------------
    # Visualizations (3+ meaningful plots)
    # -------------------------------
    st.subheader("📊 Auto Visualizations")
    cols = st.columns(3)
    img_files = []

    # Histogram
    if num_cols:
        best_num = df[num_cols].var().sort_values(ascending=False).index[0]
        fig1, ax1 = plt.subplots(figsize=(5,4))
        sns.histplot(df[best_num].dropna(), bins=40, kde=True, ax=ax1)
        ax1.set_title(f"Distribution: {best_num}")
        cols[0].pyplot(fig1)
        tmp1 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig1.savefig(tmp1.name); img_files.append(tmp1.name)
        plt.close(fig1)

    # Bar Chart
    if cat_cols:
        best_cat = None
        for c in cat_cols:
            if df[c].nunique() <= 30:
                best_cat = c
                break
        if best_cat:
            fig2, ax2 = plt.subplots(figsize=(5,4))
            vc = df[best_cat].value_counts().nlargest(15)
            sns.barplot(x=vc.index, y=vc.values, ax=ax2)
            ax2.set_title(f"Counts: {best_cat}")
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")
            cols[1].pyplot(fig2)
            tmp2 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig2.savefig(tmp2.name, bbox_inches="tight"); img_files.append(tmp2.name)
            plt.close(fig2)

    # Correlation Heatmap
    if len(num_cols) >= 2:
        fig3, ax3 = plt.subplots(figsize=(5,4))
        sns.heatmap(df[num_cols].corr(), cmap="coolwarm", center=0, annot=False, ax=ax3)
        ax3.set_title("Correlation Heatmap")
        cols[2].pyplot(fig3)
        tmp3 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig3.savefig(tmp3.name, bbox_inches="tight"); img_files.append(tmp3.name)
        plt.close(fig3)

    # Line chart (extra plot)
    if num_cols:
        fig4, ax4 = plt.subplots(figsize=(6,4))
        df[num_cols].head(50).plot(ax=ax4)
        ax4.set_title("Line Chart (first 50 rows)")
        st.pyplot(fig4)
        tmp4 = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig4.savefig(tmp4.name, bbox_inches="tight"); img_files.append(tmp4.name)
        plt.close(fig4)

    # -------------------------------
    # Report Generation (PDF Export)
    # -------------------------------
    if st.button("📑 Generate PDF Report"):
        with st.spinner("Generating report..."):
            pdf_path = save_report_pdf(heuristic_text, img_files, llm_text)
            with open(pdf_path, "rb") as f:
                st.download_button("Download Report (PDF)", f, file_name="EDA_Report.pdf")