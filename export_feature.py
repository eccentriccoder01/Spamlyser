import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO
from fpdf import FPDF

def dataframe_to_pdf(df, title="Spamlyser Results"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    # Table header
    for col in df.columns:
        pdf.cell(35, 8, str(col), border=1)
    pdf.ln()
    # Table rows
    for i in range(len(df)):
        for col in df.columns:
            value = str(df.iloc[i][col])
            # Truncate long cell text for PDF
            if len(value) > 30:
                value = value[:27] + "..."
            pdf.cell(35, 8, value, border=1)
        pdf.ln()
    # Output as bytes
    pdf_bytes = BytesIO(pdf.output(dest="S").encode('latin-1'))
    return pdf_bytes

def export_results_button(history, filename_prefix="spamlyser_results"):
    """
    Displays a button & dropdown to export spam analysis results as CSV or PDF.
    :param history: List of dicts with analysis results
    :param filename_prefix: Prefix for the exported file
    """
    if not history:
        st.info("No results to export yet.")
        return

    # Prepare DataFrame
    df = pd.DataFrame(history)
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(str)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    export_format = st.selectbox(
        "Export results as:",
        options=["CSV", "PDF"],
        key=f"export_format_{filename_prefix}_{ts}"
    )

    if export_format == "CSV":
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_data,
            file_name=f"{filename_prefix}_{ts}.csv",
            mime="text/csv"
        )
    elif export_format == "PDF":
        pdf_data = dataframe_to_pdf(df)
        st.download_button(
            label="📥 Download Results PDF",
            data=pdf_data,
            file_name=f"{filename_prefix}_{ts}.pdf",
            mime="application/pdf"
        )