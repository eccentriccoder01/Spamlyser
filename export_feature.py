import streamlit as st
import pandas as pd
from datetime import datetime
from io import BytesIO
from fpdf import FPDF

def dataframe_to_pdf(df, title="Spamlyser Results"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", "B", 14)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("helvetica", size=9)
    col_width = pdf.w / (len(df.columns) + 1)
    row_height = pdf.font_size * 1.5

    # Header
    for col in df.columns:
        pdf.cell(col_width, row_height, str(col), border=1)
    pdf.ln(row_height)

    # Rows
    for i in range(len(df)):
        for col in df.columns:
            val = str(df.iloc[i][col])
            if len(val) > 25:
                val = val[:22] + "..."
            pdf.cell(col_width, row_height, val, border=1)
        pdf.ln(row_height)

    pdf_bytes = BytesIO(pdf.output(dest="S").encode('latin-1'))
    return pdf_bytes

def export_results_button(history, filename_prefix="spamlyser_results"):
    if not history:
        st.info("No results to export yet.")
        return
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