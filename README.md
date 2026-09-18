# SWIL Invoice Converter

Streamlit app to upload supplier invoices as PDF/JPG/PNG, review extracted items, and export SWIL `MargNewCSV` format.

## Deployment

For Streamlit Community Cloud, `packages.txt` installs the system Tesseract OCR engine required for JPG/PNG OCR. `requirements.txt` installs the Python packages.

## Local

Run:
`python -m streamlit run app.py`
