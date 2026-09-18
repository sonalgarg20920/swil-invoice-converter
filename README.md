# SWIL Invoice → CSV

A local Streamlit interface for converting supplier invoice PDFs/JPGs into a CSV based on an existing SWIL import CSV template.

## Run on Mac

```bash
cd swil_invoice_converter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL shown by Streamlit (normally http://localhost:8501).

## Image invoices

For JPG/PNG OCR, install Tesseract once:

```bash
brew install tesseract
```

PDF invoices use PyMuPDF and do not require Tesseract.

## Important

The app deliberately does not invent SWIL/MARG item codes. If the supplier invoice doesn't contain a SWIL item code, the item-code/link field stays blank and can be mapped in SWIL.

The bundled `swil_template.csv` is the user's working SWIL CSV structure. For a different SWIL export/version, upload a known-good SWIL CSV as the template in the sidebar.

## Recommended next upgrade

Once 3–5 invoices from different suppliers have been tested, add supplier-specific parsers so the extraction becomes much more reliable and requires less manual correction.
