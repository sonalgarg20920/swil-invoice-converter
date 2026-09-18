import io, re, csv
import shutil
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SWIL Invoice → CSV", page_icon="🧾", layout="wide")
st.title("🧾 SWIL Invoice → CSV")
st.caption("Upload a supplier invoice PDF/JPG/PNG, review the extracted items, and export a SWIL CSV using the exact 38-column structure of the known-good working file.")

END_COL = 37  # Exact 38-column structure of the known-good SWIL CSV.


def extract_document(uploaded_file):
    data = uploaded_file.getvalue()
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        try:
            import fitz
            doc = fitz.open(stream=data, filetype="pdf")
            texts, pages = [], []
            for page in doc:
                texts.append(page.get_text("text"))
                pages.append(page.get_text("words"))
            return "\n".join(texts), pages
        except Exception as e:
            raise RuntimeError(f"Could not read PDF: {e}")
    try:
        from PIL import Image, ImageOps
        import pytesseract
        if not shutil.which("tesseract"):
            raise RuntimeError("Tesseract OCR engine is not installed on this server. Add packages.txt with tesseract-ocr and redeploy.")
        img = Image.open(io.BytesIO(data)).convert("RGB")
        # Light preprocessing improves OCR on photographed/low-contrast invoices.
        gray = ImageOps.grayscale(img)
        if max(gray.size) < 1800:
            scale = 1800 / max(gray.size)
            gray = gray.resize((int(gray.width * scale), int(gray.height * scale)))
        text = pytesseract.image_to_string(gray, config="--psm 6")
        return text, None
    except Exception as e:
        raise RuntimeError(f"Could not OCR the image: {e}") from e


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def first_match(pattern, text, flags=re.I | re.M):
    m = re.search(pattern, text, flags)
    return norm(m.group(1)) if m else ""


def parse_coordinate_table(page_words):
    """Parse Smartway/Arjav-style PDFs using word coordinates.

    Normal PDF text extraction can be column-major, so reading lines loses the
    visual row structure. Coordinates rebuild each numbered row and preserve
    duplicate rows (e.g. two identical promotional items).
    """
    items = []
    for words in page_words or []:
        serials = []
        for w in words:
            x0, y0, x1, y1, t, *_ = w
            if x0 < 22 and re.fullmatch(r"\d+\.?", str(t).strip()) and y0 > 220:
                serials.append((int(str(t).strip().rstrip(".")), (y0 + y1) / 2))
        serials.sort(key=lambda z: z[1])
        anchors = []
        expected = 1
        for sr, y in serials:
            if sr == expected and y < 330:
                anchors.append((sr, y))
                expected += 1
        if not anchors:
            continue

        def col(x):
            if x < 22: return 0       # Sr
            if x < 62: return 1       # HSN
            if x < 170: return 2      # Product
            if x < 204: return 3      # Pack
            if x < 232: return 4      # Mfg
            if x < 292: return 5      # Batch
            if x < 320: return 6      # Exp
            if x < 355: return 7      # MRP
            if x < 382: return 8      # PTR
            if x < 430: return 9      # Sale rate
            if x < 499: return 10     # Billed qty / qty disc area
            if x < 540: return 12     # Amount
            if x < 565: return 13     # Disc %
            if x < 605: return 14     # Taxable
            if x < 638: return 15     # CGST %
            if x < 665: return 16     # CGST amount
            if x < 698: return 17     # SGST %
            if x < 725: return 18     # SGST amount
            if x < 760: return 19     # Cess
            return 20                  # Total

        for i, (sr, y) in enumerate(anchors):
            lo = (anchors[i - 1][1] + y) / 2 if i else y - 5
            hi = (y + anchors[i + 1][1]) / 2 if i + 1 < len(anchors) else y + 7
            row = [w for w in words if lo <= (w[1] + w[3]) / 2 < hi]
            groups = {i: [] for i in range(21)}
            for w in row:
                x0, y0, x1, y1, t, *_ = w
                groups[col(x0)].append((x0, (y0 + y1) / 2, str(t)))
            vals = {k: " ".join(t for _, _, t in sorted(v)) for k, v in groups.items() if v}
            # Keep fields strictly separated by their visual columns. Some invoices
            # wrap product names into the next line, so never let Batch leak into
            # Product Name even when OCR/text extraction is imperfect.
            hsn = vals.get(1, "").strip()
            if not re.fullmatch(r"\d{8}", hsn):
                continue
            # Quantity columns are visually separate on the invoice:
            # x≈433–478 = Billed Qty, x≈478–499 = Qty Disc (Free Qty).
            # Promotional/free rows can have ONLY Qty Disc (e.g. the two
            # Welspun towels), so a number in the Qty Disc column must never
            # be treated as billed quantity.
            billed_parts = []
            free_parts = []
            for w in row:
                x0, y0, x1, y1, t, *_ = w
                tx = str(t).strip()
                if 430 <= x0 < 478 and re.fullmatch(r"\d+(?:\.\d+)?", tx):
                    billed_parts.append(tx)
                elif 478 <= x0 < 499 and re.fullmatch(r"\d+(?:\.\d+)?", tx):
                    free_parts.append(tx)
            billed = billed_parts[0] if billed_parts else ""
            free = free_parts[0] if free_parts else ""
            cgst = norm(vals.get(15, ""))
            sgst = norm(vals.get(17, ""))
            if cgst and sgst:
                try:
                    gst = f"{float(cgst) + float(sgst):g}"
                except ValueError:
                    gst = cgst
            else:
                gst = cgst or sgst or "5"
            product = norm(vals.get(2, ""))
            pack = norm(vals.get(3, ""))
            manufacturer = norm(vals.get(4, ""))
            batch = norm(vals.get(5, ""))
            # Defensive cleanup: if a parser/OCR pass accidentally appends the
            # batch token to the product field, remove only that exact token.
            if batch:
                product = re.sub(r"(?<!\w)" + re.escape(batch) + r"(?!\w)", "", product, flags=re.I)
                product = norm(product)
            items.append({
                "Product Name": product,
                "Pack": pack,
                "Manufacturer": manufacturer,
                "Batch": batch,
                "HSN": hsn,
                "Expiry": norm(vals.get(6, "")),
                "PTR": norm(vals.get(8, "")),
                "Sale Rate": norm(vals.get(9, "")),
                "MRP": norm(vals.get(7, "")),
                "Billed Qty": billed,
                "Free Qty": free,  # Qty Disc from the invoice maps to SWIL Free Qty.
                "Taxable Amount": norm(vals.get(14, "")),
                "GST %": gst,
            })
    return items


def parse_leeford_style(text):
    """Fallback parser for the original Leeford-style invoices."""
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    items = []
    hsn_positions = [i for i, l in enumerate(lines) if re.fullmatch(r"\d{8}", l)]
    for pos in hsn_positions:
        hsn = lines[pos]
        if pos < 3:
            continue
        batch = ""
        manufacturer = ""
        pack = ""
        for j in range(max(0, pos - 4), pos):
            if re.fullmatch(r"[A-Z0-9-]{4,}", lines[j], re.I) and any(c.isalpha() for c in lines[j]):
                if lines[j].upper() != "LEEFORD" and not re.fullmatch(r"\d+", lines[j]):
                    batch = lines[j]
            if lines[j].upper() == "LEEFORD":
                manufacturer = lines[j]
        for j in range(max(0, pos - 7), pos):
            if re.search(r"(\d+X\d+|\b\d+\s*(?:ML|GM)\b)", lines[j], re.I):
                pack = lines[j]
        candidates = []
        for j in range(max(0, pos - 10), pos):
            s = lines[j]
            if s in {manufacturer, batch, pack} or re.fullmatch(r"\d+(?:\.\d+)?", s):
                continue
            if re.fullmatch(r"\d{2}-\d{2}", s) or re.fullmatch(r"\d{8}", s):
                continue
            if s.upper() in {"WELLNESS - W", "ESSENCIA - I", "ALL", "TOTAL"}:
                continue
            if re.search(r"PTR|Sale Rate|Billed|Tax Invoice|Product Name|HSN", s, re.I):
                continue
            candidates.append(s)
        product = next((s for s in reversed(candidates) if len(s) >= 4 and not re.fullmatch(r"\d+", s)), "")
        window = " ".join(lines[pos + 1:pos + 12])
        exp_m = re.search(r"(\d{2}-\d{2})", window)
        expiry = exp_m.group(1) if exp_m else ""
        nums = re.findall(r"\d+(?:\.\d+)?", window)
        ptr = nums[0] if len(nums) > 0 else ""
        sale = nums[1] if len(nums) > 1 else ""
        mrp = nums[2] if len(nums) > 2 else ""
        tax_m = re.search(r"(\d+(?:\.\d+)?)\s+2\.50\s+(\d+(?:\.\d+)?)\s+2\.50\s+(\d+(?:\.\d+)?)", window)
        taxable = tax_m.group(1) if tax_m else ""
        items.append({
            "Product Name": product, "Pack": pack, "Manufacturer": manufacturer,
            "Batch": batch, "HSN": hsn, "Expiry": expiry, "PTR": ptr,
            "Sale Rate": sale, "MRP": mrp, "Billed Qty": "", "Free Qty": "",  # Qty Disc -> Free Qty
            "Taxable Amount": taxable, "GST %": "5"
        })
    return items


def parse_invoice(text, page_words=None):
    invoice_no = first_match(r"Bill\s+No\.?\s*:\s*([^\n]+)", text)
    date = first_match(r"\bDATE\s*\n?\s*(\d{2}-\d{2}-\d{4})", text)
    if not date:
        date = first_match(r"Invoice\s+Date\s+(\d{2}-\d{2}-\d{4})", text)
    supplier_gstin = first_match(r"GST\s+(?:No\.?|IN)\s*:\s*([0-9A-Z]{15})", text)
    eway = first_match(r"E\.Way\s+Bill\s*\n?\s*No\.\s*&\s*Date\s*\n?\s*([0-9]+)", text)

    # Coordinate parser is used for the Smartway/Arjav layout where PDF text
    # is column-major. Other PDFs continue through the original fallback.
    if re.search(r"SMARTWAY|ARJAV PHARMA", text, re.I) and page_words:
        items = parse_coordinate_table(page_words)
    else:
        items = []
    if not items:
        items = parse_leeford_style(text)

    return {"invoice_no": invoice_no, "date": date, "supplier_gstin": supplier_gstin,
            "eway": eway, "items": items, "raw_text": text}


def read_template(uploaded):
    if uploaded is None:
        p = Path(__file__).with_name("swil_template.csv")
        if not p.exists():
            return None
        data = p.read_bytes()
    else:
        data = uploaded.getvalue()
    for enc in ("cp1252", "utf-8-sig", "utf-8"):
        try:
            txt = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    return list(csv.reader(io.StringIO(txt)))


def swil_csv(template_rows, meta, items):
    """Generate the CSV according to the user's actual MargNewCSV import definition.

    Source of truth: the supplied SWIL/MARG XML definition.  SWIL maps only
    specific fields for H, T and F records.  We retain the proven 38-column
    physical layout, but only populate fields that are defined by that mapping.
    """
    def clean_date(v):
        s = str(v or '').strip()
        # Invoice date: DDMMYYYY.  Also accept DD-MM-YYYY / DD/MM/YYYY.
        m = re.fullmatch(r'(\d{2})[-/]?(\d{2})[-/]?(\d{4})', s)
        if m:
            return ''.join(m.groups())
        return s.replace('-', '').replace('/', '')

    def num(v):
        try:
            return float(str(v).replace(',', '').strip() or 0)
        except Exception:
            return 0.0

    def fmt_num(v, decimals=2):
        x = num(v)
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.{decimals}f}"

    def expiry_ddmmyyyy(v):
        s = str(v or '').strip()
        # Common invoice expiry forms: MM-YY, MM/YYYY, MMYY, MM-YYYY.
        m = re.fullmatch(r'(\d{1,2})[-/]?(\d{2}|\d{4})', s)
        if m:
            month = int(m.group(1))
            year = int(m.group(2))
            if year < 100:
                year += 2000
            if 1 <= month <= 12:
                return f"01{month:02d}{year:04d}"
        # Already DDMMYYYY.
        m = re.fullmatch(r'\d{8}', s)
        return s if m else s.replace('-', '').replace('/', '')

    # Always use 38 columns, matching the proven import file.  The XML
    # definition itself only maps positions 0..22 for the fields we need.
    WIDTH = 38
    out = []

    invoice_no = str(meta.get('invoice_no', '') or '').strip()
    invoice_date = clean_date(meta.get('date', ''))

    # H: A=Type, C=Invoice Number, D=Invoice Date.
    h = [''] * WIDTH
    h[0] = 'H'
    h[2] = invoice_no
    h[3] = invoice_date
    out.append(h)

    # T: A=Type, C=Company Name, F=Product, G=Pack, I=Batch,
    # J=Expiry (DDMMYYYY), O=PTS without tax, Q=MRP, U=Qty,
    # V=Free Qty, W=PTR Discount %.
    for item in items:
        r = [''] * WIDTH
        r[0] = 'T'
        company = str(item.get('Manufacturer', '') or '').strip()
        # The supplied definition maps Company Name to C (index 2).
        # Manufacturer is not a mapped field, so do not put it in B.
        r[2] = company
        r[5] = str(item.get('Product Name', '') or '').strip()
        r[6] = str(item.get('Pack', '') or '').strip()
        r[8] = str(item.get('Batch', '') or '').strip()
        r[9] = expiry_ddmmyyyy(item.get('Expiry', ''))
        r[14] = fmt_num(item.get('PTR', ''))
        r[16] = fmt_num(item.get('MRP', ''))
        r[20] = fmt_num(item.get('Billed Qty', ''))
        r[21] = fmt_num(item.get('Free Qty', ''))
        r[22] = fmt_num(item.get('PTR Discount %', '0'))
        out.append(r)

    # F: A=Type, B=Total Gross Amount, C=PTR Discount Amount.
    # Prefer the invoice total if extracted; otherwise sum taxable + GST.
    invoice_total = num(meta.get('invoice_total', ''))
    if not invoice_total:
        invoice_total = sum(
            num(x.get('Taxable Amount', '')) * (1 + num(x.get('GST %', '')) / 100)
            for x in items
        )
    ptr_discount = sum(
        num(x.get('PTR Discount Amount', '')) for x in items
    )
    f = [''] * WIDTH
    f[0] = 'F'
    f[1] = f"{invoice_total:.2f}"
    f[2] = f"{ptr_discount:.2f}"
    out.append(f)
    return out

st.sidebar.header("1. Upload")
invoice = st.sidebar.file_uploader("Invoice PDF / JPG / PNG", type=["pdf", "jpg", "jpeg", "png"])
template = st.sidebar.file_uploader("Optional: your working SWIL CSV template", type=["csv"])

if not invoice:
    st.info("Upload an invoice to begin.")
    st.markdown("""
### What this tool does
1. Reads the invoice.
2. Extracts supplier/invoice details and every numbered item row.
3. Preserves duplicate product rows.
4. Lets you correct the extracted table.
5. Generates the **full 51-column SWIL CSV** using the working template structure.

SWIL/MARG item-code matching is intentionally **not** used in this version.
""")
    st.stop()

try:
    raw, page_words = extract_document(invoice)
    result = parse_invoice(raw, page_words)
except Exception as e:
    st.error(str(e))
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Invoice", result["invoice_no"] or "Not found")
c2.metric("Date", result["date"] or "Not found")
c3.metric("Supplier GSTIN", result["supplier_gstin"] or "Not found")
c4.metric("Extracted items", len(result["items"]))

st.subheader("2. Review / correct invoice items")
st.caption("Invoice 'Qty Disc' is mapped to SWIL 'Free Qty'. Promotional/free rows with a quantity only under 'Qty Disc' are treated as Billed Qty = 0 and Free Qty = that quantity.")
df = pd.DataFrame(result["items"])
if df.empty:
    st.warning("No line items were detected. Use the raw text below to diagnose the invoice layout.")
else:
    edited = st.data_editor(df, use_container_width=True, num_rows="dynamic", hide_index=True,
        column_config={
            "Billed Qty": st.column_config.NumberColumn(format="%.0f"),
            "Free Qty": st.column_config.NumberColumn("Free Qty (Qty Disc)", format="%.0f"),
            "PTR": st.column_config.NumberColumn(format="%.2f"),
            "Sale Rate": st.column_config.NumberColumn(format="%.2f"),
            "MRP": st.column_config.NumberColumn(format="%.2f"),
            "Taxable Amount": st.column_config.NumberColumn(format="%.2f"),
        })

    st.subheader("3. Generate SWIL CSV")
    if st.button("Generate CSV", type="primary"):
        try:
            rows = swil_csv(read_template(template), result, edited.to_dict("records"))
            buf = io.StringIO()
            csv.writer(buf, lineterminator="\n").writerows(rows)
            csv_bytes = buf.getvalue().encode("cp1252", errors="replace")
            filename = f"SWIL_{result['invoice_no'] or 'invoice'}.csv".replace("/", "_")
            st.download_button("⬇️ Download SWIL CSV", data=csv_bytes, file_name=filename, mime="text/csv")
            st.success(f"CSV generated with {len(edited)} item rows and {len(rows[0])} columns (known-good SWIL structure).")
        except Exception as e:
            st.error(f"Could not generate CSV: {e}")

with st.expander("Raw extracted text (for troubleshooting)"):
    st.text_area("Invoice text", raw, height=350)
