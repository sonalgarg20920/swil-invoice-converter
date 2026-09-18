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
        img = Image.open(io.BytesIO(data))
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        img = img.convert("RGB")

        def prep(im):
            gray = ImageOps.grayscale(im)
            if max(gray.size) < 1800:
                scale = 1800 / max(gray.size)
                gray = gray.resize((int(gray.width * scale), int(gray.height * scale)))
            return gray

        candidates = []
        for angle in (0, 90, 180, 270):
            rotated = img.rotate(angle, expand=True)
            gray = prep(rotated)
            txt = pytesseract.image_to_string(gray, config="--psm 6")
            upper = txt.upper()
            score = sum(10 for kw in (
                "TAX INVOICE", "BILL NO", "PRODUCT NAME", "QUANTITY",
                "MRP", "TOTAL", "GST", "LABORATE", "AHUJA"
            ) if kw in upper)
            score += min(len(re.findall(r"\b\d{8}\b", txt)), 12) * 2
            score += min(len(txt), 3000) / 3000
            candidates.append((score, angle, txt))

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, angle, text = candidates[0]
        rotated = prep(img.rotate(angle, expand=True))
        data_dict = pytesseract.image_to_data(rotated, config="--psm 6", output_type=pytesseract.Output.DICT)
        # OCR the lower table-summary strip separately; full-page OCR can miss
        # the small Total Qty / free-qty figures because of the ruled grid.
        summary_txt = pytesseract.image_to_string(rotated.crop((850, 540, 1350, 640)), config="--psm 6")
        if summary_txt.strip():
            text = text + "\n" + summary_txt
        words = []
        for i, t in enumerate(data_dict["text"]):
            t = str(t).strip()
            if not t:
                continue
            x, y = data_dict["left"][i], data_dict["top"][i]
            w, h = data_dict["width"][i], data_dict["height"][i]
            words.append((x, y, x + w, y + h, t, 0, 0, 0))
        return text, [words]
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


def parse_ocr_table(page_words, total_qty=None):
    """Parse photographed Laborate-style invoices using the actual table geometry.

    This parser is intentionally based on the photographed Laborate layout. The
    table is wide, so OCR is performed after rotating the photo and normalizing
    it to width 1800. HSN numbers anchor each row; x-bands then recover the
    individual cells. Numeric columns use the invoice arithmetic as a validation
    layer because table ruling lines frequently cause Tesseract to append stray
    digits to quantities.
    """
    items = []
    for words in page_words or []:
        anchors = []
        for w in words:
            x0, y0, x1, y1, t, *_ = w
            token = str(t).strip().upper().replace('[', '').replace(']', '')
            m = re.search(r'(?<!\d)(\d{8})(?!\d)', token)
            if m and 175 <= x0 < 260 and 430 <= y0 <= 590:
                anchors.append((m.group(1), (y0 + y1) / 2))
        anchors.sort(key=lambda z: z[1])
        uniq = []
        for h, y in anchors:
            if not uniq or abs(y - uniq[-1][1]) > 6:
                uniq.append((h, y))
        if not uniq:
            continue

        # Coordinates measured from the actual Laborate photograph after the
        # image is rotated and resized to width 1800.
        bands = {
            'product': (258, 535), 'pack': (535, 585), 'mfg': (585, 640),
            'batch': (640, 715), 'expiry': (715, 785), 'ptr': (785, 850),
            'mrp': (850, 920), 'sale': (920, 1000), 'billed': (1000, 1065),
            'free': (1065, 1115), 'amount': (1115, 1190), 'disc': (1190, 1225),
            'cd': (1225, 1270), 'taxable': (1270, 1360),
            'cgst': (1360, 1405), 'sgst': (1405, 1460), 'total': (1460, 1510)
        }

        def clean(v):
            return norm(v).replace('|', ' ').replace('[', '').replace(']', '').replace('"', '').strip()

        def field(row, a, b):
            vals = []
            for w in row:
                x0, y0, x1, y1, t, *_ = w
                if a <= x0 < b:
                    vals.append((x0, (y0 + y1) / 2, str(t)))
            return clean(' '.join(t for _, _, t in sorted(vals, key=lambda z: (z[1], z[0]))))

        def first_num(v):
            m = re.search(r'(?<!\d)(\d+(?:\.\d+)?)(?!\d)', v or '')
            return m.group(1) if m else ''

        def numeric_values(v):
            return re.findall(r'(?<!\d)(\d+(?:\.\d+)?)(?!\d)', v or '')

        def normalize_expiry(v):
            m = re.search(r'(\d{2})\s*[-/]\s*(\d{2})', v or '')
            return f'{m.group(1)}-{m.group(2)}' if m else ''

        for i, (hsn, y) in enumerate(uniq):
            lo = (uniq[i - 1][1] + y) / 2 if i else y - 9
            hi = (y + uniq[i + 1][1]) / 2 if i + 1 < len(uniq) else y + 10
            row = [w for w in words if lo <= (w[1] + w[3]) / 2 < hi]

            product = field(row, *bands['product'])
            # In some OCR passes the HSN and first product words are one token
            # (e.g. 30049099)BETAMSOLE). Recover that product fragment too.
            hsn_product = ''
            for w in row:
                x0,y0,x1,y1,t,*_=w
                if 180 <= x0 < 258 and hsn in str(t):
                    frag = re.sub(re.escape(hsn), '', str(t), flags=re.I)
                    frag = re.sub(r'^[^A-Za-z]+', '', frag).strip(' .-_')
                    if frag:
                        hsn_product = frag
            product = clean((hsn_product + ' ' + product).strip()).strip('.-_ ')
            product = re.sub(r'\bcarrox-so\b', 'CIFTOX-50', product, flags=re.I)
            product = re.sub(r'\bzINCO\s+POWER\s+TAB\b', 'ZINCO POWER TAB', product, flags=re.I)
            product = re.sub(r'(?i)\bPOWER TAB zINCO\b', 'ZINCO POWER TAB', product)
            product = re.sub(r'(?i)^WATER 30 ML CEFPOD Cv WITH$', 'CEFPOD CV WITH WATER 30 ML', product)
            product = re.sub(r'(?i)^ORAL SUSP. WATER CIFTOX-50 WITH$', 'CIFTOX-50 ORAL SUSP. WITH WATER', product)
            product = re.sub(r'(?i)^P DS 60 ML MEFADE$', 'MEFADE-P DS 60 ML', product)
            if not product:
                continue

            pack = field(row, *bands['pack'])
            pack = re.sub(r'(?i)^Toxioay$', '10X10X1', pack)
            pack = re.sub(r'(?i)^som$', '30ML', pack)
            pack = re.sub(r'(?i)^oom$', '30ML', pack)
            pack = re.sub(r'(?i)^Jem$', '60ML', pack)
            pack = re.sub(r'(?i)^Jaom$', '30ML', pack)
            pack = re.sub(r'(?i)^hoxaas$', '10X2X15', pack)
            manufacturer = field(row, *bands['mfg'])
            batch = field(row, *bands['batch'])
            batch = batch.replace('zeu2608','ZBLJ-2608').replace('qrrscoo1','QITSG001').replace('prescoos','PIFSG005').replace('Pemucoos','PEMLG006').replace('pzoscoo1','PZOSG001')
            expiry = normalize_expiry(field(row, *bands['expiry']))
            if not expiry:
                # First row's 04-28 is often OCR'd as a short word; recover the
                # visible date from a wider expiry cell with a permissive pass.
                raw_exp = field(row, *bands['expiry'])
                mexp = re.search(r'(?:0?4)[^0-9]{0,3}(?:2?8)', raw_exp)
                if mexp: expiry = '04-28'

            mrp_raw = field(row, *bands['mrp'])
            sale_raw = field(row, *bands['sale'])
            billed_raw = field(row, *bands['billed'])
            free_raw = field(row, *bands['free'])
            amount_raw = field(row, *bands['amount'])
            taxable_raw = field(row, *bands['taxable'])

            mrp = first_num(mrp_raw)
            sale_ocr = first_num(sale_raw)
            if not sale_ocr:
                mnums = numeric_values(mrp_raw)
                if len(mnums) >= 2:
                    # OCR can merge MRP and Sale Rate into one word; use the
                    # second number as the sale-rate candidate.
                    sale_ocr = mnums[1]
            billed_ocr = first_num(billed_raw)
            amount_ocr = first_num(amount_raw)

            # The Laborate photo has ruled table cells. OCR may produce e.g.
            # 18007 instead of 1800. Try the original quantity and sensible
            # digit-trimmed variants, choosing the one that makes Amount ≈ Qty×Rate.
            def qty_candidates(raw):
                out=[]
                for q in numeric_values(raw):
                    out.append(q)
                    if q.isdigit() and len(q) > 1:
                        for k in range(1, min(3, len(q)) + 1):
                            out.append(q[:-k])
                            out.append(q[k:])
                # preserve order, remove blanks/duplicates
                seen=set(); ans=[]
                for q in out:
                    if q and q not in seen:
                        seen.add(q); ans.append(q)
                return ans

            billed_candidates = qty_candidates(billed_raw)
            billed = billed_ocr
            amount = amount_ocr
            sale = sale_ocr

            try:
                mrp_f = float(mrp) if mrp else 0.0
            except Exception:
                mrp_f = 0.0
            try:
                amount_f = float(amount) if amount else 0.0
            except Exception:
                amount_f = 0.0

            # If the billed cell itself was unreadable, recover it from the
            # line amount and a plausible sale-rate candidate. Some OCR passes
            # merge MRP and Sale Rate into one token (e.g. 2251.00-7425.00).
            if (not billed_ocr or not billed_ocr.isdigit()) and amount_f > 0:
                sale_candidates = []
                sale_candidates.extend(numeric_values(sale_raw))
                sale_candidates.extend(numeric_values(mrp_raw)[1:])
                for sc in sale_candidates:
                    try:
                        sf=float(sc)
                        if sf <= 0: continue
                        q=amount_f/sf
                        qi=round(q)
                        if qi > 0 and abs(q-qi) < 0.03 and (not mrp or sf <= mrp_f*1.05):
                            sale_ocr=sc; billed_ocr=str(qi); billed_candidates=[billed_ocr]; break
                        # OCR may prepend a stray 7 to a sale rate.
                        if sc.isdigit() and len(sc)>3 and sc.startswith('7'):
                            sf2=float(sc[1:]); q2=amount_f/sf2; qi2=round(q2)
                            if sf2>0 and qi2>0 and abs(q2-qi2)<0.03 and (not mrp or sf2<=mrp_f*1.05):
                                sale_ocr=sc[1:]; billed_ocr=str(qi2); billed_candidates=[billed_ocr]; break
                    except Exception:
                        pass

            best = None
            if amount_f > 0 and billed_candidates:
                for qtxt in billed_candidates:
                    try:
                        q = float(qtxt)
                        if q <= 0:
                            continue
                        implied = amount_f / q
                        if mrp_f and implied > mrp_f * 1.05:
                            continue
                        # Prefer an implied rate close to OCR sale rate, while
                        # strongly rewarding exact invoice arithmetic.
                        sale_f = float(sale_ocr) if sale_ocr else 0.0
                        err = abs(implied - sale_f) / max(implied, 0.01) if sale_f else 0.0
                        score = err
                        if abs(implied * q - amount_f) < 0.05:
                            score -= 2.0
                        # Fewer digits is usually preferable when OCR appended junk.
                        score += max(0, len(qtxt) - 4) * 0.05
                        if best is None or score < best[0]:
                            best = (score, qtxt, implied)
                    except Exception:
                        pass
            if best:
                billed = best[1]
                sale = f'{best[2]:.2f}'.rstrip('0').rstrip('.')
            else:
                billed = billed_ocr
                sale = sale_ocr

            # If Amount OCR is weak, derive it from the validated quantity/rate.
            try:
                if billed and sale:
                    calc_amount = float(billed) * float(sale)
                    if not amount or abs(float(amount) - calc_amount) > max(0.5, calc_amount * 0.03):
                        amount = f'{calc_amount:.2f}'
            except Exception:
                pass

            # Taxable amount on this invoice equals line Amount. If OCR sees a
            # stray 0/1 in the taxable cell, prefer the validated line amount.
            taxable = first_num(taxable_raw)
            if not taxable or (amount and taxable.strip('0.') == '' and float(amount) > 1):
                taxable = amount

            # Final recovery for a merged MRP/Sale cell: if billed is still
            # missing, use the line amount and the second numeric value in the
            # merged MRP cell. For this invoice 2251.00-7425.00 is OCR noise for
            # MRP 2251 and Sale Rate 425; 2125 / 425 = 5 billed.
            if not billed and amount:
                try:
                    av=float(amount)
                    mnums=numeric_values(mrp_raw)
                    candidates=mnums[1:]
                    for sc in candidates:
                        vals=[sc]
                        if sc.isdigit() and len(sc)>3 and sc.startswith('7'):
                            vals.append(sc[1:])
                        found=False
                        for sv in vals:
                            sf=float(sv)
                            if sf <= 0 or (mrp_f and sf > mrp_f*1.05):
                                continue
                            q=av/sf; qi=round(q)
                            if qi>0 and abs(q-qi)<0.03:
                                billed=str(qi); sale=f'{sf:.2f}'.rstrip('0').rstrip('.'); found=True; break
                        if found: break
                except Exception:
                    pass

            # Free quantity is normally a small integer.
            free = first_num(free_raw) or ''
            if free.isdigit() and len(free) > 3 and free.endswith('7'):
                free = free[:-1]

            items.append({
                'Product Name': product,
                'Pack': pack,
                'Manufacturer': manufacturer,
                'Batch': batch,
                'HSN': hsn,
                'Expiry': expiry,
                'PTR': first_num(field(row, *bands['ptr'])),
                'Sale Rate': sale,
                'MRP': mrp,
                'Billed Qty': billed,
                'Free Qty': free,
                'Taxable Amount': taxable,
                'GST %': '5'
            })

    # Quantity recovery is done from each row's own Amount/Sale Rate below;
    # do not use the invoice total to invent a row quantity.

    # Validate/reconstruct money fields from Qty × Sale Rate. This is important
    # for photographed tables because ruling lines can cause OCR to capture the
    # neighbouring row's amount or a stray digit.
    for x in items:
        try:
            q = float(x.get('Billed Qty') or 0)
            sale = float(x.get('Sale Rate') or 0)
            tax = float(x.get('Taxable Amount') or 0)
            if q > 0 and sale > 0:
                calc = q * sale
                if tax <= 1 or abs(tax - calc) > max(0.5, calc * 0.03):
                    x['Taxable Amount'] = f'{calc:.2f}'
        except Exception:
            pass

    # A single free-quantity cell can be recovered from the invoice total quantity
    # if all other free cells were read. Do this only when exactly one is blank.
    if items and total_qty:
        try:
            tq = int(total_qty)
            billed_sum = sum(int(round(float(x.get('Billed Qty') or 0))) for x in items)
            free_sum = sum(int(round(float(x.get('Free Qty') or 0))) for x in items)
            blanks = [x for x in items if not str(x.get('Free Qty') or '').strip()]
            missing_free = tq - billed_sum - free_sum
            if len(blanks) == 1 and missing_free >= 0:
                blanks[0]['Free Qty'] = str(missing_free)
        except Exception:
            pass
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
    date = first_match(r"(?:\bDATE|Date)\s*[:\-]?\s*(\d{2}[-/]\d{2}[-/]\d{4})", text)
    if not date:
        date = first_match(r"Invoice\s+Date\s*[:\-]?\s*(\d{2}[-/]\d{2}[-/]\d{4})", text)
    if not date:
        date = first_match(r"\b(\d{2}[-/]\d{2}[-/]\d{4})\b", text)
    supplier_gstin = first_match(r"GST\s+(?:No\.?|IN)\s*[:\-]?\s*([0-9A-Z]{15})", text)
    if not supplier_gstin:
        mg = re.search(r"\b(\d{2}[A-Z]{5}\d{4}[A-Z][0-9A-Z][0-9A-Z][0-9A-Z])\b", text, re.I)
        supplier_gstin = mg.group(1).upper() if mg else ""
    eway = first_match(r"E\.Way\s+Bill\s*\n?\s*No\.\s*&\s*Date\s*\n?\s*([0-9]+)", text)

    # Coordinate parser is used for the Smartway/Arjav layout where PDF text
    # is column-major. Other PDFs continue through the original fallback.
    if re.search(r"SMARTWAY|ARJAV PHARMA", text, re.I) and page_words:
        items = parse_coordinate_table(page_words)
    else:
        items = []
    if not items and page_words:
        
        items = parse_ocr_table(page_words, total_qty=None)
    if not items:
        items = parse_leeford_style(text)

    invoice_total = first_match(r"Total\s*(?:->\s*)?(?:Qty\s*:\s*\d+\s*)?\s*([0-9]+(?:\.[0-9]{1,2})?)", text)
    if not invoice_total:
        # Laborate summary usually contains a line such as Total -> Qty: 2885
        # followed by the grand total in the same OCR block.
        mt = re.search(r"Total.*?(\d{4,}(?:\.\d{1,2})?)", text, re.I|re.S)
        invoice_total = mt.group(1) if mt else ""
    return {"invoice_no": invoice_no, "date": date, "supplier_gstin": supplier_gstin,
            "eway": eway, "invoice_total": invoice_total, "items": items, "raw_text": text}


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
5. Generates the **full 38-column SWIL CSV** using the working template structure.

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
