# app.py
import io
import re
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# Payment & Emergency Contact Audit Tool (Streamlit)
# - Input workbook contains:
#     1) Uzio Data
#     2) ADP Payment Data
#     3) ADP Emergency Contact Data
#     4) Mapping Sheet
# - Generates Excel report and provides download button
#
# OUTPUT TABS (same structure as previous tool):
#   - Summary
#   - Field_Summary_By_Status
#   - Mapping_ADP_Col_Missing
#   - Comparison_Detail_AllFields
#   - Mismatches_Only
#
# ADP is source of truth.
# =========================================================

APP_TITLE = "UZIO vs ADP – Payment & Emergency Contact Comparison Tool"
OUTPUT_FILENAME = "UZIO_vs_ADP_Payment_EmergencyContact_Report_ADP_SourceOfTruth.xlsx"

UZIO_SHEET = "Uzio Data"
ADP_PAY_SHEET = "ADP Payment Data"
ADP_EC_SHEET = "ADP Emergency Contact Data"
MAP_SHEET = "Mapping Sheet"

# ---------- UI: Hide sidebar + Streamlit chrome ----------
st.set_page_config(page_title=APP_TITLE, layout="centered", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
      [data-testid="stSidebar"] { display: none !important; }
      [data-testid="collapsedControl"] { display: none !important; }
      header { display: none !important; }
      footer { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Helpers ----------
def norm_colname(c: str) -> str:
    if c is None:
        return ""
    c = str(c).replace("\n", " ").replace("\r", " ")
    c = c.replace("\u00A0", " ")
    c = c.replace("’", "'").replace("“", '"').replace("”", '"')
    c = re.sub(r"\s+", " ", c).strip()
    c = c.replace("*", "")
    c = c.strip('"').strip("'")
    return c

def norm_blank(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    if isinstance(x, str) and x.strip().lower() in {"", "nan", "none", "null"}:
        return ""
    return x

def digits_only(x):
    x = norm_blank(x)
    if x == "":
        return ""
    return re.sub(r"\D", "", str(x))

def try_parse_date(x):
    x = norm_blank(x)
    if x == "":
        return ""
    if isinstance(x, (datetime, date, np.datetime64, pd.Timestamp)):
        return pd.to_datetime(x).date().isoformat()
    if isinstance(x, str):
        s = x.strip()
        try:
            return pd.to_datetime(s, errors="raise").date().isoformat()
        except Exception:
            return s
    return str(x)

NUMERIC_KEYWORDS = {"salary", "rate", "hours", "amount", "percent", "percentage"}
DATE_KEYWORDS = {"date", "dob", "birth", "effective"}
ZIP_KEYWORDS = {"zip", "zipcode", "postal"}
PHONE_KEYWORDS = {"phone", "mobile"}

def norm_phone_digits(x):
    return digits_only(x)

def norm_zip_first5(x):
    x = norm_blank(x)
    if x == "":
        return ""
    if isinstance(x, (int, np.integer)):
        s = str(int(x))
    elif isinstance(x, (float, np.floating)) and float(x).is_integer():
        s = str(int(x))
    else:
        s = re.sub(r"[^\d]", "", str(x).strip())
    if s == "":
        return ""
    if 0 < len(s) < 5:
        s = s.zfill(5)
    return s[:5]

def norm_value(x, field_name: str):
    f = norm_colname(field_name).casefold()
    x = norm_blank(x)
    if x == "":
        return ""

    if any(k in f for k in PHONE_KEYWORDS):
        return norm_phone_digits(x)

    if any(k in f for k in ZIP_KEYWORDS):
        return norm_zip_first5(x)

    if any(k in f for k in DATE_KEYWORDS):
        return try_parse_date(x)

    if any(k in f for k in NUMERIC_KEYWORDS):
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        if isinstance(x, str):
            s = x.strip().replace(",", "").replace("$", "")
            try:
                return float(s)
            except Exception:
                return re.sub(r"\s+", " ", x.strip()).casefold()

    if isinstance(x, str):
        return re.sub(r"\s+", " ", x.strip()).casefold()

    return str(x).casefold()

def norm_key_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(object).where(~s.isna(), "")
    def _fix(v):
        v = str(v).strip()
        v = v.replace("\u00A0", " ")
        if re.fullmatch(r"\d+\.0+", v):
            v = v.split(".")[0]
        return v
    return s2.map(_fix)

def find_col(df_cols, *candidate_names):
    """Return actual column name from df_cols matching any candidate (case-insensitive, normalized)."""
    norm_map = {norm_colname(c).casefold(): c for c in df_cols}
    for cand in candidate_names:
        key = norm_colname(cand).casefold()
        if key in norm_map:
            return norm_map[key]
    return None

def resolve_adp_col_label(label: str, adp_cols_all) -> str:
    """
    Mapping sheet sometimes has values like:
      'ASSOCIATE ID (or “Associate ID”)'
      'DEPOSIT PERCENT,' (trailing comma)
    This resolves to the real column name present in either ADP sheet.
    """
    if label is None:
        return ""
    raw = str(label).strip()
    raw = raw.replace("’", "'").replace("“", '"').replace("”", '"')
    raw = raw.strip().strip(",")
    if raw == "":
        return ""

    adp_norm = {norm_colname(c).casefold(): c for c in adp_cols_all}

    # Try direct normalized match
    direct = norm_colname(raw).casefold()
    if direct in adp_norm:
        return adp_norm[direct]

    # Try split candidates by common separators
    parts = re.split(r"\(|\)|\bor\b|/|,|;", raw, flags=re.IGNORECASE)
    parts = [norm_colname(p) for p in parts if norm_colname(p)]
    # also try the left side before " - " or " – "
    extra = []
    for p in parts:
        extra.extend([norm_colname(x) for x in re.split(r"\s[-–]\s", p) if norm_colname(x)])
    parts = parts + extra

    for p in parts:
        k = norm_colname(p).casefold()
        if k in adp_norm:
            return adp_norm[k]

    # As a last resort, try partial containment
    for k_norm, actual in adp_norm.items():
        if k_norm and (k_norm in direct or direct in k_norm):
            return actual

    return ""

# ---------- Section detection ----------
PAYMENT_FIELDS_HINTS = {"routing", "account", "deposit", "paycheck", "priority", "payment method", "distribution"}
CONTACT_FIELDS_HINTS = {"contact", "relationship", "mobile", "phone", "primary"}

def detect_section_from_adp_col(adp_col: str, pay_cols, ec_cols) -> str:
    if adp_col in pay_cols:
        return "Payment"
    if adp_col in ec_cols:
        return "Emergency Contact"
    # fallback by name hint
    c = norm_colname(adp_col).casefold()
    if any(h in c for h in CONTACT_FIELDS_HINTS):
        return "Emergency Contact"
    return "Payment"

# ---------- Record key builders ----------
def build_payment_base_key(df: pd.DataFrame, emp_col: str):
    routing_col = find_col(df.columns, "ROUTING NUMBER", "Routing Number")
    acct_col = find_col(df.columns, "ACCOUNT NUMBER", "Account Number")
    dep_type_col = find_col(df.columns, "DEPOSIT TYPE", "Deposit Type")
    dep_amt_col = find_col(df.columns, "DEPOSIT AMOUNT", "Deposit Amount")
    dep_pct_col = find_col(df.columns, "DEPOSIT PERCENT", "Deposit Percent", "DEPOSIT PERCENTAGE", "Deposit Percentage")

    def _row_key(r):
        emp = norm_key_series(pd.Series([r.get(emp_col, "")])).iloc[0]
        routing = digits_only(r.get(routing_col, "")) if routing_col else ""
        acct = digits_only(r.get(acct_col, "")) if acct_col else ""

        dep_type = norm_value(r.get(dep_type_col, ""), "Deposit Type") if dep_type_col else ""
        amt = norm_value(r.get(dep_amt_col, ""), "Deposit Amount") if dep_amt_col else ""
        pct = norm_value(r.get(dep_pct_col, ""), "Deposit Percent") if dep_pct_col else ""

        # Prefer bank identifiers when present
        if routing != "" or acct != "":
            return f"{emp}|BANK|{routing}|{acct}"
        # fallback when bank details missing
        return f"{emp}|NOBANK|{dep_type}|{amt}|{pct}"

    return df.apply(_row_key, axis=1)

def build_contact_base_key(df: pd.DataFrame, emp_col: str):
    name_col = find_col(df.columns, "Contact Name", "NAME", "Name")
    phone_col = find_col(df.columns, "Mobile Phone", "Phone", "MOBILE PHONE")
    rel_col = find_col(df.columns, "Relationship Description", "Relationship")

    def _row_key(r):
        emp = norm_key_series(pd.Series([r.get(emp_col, "")])).iloc[0]
        nm = norm_value(r.get(name_col, ""), "Contact Name") if name_col else ""
        ph = norm_phone_digits(r.get(phone_col, "")) if phone_col else ""
        rl = norm_value(r.get(rel_col, ""), "Relationship Description") if rel_col else ""
        # name+phone is typically enough; add relationship to reduce collisions
        return f"{emp}|{nm}|{ph}|{rl}"

    return df.apply(_row_key, axis=1)

def filter_section_rows(uzio_df: pd.DataFrame, section: str):
    """UZIO sheet may contain both sections; infer by which fields are populated."""
    if section == "Payment":
        routing = find_col(uzio_df.columns, "Routing Number", "ROUTING NUMBER")
        acct = find_col(uzio_df.columns, "Account Number", "ACCOUNT NUMBER")
        dep = find_col(uzio_df.columns, "Paycheck Distribution", "Deposit Type")
        pct = find_col(uzio_df.columns, "Paycheck Percentage", "Deposit Percent")
        amt = find_col(uzio_df.columns, "Paycheck Amount", "Deposit Amount")
        cols = [c for c in [routing, acct, dep, pct, amt] if c]
    else:
        name = find_col(uzio_df.columns, "Name", "Contact Name")
        rel = find_col(uzio_df.columns, "Relationship", "Relationship Description")
        ph = find_col(uzio_df.columns, "Phone", "Mobile Phone")
        cols = [c for c in [name, rel, ph] if c]

    if not cols:
        return uzio_df.iloc[0:0].copy()

    mask = False
    for c in cols:
        mask = mask | (uzio_df[c].map(norm_blank) != "")
    return uzio_df[mask].copy()

# ---------- Core compare ----------
def run_comparison(file_bytes: bytes) -> dict:
    xls = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")

    uzio = pd.read_excel(xls, sheet_name=UZIO_SHEET, dtype=object)
    adp_pay = pd.read_excel(xls, sheet_name=ADP_PAY_SHEET, dtype=object)
    adp_ec  = pd.read_excel(xls, sheet_name=ADP_EC_SHEET, dtype=object)
    mapping = pd.read_excel(xls, sheet_name=MAP_SHEET, dtype=object)

    # normalize column headers
    uzio.columns = [norm_colname(c) for c in uzio.columns]
    adp_pay.columns = [norm_colname(c) for c in adp_pay.columns]
    adp_ec.columns  = [norm_colname(c) for c in adp_ec.columns]
    mapping.columns = [norm_colname(c) for c in mapping.columns]

    # mapping columns: accept both spellings
    uz_col_name = None
    adp_col_name = None
    for c in mapping.columns:
        if norm_colname(c).casefold() in {"uzio coloumn", "uzio column"}:
            uz_col_name = c
        if norm_colname(c).casefold() in {"adp coloumn", "adp column"}:
            adp_col_name = c
    if uz_col_name is None or adp_col_name is None:
        raise ValueError("Mapping sheet must contain columns: 'UZIO Column' and 'ADP Column' (or 'Uzio Coloumn'/'ADP Coloumn').")

    mapping[uz_col_name] = mapping[uz_col_name].map(norm_colname)
    mapping[adp_col_name] = mapping[adp_col_name].map(norm_colname)

    mapping_valid = mapping.dropna(subset=[uz_col_name, adp_col_name]).copy()
    mapping_valid = mapping_valid[(mapping_valid[uz_col_name] != "") & (mapping_valid[adp_col_name] != "")]
    mapping_valid = mapping_valid.drop_duplicates(subset=[uz_col_name], keep="first").copy()

    # Resolve ADP columns against both ADP tabs
    adp_all_cols = list(adp_pay.columns) + list(adp_ec.columns)
    mapping_valid["ADP_Resolved_Column"] = mapping_valid[adp_col_name].map(lambda x: resolve_adp_col_label(x, adp_all_cols))

    # Identify key columns
    UZIO_KEY = find_col(uzio.columns, "Employee ID", "EmployeeID", "Employee Id")
    if UZIO_KEY is None:
        raise ValueError("UZIO key column 'Employee ID' not found in Uzio Data tab.")

    # ADP key columns per sheet
    ADP_PAY_KEY = find_col(adp_pay.columns, "ASSOCIATE ID", "Associate ID")
    ADP_EC_KEY  = find_col(adp_ec.columns,  "Associate ID", "ASSOCIATE ID")
    if ADP_PAY_KEY is None:
        raise ValueError("ADP Payment Data must contain 'ASSOCIATE ID' (or 'Associate ID').")
    if ADP_EC_KEY is None:
        raise ValueError("ADP Emergency Contact Data must contain 'Associate ID' (or 'ASSOCIATE ID').")

    # normalize key values
    uzio[UZIO_KEY] = norm_key_series(uzio[UZIO_KEY])
    adp_pay[ADP_PAY_KEY] = norm_key_series(adp_pay[ADP_PAY_KEY])
    adp_ec[ADP_EC_KEY]   = norm_key_series(adp_ec[ADP_EC_KEY])

    # Split UZIO into two sections (based on populated columns)
    uzio_pay = filter_section_rows(uzio, "Payment")
    uzio_ec  = filter_section_rows(uzio, "Emergency Contact")

    # Build base record keys
    uzio_pay["_base_key"] = build_payment_base_key(uzio_pay, UZIO_KEY) if len(uzio_pay) else pd.Series(dtype=str)
    adp_pay["_base_key"]  = build_payment_base_key(adp_pay, ADP_PAY_KEY)

    uzio_ec["_base_key"]  = build_contact_base_key(uzio_ec, UZIO_KEY) if len(uzio_ec) else pd.Series(dtype=str)
    adp_ec["_base_key"]   = build_contact_base_key(adp_ec, ADP_EC_KEY)

    # Group rows by base_key (to support duplicates)
    def group_indices(df: pd.DataFrame):
        g = {}
        for idx, k in df["_base_key"].items():
            k2 = "" if norm_blank(k) == "" else str(k)
            g.setdefault(k2, []).append(idx)
        return g

    g_uz_pay = group_indices(uzio_pay) if len(uzio_pay) else {}
    g_ad_pay = group_indices(adp_pay)
    g_uz_ec  = group_indices(uzio_ec) if len(uzio_ec) else {}
    g_ad_ec  = group_indices(adp_ec)

    # Determine section for each mapping row
    pay_cols_set = set(adp_pay.columns)
    ec_cols_set  = set(adp_ec.columns)

    mapping_valid["Section"] = mapping_valid["ADP_Resolved_Column"].map(
        lambda c: detect_section_from_adp_col(c, pay_cols_set, ec_cols_set) if c else "UNKNOWN"
    )

    # fields to compare (exclude Employee ID mapping itself)
    mapped_fields = [f for f in mapping_valid[uz_col_name].tolist() if norm_colname(f).casefold() != "employee id"]

    # Mapping missing ADP col
    mapping_missing_adp_col = mapping_valid[(mapping_valid["ADP_Resolved_Column"] == "")].copy()

    # ---------- Compare ----------
    def build_report_for_section(
        section: str,
        uz_df: pd.DataFrame,
        ad_df: pd.DataFrame,
        g_uz: dict,
        g_ad: dict,
        emp_key_uz: str,
        emp_key_ad: str,
    ) -> bytes:
        """
        Build an output workbook (bytes) for a single section (Payment or Emergency Contact).
        Supports multiple records per employee by pairing records within the same base key.
        """
        # mapping rows for this section
        sec_map = mapping_valid[mapping_valid["Section"] == section].copy()
        sec_map = sec_map[sec_map["ADP_Resolved_Column"] != ""]

        # Mapping missing ADP columns (best-effort section guess using raw ADP label)
        missing = mapping_valid[mapping_valid["ADP_Resolved_Column"] == ""].copy()
        if len(missing):
            missing["Section_Guess"] = missing[adp_col_name].map(
                lambda c: detect_section_from_adp_col(c, pay_cols_set, ec_cols_set) if c else "UNKNOWN"
            )
            mapping_missing_adp_col = missing[(missing["Section_Guess"] == section) | (missing["Section_Guess"] == "UNKNOWN")].copy()
        else:
            mapping_missing_adp_col = missing

        # which base keys to compare
        base_keys = sorted(set(g_uz.keys()).union(g_ad.keys()) - {""})

        rows = []
        for bk in base_keys:
            uz_list = g_uz.get(bk, [])
            ad_list = g_ad.get(bk, [])
            pairs = max(len(uz_list), len(ad_list))

            for i in range(pairs):
                uz_idx = uz_list[i] if i < len(uz_list) else None
                ad_idx = ad_list[i] if i < len(ad_list) else None

                record_key = f"{bk}#{i+1}"

                for _, r in sec_map.iterrows():
                    uz_field = r[uz_col_name]
                    adp_col_resolved = r["ADP_Resolved_Column"]

                    # Values (single UZIO value column only)
                    uz_val = (
                        uz_df.loc[uz_idx, uz_field]
                        if (uz_idx is not None and uz_field in uz_df.columns)
                        else ""
                    )
                    adp_val = (
                        ad_df.loc[ad_idx, adp_col_resolved]
                        if (ad_idx is not None and adp_col_resolved in ad_df.columns)
                        else ""
                    )

                    # status logic (ADP truth)
                    if ad_idx is None and uz_idx is not None:
                        status = "MISSING_IN_ADP"
                    elif ad_idx is not None and uz_idx is None:
                        status = "MISSING_IN_UZIO"
                    elif adp_col_resolved not in ad_df.columns:
                        status = "ADP_COLUMN_MISSING"
                    elif uz_field not in uz_df.columns:
                        status = "UZIO_COLUMN_MISSING"
                    else:
                        uz_n = norm_value(uz_val, uz_field)
                        ad_n = norm_value(adp_val, uz_field)

                        if (uz_n == ad_n) or (uz_n == "" and ad_n == ""):
                            status = "OK"
                        elif uz_n == "" and ad_n != "":
                            status = "UZIO_MISSING_VALUE"
                        elif uz_n != "" and ad_n == "":
                            status = "ADP_MISSING_VALUE"
                        else:
                            status = "MISMATCH"

                    rows.append(
                        {
                            "Employee ID": bk,
                            "Section": section,
                            "Record Key": record_key,
                            "Field": uz_field,
                            "UZIO_Value": uz_val,  # single UZIO value column
                            "ADP_Value": adp_val,
                            "ADP_SourceOfTruth_Status": status,
                        }
                    )

        comparison_detail = pd.DataFrame(
            rows,
            columns=[
                "Employee ID",
                "Section",
                "Record Key",
                "Field",
                "UZIO_Value",
                "ADP_Value",
                "ADP_SourceOfTruth_Status",
            ],
        )

        mismatches_only = comparison_detail[comparison_detail["ADP_SourceOfTruth_Status"] != "OK"].copy()

        # Field Summary By Status
        if len(comparison_detail):
            comparison_detail["_FieldKey"] = comparison_detail["Section"] + " :: " + comparison_detail["Field"]
            statuses = [
                "OK",
                "MISMATCH",
                "UZIO_MISSING_VALUE",
                "ADP_MISSING_VALUE",
                "MISSING_IN_UZIO",
                "MISSING_IN_ADP",
                "ADP_COLUMN_MISSING",
                "UZIO_COLUMN_MISSING",
            ]
            field_summary_by_status = (
                comparison_detail.pivot_table(
                    index="_FieldKey",
                    columns="ADP_SourceOfTruth_Status",
                    values="Employee ID",
                    aggfunc="count",
                    fill_value=0,
                )
                .reindex(columns=statuses, fill_value=0)
                .reset_index()
                .rename(columns={"_FieldKey": "Field"})
            )
            field_summary_by_status["Total"] = field_summary_by_status[statuses].sum(axis=1)
        else:
            field_summary_by_status = pd.DataFrame(
                columns=[
                    "Field",
                    "OK",
                    "MISMATCH",
                    "UZIO_MISSING_VALUE",
                    "ADP_MISSING_VALUE",
                    "MISSING_IN_UZIO",
                    "MISSING_IN_ADP",
                    "ADP_COLUMN_MISSING",
                    "UZIO_COLUMN_MISSING",
                    "Total",
                ]
            )

        # Summary
        uzio_emp = set(uz_df[emp_key_uz].dropna().map(str)) if (len(uz_df) and emp_key_uz in uz_df.columns) else set()
        adp_emp = set(ad_df[emp_key_ad].dropna().map(str)) if (len(ad_df) and emp_key_ad in ad_df.columns) else set()

        summary = pd.DataFrame(
            {
                "Metric": [
                    "Section",
                    "Total UZIO Employees",
                    "Total ADP Employees",
                    "Employees in both",
                    "Employees only in UZIO",
                    "Employees only in ADP",
                    "Total UZIO Records",
                    "Total ADP Records",
                    "Fields Compared",
                    "Mapping Missing ADP Col (this section)",
                    "Total Comparisons (field-level rows)",
                    "Total Mismatches/Issues (non-OK)",
                ],
                "Value": [
                    section,
                    len(uzio_emp),
                    len(adp_emp),
                    len(uzio_emp & adp_emp),
                    len(uzio_emp - adp_emp),
                    len(adp_emp - uzio_emp),
                    len(uz_df),
                    len(ad_df),
                    len(sec_map),
                    int(mapping_missing_adp_col.shape[0]),
                    int(comparison_detail.shape[0]),
                    int(mismatches_only.shape[0]),
                ],
            }
        )

        # write excel
        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="Summary", index=False)
            field_summary_by_status.to_excel(writer, sheet_name="Field_Summary_By_Status", index=False)
            mapping_missing_adp_col.to_excel(writer, sheet_name="Mapping_ADP_Col_Missing", index=False)
            comparison_detail.to_excel(writer, sheet_name="Comparison_Detail_AllFields", index=False)
            mismatches_only.to_excel(writer, sheet_name="Mismatches_Only", index=False)

        return out.getvalue()

    # Build independent reports
    payment_bytes = build_report_for_section(
        section="Payment",
        uz_df=uzio_pay,
        ad_df=adp_pay,
        g_uz=g_uz_pay,
        g_ad=g_ad_pay,
        emp_key_uz=UZIO_KEY,
        emp_key_ad=ADP_PAY_KEY,
    )

    emergency_contact_bytes = build_report_for_section(
        section="Emergency Contact",
        uz_df=uzio_ec,
        ad_df=adp_ec,
        g_uz=g_uz_ec,
        g_ad=g_ad_ec,
        emp_key_uz=UZIO_KEY,
        emp_key_ad=ADP_EC_KEY,
    )

    return {
        "payment": payment_bytes,
        "emergency_contact": emergency_contact_bytes,
    }


# ---------- Minimal UI ----------
st.title(APP_TITLE)
st.write("Upload the Excel workbook (.xlsx). The tool will generate the audit report and provide a download button.")

uploaded_file = st.file_uploader("Upload Excel workbook", type=["xlsx"])
run_btn = st.button("Run Audit", type="primary", disabled=(uploaded_file is None))

if run_btn:
    try:
        with st.spinner("Running audit..."):
            reports = run_comparison(uploaded_file.getvalue())

        st.success("Reports generated (Payment + Emergency Contact).")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        payment_name = f"UZIO_vs_ADP_Payment_Report_ADP_SourceOfTruth_{ts}.xlsx"
        ec_name = f"UZIO_vs_ADP_EmergencyContact_Report_ADP_SourceOfTruth_{ts}.xlsx"

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Payment Report (.xlsx)",
                data=reports["payment"],
                file_name=payment_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )
        with col2:
            st.download_button(
                label="Download Emergency Contact Report (.xlsx)",
                data=reports["emergency_contact"],
                file_name=ec_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="secondary",
            )
    except Exception as e:
        st.error(f"Failed: {e}")
