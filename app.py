# app.py
import io
import re
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st

# =========================================================
# UZIO vs ADP – Payment & Emergency Contact Comparison Tool
#
# INPUT workbook tabs:
#   - Uzio Data
#   - ADP Payment Data
#   - ADP Emergency Contact Data
#   - Payment_Mapping
#   - Emergency_Mapping
#
# OUTPUT (two independent reports):
#   - Summary
#   - Field_Summary_By_Status   (columns removed: MISSING_IN_ADP, ADP_COLUMN_MISSING, UZIO_COLUMN_MISSING)
#   - Comparison_Detail_AllFields (NO FieldKey column)
#
# PAYMENT BUSINESS RULE (ADP -> UZIO-like normalization):
#   - FULL => percentage based
#   - Partial => amount based
#   - Partial% => percentage based
#   - Full + Partial => Full is last priority, Partial accounts before it
#   - Full + Partial% => Full% = 100 - sum(Partial%); Full is last priority
#
# ADDITIONAL FIXES:
#   1) UZIO "Flat Dollar" distribution is treated as "amount" (so no mismatch with ADP "amount")
#   2) UZIO FULL-row inference:
#        if an employee has multiple payment rows and exactly one row has BOTH
#        Paycheck Amount blank AND Paycheck Percentage blank,
#        treat that row as FULL/remainder:
#          - set Paycheck Distribution to "Percentage"
#          - set Paycheck Percentage to 100 - sum(other partial% if any), else 100
# =========================================================

APP_TITLE = "UZIO vs ADP – Payment & Emergency Contact Comparison Tool"

UZIO_SHEET = "Uzio Data"
ADP_PAY_SHEET = "ADP Payment Data"
ADP_EC_SHEET = "ADP Emergency Contact Data"
PAY_MAP_SHEET = "Payment_Mapping"
EC_MAP_SHEET = "Emergency_Mapping"

# ---------- UI ----------
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
    """Extract digits while handling numeric types safely (avoid '.0' artifacts)."""
    x = norm_blank(x)
    if x == "":
        return ""
    try:
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if isinstance(x, (float, np.floating)):
            if float(x).is_integer():
                return str(int(x))
    except Exception:
        pass

    s = str(x).strip()
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    return re.sub(r"\D", "", s)


def digits_only_padded(x, width: int):
    d = digits_only(x)
    if d == "":
        return ""
    if len(d) < width:
        d = d.zfill(width)
    return d


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


# include "priority" so 1 vs 1.0 doesn't mismatch
NUMERIC_KEYWORDS = {"salary", "rate", "hours", "amount", "percent", "percentage", "priority"}
DATE_KEYWORDS = {"date", "dob", "birth", "effective"}
ZIP_KEYWORDS = {"zip", "zipcode", "postal"}
PHONE_KEYWORDS = {"phone", "mobile"}
NAME_KEYWORDS = {"name"}
DISTRIBUTION_KEYWORDS = {"distribution", "deposit type"}

ROUTING_KEYWORDS = {"routing"}
ACCOUNTNUM_KEYWORDS = {"account number"}


def norm_phone_digits(x):
    """Normalize phone numbers so (410) 292-5939 == 4102925939."""
    d = digits_only(x)
    if d == "":
        return ""
    if len(d) == 11 and d.startswith("1"):
        d = d[1:]
    if len(d) > 10:
        d = d[-10:]
    return d


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


def normalize_person_name(x: str) -> str:
    """
    Normalize person names so these match:
      - "Bell, Ronald" == "Ronald Bell"
      - "Young, Roscoe Robert" == "Roscoe Young" (middle ignored)
    Compare FIRST + LAST only.
    """
    s = norm_blank(x)
    if s == "":
        return ""
    s = str(s).strip().replace("\u00A0", " ")

    def _clean_tokens(txt: str):
        txt = re.sub(r"[^A-Za-z0-9\s]", " ", txt)
        txt = re.sub(r"\s+", " ", txt).strip()
        return [t for t in txt.split(" ") if t]

    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        last_part = parts[0] if len(parts) >= 1 else ""
        first_part = parts[1] if len(parts) >= 2 else ""
        last_tokens = _clean_tokens(last_part)
        first_tokens = _clean_tokens(first_part)

        first = first_tokens[0] if first_tokens else ""
        last_name = last_tokens[-1] if last_tokens else ""

        if first and last_name:
            return f"{first} {last_name}".casefold()
        return (first or last_name).casefold()

    toks = _clean_tokens(s)
    if not toks:
        return ""
    if len(toks) == 1:
        return toks[0].casefold()
    first = toks[0]
    last_name = toks[-1]
    return f"{first} {last_name}".casefold()


def norm_distribution_token(val: str) -> str:
    """
    Normalize Paycheck Distribution / Deposit Type variants to two tokens:
      - 'amount'
      - 'percentage'
    """
    v = norm_blank(val)
    if v == "":
        return ""
    s = str(v).strip().casefold()

    # UZIO variants for amount
    if "flat dollar" in s or "flat amount" in s or "flat" in s and "dollar" in s:
        return "amount"
    if "dollar" in s and ("flat" in s or "amount" in s):
        return "amount"

    # Generic UZIO variants
    if "amount" in s:
        return "amount"
    if "%" in s or "percent" in s or "percentage" in s or "age" in s:
        return "percentage"

    # ADP variants
    if ("partial" in s and "%" in s) or "partial%" in s:
        return "percentage"
    if "partial" in s:
        return "amount"
    if "full" in s:
        return "percentage"

    return re.sub(r"\s+", " ", s).strip()


def norm_value(x, field_name: str):
    f = norm_colname(field_name).casefold()
    x = norm_blank(x)
    if x == "":
        return ""

    if any(k in f for k in PHONE_KEYWORDS):
        return norm_phone_digits(x)

    if any(k in f for k in ZIP_KEYWORDS):
        return norm_zip_first5(x)

    if any(k in f for k in ROUTING_KEYWORDS):
        return digits_only_padded(x, 9)

    if any(k in f for k in ACCOUNTNUM_KEYWORDS):
        return digits_only(x)

    if any(k in f for k in DATE_KEYWORDS):
        return try_parse_date(x)

    if any(k in f for k in NAME_KEYWORDS):
        return normalize_person_name(x)

    if any(k in f for k in DISTRIBUTION_KEYWORDS):
        return norm_distribution_token(x)

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
    norm_map = {norm_colname(c).casefold(): c for c in df_cols}
    for cand in candidate_names:
        key = norm_colname(cand).casefold()
        if key in norm_map:
            return norm_map[key]
    return None


def resolve_adp_col_label(label: str, adp_cols_all) -> str:
    """
    Resolve mapping sheet labels to actual ADP columns when possible.
    If cannot resolve, keep it as a constant: __CONST__:<label>
    """
    if label is None:
        return ""
    raw = str(label).strip()
    raw = raw.replace("’", "'").replace("“", '"').replace("”", '"')
    raw = raw.strip().strip(",")
    if raw == "":
        return ""

    adp_norm = {norm_colname(c).casefold(): c for c in adp_cols_all}

    direct = norm_colname(raw).casefold()
    if direct in adp_norm:
        return adp_norm[direct]

    parts = re.split(r"\(|\)|\bor\b|/|,|;", raw, flags=re.IGNORECASE)
    parts = [norm_colname(p) for p in parts if norm_colname(p)]

    extra = []
    for p in parts:
        extra.extend([norm_colname(x) for x in re.split(r"\s[-–]\s", p) if norm_colname(x)])
    parts = parts + extra

    for p in parts:
        k = norm_colname(p).casefold()
        if k in adp_norm:
            return adp_norm[k]

    for k_norm, actual in adp_norm.items():
        if k_norm and (k_norm in direct or direct in k_norm):
            return actual

    return f"__CONST__:{raw}"


# ---------- Numeric helpers ----------
def _safe_float(x):
    x = norm_blank(x)
    if x == "":
        return np.nan
    try:
        if isinstance(x, (int, float, np.integer, np.floating)):
            return float(x)
        s = str(x).strip().replace(",", "").replace("$", "")
        return float(s)
    except Exception:
        return np.nan


def _is_blank_money_or_percent(v) -> bool:
    s = str(norm_blank(v)).strip()
    if s == "":
        return True
    s2 = s.replace("$", "").replace(",", "").replace("%", "").strip()
    return s2 == ""


# ---------- Payment normalization (business rule) ----------
def normalize_adp_payment_table(adp_pay: pd.DataFrame, emp_col: str) -> pd.DataFrame:
    """
    Adds UZIO-like derived columns to ADP payment DF:
      - Paycheck Distribution (amount/percentage)
      - Paycheck Percentage
      - Paycheck Amount
      - Priority (recomputed so FULL is last when mixed)
      - Payment Method (constant 'Direct Deposit')

    Applies:
      - FULL + Partial => FULL last, Partial before it
      - FULL + Partial% => FULL% = 100 - sum(Partial%); FULL last
    """
    df = adp_pay.copy()

    dep_type_col = find_col(df.columns, "DEPOSIT TYPE", "Deposit Type")
    dep_pct_col = find_col(df.columns, "DEPOSIT PERCENT", "Deposit Percent", "DEPOSIT PERCENTAGE", "Deposit Percentage")
    dep_amt_col = find_col(df.columns, "DEPOSIT AMOUNT", "Deposit Amount")
    prio_col = find_col(df.columns, "PRIORITY #", "Priority #", "PRIORITY")

    df["Payment Method"] = "Direct Deposit"
    df["Paycheck Distribution"] = ""
    df["Paycheck Percentage"] = ""
    df["Paycheck Amount"] = ""
    df["Priority"] = ""

    if dep_type_col is None:
        return df

    df["_row_ord"] = np.arange(len(df))

    for emp, g in df.groupby(emp_col, sort=False):
        idxs = list(g.index)

        cats = []
        orig_prios = []
        for i in idxs:
            dt = str(norm_blank(df.at[i, dep_type_col])).strip().casefold()
            if ("partial" in dt and "%" in dt) or "partial%" in dt:
                cat = "partial_pct"
            elif "partial" in dt:
                cat = "partial_amt"
            elif "full" in dt:
                cat = "full"
            else:
                cat = "other"
            cats.append(cat)

            p = np.nan
            if prio_col is not None:
                p = _safe_float(df.at[i, prio_col])
            orig_prios.append(p)

        has_partial_pct = any(c == "partial_pct" for c in cats)
        sum_partial_pct = 0.0
        if has_partial_pct and dep_pct_col is not None:
            for i, c in zip(idxs, cats):
                if c == "partial_pct":
                    v = _safe_float(df.at[i, dep_pct_col])
                    if not np.isnan(v):
                        sum_partial_pct += v
        full_pct = max(0.0, 100.0 - sum_partial_pct) if has_partial_pct else 100.0

        for i, c in zip(idxs, cats):
            if c == "partial_amt":
                df.at[i, "Paycheck Distribution"] = "amount"
                df.at[i, "Paycheck Amount"] = df.at[i, dep_amt_col] if dep_amt_col is not None else ""
                df.at[i, "Paycheck Percentage"] = ""
            elif c == "partial_pct":
                df.at[i, "Paycheck Distribution"] = "percentage"
                df.at[i, "Paycheck Percentage"] = df.at[i, dep_pct_col] if dep_pct_col is not None else ""
                df.at[i, "Paycheck Amount"] = ""
            elif c == "full":
                df.at[i, "Paycheck Distribution"] = "percentage"
                df.at[i, "Paycheck Percentage"] = full_pct
                df.at[i, "Paycheck Amount"] = ""
            else:
                pct = df.at[i, dep_pct_col] if dep_pct_col is not None else ""
                amt = df.at[i, dep_amt_col] if dep_amt_col is not None else ""
                if norm_blank(pct) != "":
                    df.at[i, "Paycheck Distribution"] = "percentage"
                    df.at[i, "Paycheck Percentage"] = pct
                    df.at[i, "Paycheck Amount"] = ""
                elif norm_blank(amt) != "":
                    df.at[i, "Paycheck Distribution"] = "amount"
                    df.at[i, "Paycheck Amount"] = amt
                    df.at[i, "Paycheck Percentage"] = ""
                else:
                    df.at[i, "Paycheck Distribution"] = ""
                    df.at[i, "Paycheck Amount"] = ""
                    df.at[i, "Paycheck Percentage"] = ""

        # recompute priorities: partial_amt -> partial_pct -> full -> other
        def _sort_key(i, orig_p):
            pkey = orig_p if not np.isnan(orig_p) else 1e18
            return (pkey, int(df.at[i, "_row_ord"]))

        ordered = []
        for cat_name in ["partial_amt", "partial_pct", "full", "other"]:
            bucket = [(i, op) for i, cat, op in zip(idxs, cats, orig_prios) if cat == cat_name]
            bucket_sorted = sorted(bucket, key=lambda t: _sort_key(t[0], t[1]))
            ordered.extend([t[0] for t in bucket_sorted])

        for new_p, i in enumerate(ordered, start=1):
            df.at[i, "Priority"] = new_p

    df.drop(columns=["_row_ord"], inplace=True, errors="ignore")
    return df


def normalize_uzio_payment_full_inference(uzio_pay: pd.DataFrame, emp_col: str) -> pd.DataFrame:
    """
    If employee has multiple payment rows and exactly one row has BOTH
    Paycheck Amount blank AND Paycheck Percentage blank:
      - treat it as FULL/remainder row
      - set Paycheck Distribution to 'Percentage'
      - set Paycheck Percentage = 100 - sum(other partial% rows) (else 100)
    """
    df = uzio_pay.copy()

    dist_col = find_col(df.columns, "Paycheck Distribution", "Deposit Type")
    pct_col = find_col(df.columns, "Paycheck Percentage", "Deposit Percent")
    amt_col = find_col(df.columns, "Paycheck Amount", "Deposit Amount")

    if dist_col is None or pct_col is None or amt_col is None:
        return df

    for emp, g in df.groupby(emp_col, sort=False):
        if len(g) < 2:
            continue

        # candidate rows where both amount and percent are blank
        candidate_idxs = []
        for i in g.index:
            if _is_blank_money_or_percent(df.at[i, amt_col]) and _is_blank_money_or_percent(df.at[i, pct_col]):
                candidate_idxs.append(i)

        if len(candidate_idxs) != 1:
            continue

        full_idx = candidate_idxs[0]

        # compute sum of partial percentages (rows with pct not blank)
        sum_partial_pct = 0.0
        for i in g.index:
            if i == full_idx:
                continue
            v = _safe_float(df.at[i, pct_col])
            if not np.isnan(v):
                sum_partial_pct += v

        full_pct = max(0.0, 100.0 - sum_partial_pct) if sum_partial_pct > 0 else 100.0

        # update inferred full row
        df.at[full_idx, dist_col] = "Percentage"
        df.at[full_idx, pct_col] = full_pct

    return df


# ---------- Record key builders ----------
def build_payment_base_key(df: pd.DataFrame, emp_col: str):
    routing_col = find_col(df.columns, "ROUTING NUMBER", "Routing Number")
    acct_col = find_col(df.columns, "ACCOUNT NUMBER", "Account Number")
    dep_type_col = find_col(df.columns, "DEPOSIT TYPE", "Deposit Type")
    dep_amt_col = find_col(df.columns, "DEPOSIT AMOUNT", "Deposit Amount")
    dep_pct_col = find_col(df.columns, "DEPOSIT PERCENT", "Deposit Percent", "DEPOSIT PERCENTAGE", "Deposit Percentage")

    def _row_key(r):
        emp = norm_key_series(pd.Series([r.get(emp_col, "")])).iloc[0]
        routing = digits_only_padded(r.get(routing_col, ""), 9) if routing_col else ""
        acct = digits_only(r.get(acct_col, "")) if acct_col else ""

        dep_type = norm_value(r.get(dep_type_col, ""), "Deposit Type") if dep_type_col else ""
        amt = norm_value(r.get(dep_amt_col, ""), "Deposit Amount") if dep_amt_col else ""
        pct = norm_value(r.get(dep_pct_col, ""), "Deposit Percent") if dep_pct_col else ""

        if routing != "" or acct != "":
            return f"{emp}|BANK|{routing}|{acct}"
        return f"{emp}|NOBANK|{dep_type}|{amt}|{pct}"

    return df.apply(_row_key, axis=1)


def build_contact_base_key(df: pd.DataFrame, emp_col: str):
    name_col = find_col(df.columns, "Contact Name", "NAME", "Name")
    phone_col = find_col(df.columns, "Mobile Phone", "Phone", "MOBILE PHONE")
    rel_col = find_col(df.columns, "Relationship Description", "Relationship")

    def _row_key(r):
        emp = norm_key_series(pd.Series([r.get(emp_col, "")])).iloc[0]
        nm = normalize_person_name(r.get(name_col, "")) if name_col else ""
        ph = norm_phone_digits(r.get(phone_col, "")) if phone_col else ""
        rl = norm_value(r.get(rel_col, ""), "Relationship Description") if rel_col else ""
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


def group_indices(df: pd.DataFrame):
    g = {}
    for idx, k in df["_base_key"].items():
        k2 = "" if norm_blank(k) == "" else str(k)
        g.setdefault(k2, []).append(idx)
    return g


def read_mapping_sheet(xls: pd.ExcelFile, sheet_name: str, adp_all_cols: list) -> pd.DataFrame:
    m = pd.read_excel(xls, sheet_name=sheet_name, dtype=object)
    m.columns = [norm_colname(c) for c in m.columns]

    uz_col_name = None
    adp_col_name = None
    for c in m.columns:
        if norm_colname(c).casefold() in {"uzio coloumn", "uzio column"}:
            uz_col_name = c
        if norm_colname(c).casefold() in {"adp coloumn", "adp column"}:
            adp_col_name = c
    if uz_col_name is None or adp_col_name is None:
        raise ValueError(f"'{sheet_name}' must contain columns: 'UZIO Column' and 'ADP Column'.")

    m[uz_col_name] = m[uz_col_name].map(norm_colname)
    m[adp_col_name] = m[adp_col_name].map(norm_colname)

    m = m.dropna(subset=[uz_col_name, adp_col_name]).copy()
    m = m[(m[uz_col_name] != "") & (m[adp_col_name] != "")]
    m = m.drop_duplicates(subset=[uz_col_name], keep="first").copy()

    m["UZIO_Column"] = m[uz_col_name]
    m["ADP_Label"] = m[adp_col_name]
    m["ADP_Resolved_Column"] = m["ADP_Label"].map(lambda x: resolve_adp_col_label(x, adp_all_cols))

    # exclude Employee ID row from comparisons (it is only key)
    m["_uz_norm"] = m["UZIO_Column"].map(lambda x: norm_colname(x).casefold())
    m = m[m["_uz_norm"] != "employee id"].copy()
    m.drop(columns=["_uz_norm"], inplace=True)

    return m


# ---------- Field Summary cleanup ----------
def drop_unwanted_field_summary_columns(field_summary_df: pd.DataFrame) -> pd.DataFrame:
    if field_summary_df is None or field_summary_df.empty:
        return field_summary_df
    cols_to_drop = ["MISSING_IN_ADP", "ADP_COLUMN_MISSING", "UZIO_COLUMN_MISSING"]
    existing = [c for c in cols_to_drop if c in field_summary_df.columns]
    if existing:
        field_summary_df = field_summary_df.drop(columns=existing)
    return field_summary_df


# ---------- Core comparison ----------
def run_comparison(file_bytes: bytes) -> dict:
    xls = pd.ExcelFile(io.BytesIO(file_bytes), engine="openpyxl")

    uzio = pd.read_excel(xls, sheet_name=UZIO_SHEET, dtype=object)
    adp_pay_raw = pd.read_excel(xls, sheet_name=ADP_PAY_SHEET, dtype=object)
    adp_ec = pd.read_excel(xls, sheet_name=ADP_EC_SHEET, dtype=object)

    uzio.columns = [norm_colname(c) for c in uzio.columns]
    adp_pay_raw.columns = [norm_colname(c) for c in adp_pay_raw.columns]
    adp_ec.columns = [norm_colname(c) for c in adp_ec.columns]

    # keys
    UZIO_KEY = find_col(uzio.columns, "Employee ID", "EmployeeID", "Employee Id")
    if UZIO_KEY is None:
        raise ValueError("UZIO key column 'Employee ID' not found in 'Uzio Data' tab.")

    ADP_PAY_KEY = find_col(adp_pay_raw.columns, "ASSOCIATE ID", "Associate ID")
    ADP_EC_KEY = find_col(adp_ec.columns, "ASSOCIATE ID", "Associate ID")
    if ADP_PAY_KEY is None:
        raise ValueError("ADP Payment Data must contain 'ASSOCIATE ID' (or 'Associate ID').")
    if ADP_EC_KEY is None:
        raise ValueError("ADP Emergency Contact Data must contain 'ASSOCIATE ID' (or 'Associate ID').")

    # normalize keys
    uzio[UZIO_KEY] = norm_key_series(uzio[UZIO_KEY])
    adp_pay_raw[ADP_PAY_KEY] = norm_key_series(adp_pay_raw[ADP_PAY_KEY])
    adp_ec[ADP_EC_KEY] = norm_key_series(adp_ec[ADP_EC_KEY])

    # mapping sheets
    adp_all_cols = list(adp_pay_raw.columns) + list(adp_ec.columns)
    pay_map = read_mapping_sheet(xls, PAY_MAP_SHEET, adp_all_cols)
    ec_map = read_mapping_sheet(xls, EC_MAP_SHEET, adp_all_cols)

    # Derived fields for payment are compared against ADP-normalized columns
    payment_derived_fields = {
        "Paycheck Distribution",
        "Paycheck Percentage",
        "Paycheck Amount",
        "Priority",
        "Payment Method",
    }
    pay_map.loc[pay_map["UZIO_Column"].isin(payment_derived_fields), "ADP_Resolved_Column"] = pay_map.loc[
        pay_map["UZIO_Column"].isin(payment_derived_fields), "UZIO_Column"
    ]

    # Normalize ADP payment table (business rule)
    adp_pay = normalize_adp_payment_table(adp_pay_raw, ADP_PAY_KEY)

    # split UZIO data
    uzio_pay = filter_section_rows(uzio, "Payment")
    uzio_ec = filter_section_rows(uzio, "Emergency Contact")

    # NEW: infer UZIO FULL row when amount+percent both blank for exactly one row
    if len(uzio_pay):
        uzio_pay = normalize_uzio_payment_full_inference(uzio_pay, UZIO_KEY)

    # base keys (support duplicates)
    uzio_pay["_base_key"] = build_payment_base_key(uzio_pay, UZIO_KEY) if len(uzio_pay) else pd.Series(dtype=str)
    adp_pay["_base_key"] = build_payment_base_key(adp_pay, ADP_PAY_KEY)

    uzio_ec["_base_key"] = build_contact_base_key(uzio_ec, UZIO_KEY) if len(uzio_ec) else pd.Series(dtype=str)
    adp_ec["_base_key"] = build_contact_base_key(adp_ec, ADP_EC_KEY)

    g_uz_pay = group_indices(uzio_pay) if len(uzio_pay) else {}
    g_ad_pay = group_indices(adp_pay)
    g_uz_ec = group_indices(uzio_ec) if len(uzio_ec) else {}
    g_ad_ec = group_indices(adp_ec)

    def build_report_for_section(
        section: str,
        uz_df: pd.DataFrame,
        ad_df: pd.DataFrame,
        g_uz: dict,
        g_ad: dict,
        emp_key_uz: str,
        emp_key_ad: str,
        mapping_df: pd.DataFrame,
    ) -> bytes:

        sec_map = mapping_df.copy()
        sec_map = sec_map[sec_map["ADP_Resolved_Column"].map(norm_blank) != ""].copy()

        base_keys = sorted(set(g_uz.keys()).union(g_ad.keys()) - {""})

        rows = []
        for bk in base_keys:
            uz_list = g_uz.get(bk, [])
            ad_list = g_ad.get(bk, [])
            pairs = max(len(uz_list), len(ad_list))

            for i in range(pairs):
                uz_idx = uz_list[i] if i < len(uz_list) else None
                ad_idx = ad_list[i] if i < len(ad_list) else None

                # employee id output
                employee_id_out = ""
                try:
                    if uz_idx is not None and emp_key_uz in uz_df.columns:
                        v = uz_df.loc[uz_idx, emp_key_uz]
                        employee_id_out = "" if v is None else str(v)
                    elif ad_idx is not None and emp_key_ad in ad_df.columns:
                        v = ad_df.loc[ad_idx, emp_key_ad]
                        employee_id_out = "" if v is None else str(v)
                except Exception:
                    employee_id_out = ""

                employee_id_out = employee_id_out.strip()
                if employee_id_out == "":
                    try:
                        employee_id_out = str(bk).split("|")[0]
                    except Exception:
                        employee_id_out = str(bk)

                for _, r in sec_map.iterrows():
                    uz_field = r["UZIO_Column"]
                    adp_col_resolved = r["ADP_Resolved_Column"]

                    uz_val = uz_df.loc[uz_idx, uz_field] if (uz_idx is not None and uz_field in uz_df.columns) else ""

                    if isinstance(adp_col_resolved, str) and adp_col_resolved.startswith("__CONST__:"):
                        adp_val = adp_col_resolved.split(":", 1)[1]
                        adp_col_missing = False
                    else:
                        adp_col_missing = (adp_col_resolved not in ad_df.columns)
                        adp_val = ad_df.loc[ad_idx, adp_col_resolved] if (ad_idx is not None and not adp_col_missing) else ""

                    if ad_idx is None and uz_idx is not None:
                        status = "MISSING_IN_ADP"
                    elif ad_idx is not None and uz_idx is None:
                        status = "MISSING_IN_UZIO"
                    elif adp_col_missing:
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
                            "Employee ID": employee_id_out,
                            "Section": section,
                            "Field": uz_field,
                            "UZIO_Value": uz_val,
                            "ADP_Value": adp_val,
                            "ADP_SourceOfTruth_Status": status,
                        }
                    )

        comparison_detail = pd.DataFrame(
            rows,
            columns=["Employee ID", "Section", "Field", "UZIO_Value", "ADP_Value", "ADP_SourceOfTruth_Status"],
        )

        # Field summary
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

        # Remove requested columns from Field_Summary_By_Status
        field_summary_by_status = drop_unwanted_field_summary_columns(field_summary_by_status)

        # Remove FieldKey helper from Comparison_Detail_AllFields output
        if "_FieldKey" in comparison_detail.columns:
            comparison_detail = comparison_detail.drop(columns=["_FieldKey"])

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
                    "Total Comparisons (field-level rows)",
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
                    int(sec_map.shape[0]),
                    int(comparison_detail.shape[0]),
                ],
            }
        )

        out = io.BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="Summary", index=False)
            field_summary_by_status.to_excel(writer, sheet_name="Field_Summary_By_Status", index=False)
            comparison_detail.to_excel(writer, sheet_name="Comparison_Detail_AllFields", index=False)

        return out.getvalue()

    payment_bytes = build_report_for_section(
        section="Payment",
        uz_df=uzio_pay,
        ad_df=adp_pay,
        g_uz=g_uz_pay,
        g_ad=g_ad_pay,
        emp_key_uz=UZIO_KEY,
        emp_key_ad=ADP_PAY_KEY,
        mapping_df=pay_map,
    )

    emergency_contact_bytes = build_report_for_section(
        section="Emergency Contact",
        uz_df=uzio_ec,
        ad_df=adp_ec,
        g_uz=g_uz_ec,
        g_ad=g_ad_ec,
        emp_key_uz=UZIO_KEY,
        emp_key_ad=ADP_EC_KEY,
        mapping_df=ec_map,
    )

    return {"payment": payment_bytes, "emergency_contact": emergency_contact_bytes}


# ---------- UI ----------
st.title(APP_TITLE)
st.write("Upload the Excel workbook (.xlsx). The tool will generate two independent reports (Payment + Emergency Contact).")

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
                type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
    except Exception as e:
        st.error(f"Failed: {e}")
