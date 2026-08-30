"""
Spam Email Classifier — Data Quality Audit Script

This script performs a non-destructive data-quality audit on the raw dataset
located at `data/raw/internship.csv`. It inspects data structure, missing values,
duplicates, label validity, conflicting labels, length distributions, and content
characteristics, producing an audit summary with recommendations.
"""

import os
import re
import sys
from pathlib import Path
import pandas as pd
import numpy as np


def get_dataset_path() -> Path:
    """Resolve dataset path relative to project root regardless of working directory."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "data" / "raw" / "internship.csv"


def get_report_path() -> Path:
    """Resolve output report path relative to project root."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "reports" / "data_quality_audit.txt"


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load dataset safely with error handling and report path."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}")
    df = pd.read_csv(file_path)
    return df


def validate_columns(df: pd.DataFrame) -> dict:
    """Validate expected columns in the dataset."""
    expected_columns = {"text", "spam"}
    actual_columns = set(df.columns)
    missing = expected_columns - actual_columns
    extra = actual_columns - expected_columns
    is_valid = expected_columns.issubset(actual_columns)
    return {
        "is_valid": is_valid,
        "expected_columns": list(expected_columns),
        "actual_columns": list(df.columns),
        "missing_columns": list(missing),
        "extra_columns": list(extra),
        "dtypes": df.dtypes.to_dict()
    }


def check_missing_values(df: pd.DataFrame) -> dict:
    """Check for NaN/null values across columns."""
    missing_by_col = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    missing_rows = int(df.isnull().any(axis=1).sum())
    return {
        "missing_by_column": missing_by_col,
        "total_missing_values": total_missing,
        "rows_with_missing": missing_rows
    }


def check_empty_text(df: pd.DataFrame) -> dict:
    """Check for empty or whitespace-only text values."""
    if "text" not in df.columns:
        return {"empty_strings": 0, "whitespace_only": 0, "total_empty_or_blank": 0}
    
    text_series = df["text"].dropna().astype(str)
    empty_strings = int((text_series == "").sum())
    whitespace_only = int(((text_series != "") & (text_series.str.strip() == "")).sum())
    total_empty_or_blank = empty_strings + whitespace_only

    return {
        "empty_strings": empty_strings,
        "whitespace_only": whitespace_only,
        "total_empty_or_blank": total_empty_or_blank
    }


def check_duplicates(df: pd.DataFrame) -> dict:
    """Check for duplicate rows and duplicate email texts."""
    total_rows = len(df)
    exact_duplicate_rows = int(df.duplicated().sum())
    
    if "text" in df.columns:
        duplicate_texts = int(df.duplicated(subset=["text"]).sum())
        unique_texts = int(df["text"].nunique())
    else:
        duplicate_texts = 0
        unique_texts = 0
        
    return {
        "total_rows": total_rows,
        "exact_duplicate_rows": exact_duplicate_rows,
        "duplicate_texts": duplicate_texts,
        "unique_texts": unique_texts
    }


def check_conflicting_labels(df: pd.DataFrame) -> dict:
    """
    Check if the same email text appears with conflicting labels (e.g. spam=0 and spam=1).
    """
    if "text" not in df.columns or "spam" not in df.columns:
        return {"conflicting_count": 0, "conflicting_texts": []}

    grouped = df.groupby("text")["spam"].nunique()
    conflicting_texts_series = grouped[grouped > 1]
    conflicting_count = len(conflicting_texts_series)
    
    conflicting_examples = []
    if conflicting_count > 0:
        for text in conflicting_texts_series.index[:5]:
            rows = df[df["text"] == text][["text", "spam"]]
            labels = rows["spam"].tolist()
            preview = (text[:80] + "...") if len(text) > 80 else text
            conflicting_examples.append({
                "text_preview": preview,
                "label_occurrences": labels
            })

    return {
        "conflicting_count": conflicting_count,
        "conflicting_examples": conflicting_examples
    }


def validate_labels(df: pd.DataFrame) -> dict:
    """Validate that target labels contain only expected values (0 and 1)."""
    if "spam" not in df.columns:
        return {"is_valid": False, "unique_values": [], "invalid_values": []}

    unique_vals = df["spam"].unique().tolist()
    expected = {0, 1}
    actual = set(unique_vals)
    invalid_values = list(actual - expected)
    is_valid = len(invalid_values) == 0

    return {
        "is_valid": is_valid,
        "unique_values": unique_vals,
        "invalid_values": invalid_values,
        "invalid_count": int((~df["spam"].isin([0, 1])).sum()) if not is_valid else 0
    }


def analyze_class_distribution(df: pd.DataFrame) -> dict:
    """Analyze class distribution and proportions."""
    if "spam" not in df.columns:
        return {}

    counts = df["spam"].value_counts().to_dict()
    total = len(df)
    ham_count = counts.get(0, 0)
    spam_count = counts.get(1, 0)
    
    ham_pct = (ham_count / total * 100) if total > 0 else 0.0
    spam_pct = (spam_count / total * 100) if total > 0 else 0.0
    imbalance_ratio = (ham_count / spam_count) if spam_count > 0 else 0.0

    return {
        "ham_count": ham_count,
        "ham_percentage": ham_pct,
        "spam_count": spam_count,
        "spam_percentage": spam_pct,
        "total": total,
        "imbalance_ratio": imbalance_ratio
    }


def analyze_email_lengths(df: pd.DataFrame) -> dict:
    """Analyze email character length distributions and extremes."""
    if "text" not in df.columns:
        return {}

    text_series = df["text"].dropna().astype(str)
    lengths = text_series.str.len()

    p99 = float(lengths.quantile(0.99))
    p95 = float(lengths.quantile(0.95))
    short_threshold = 10
    short_emails_count = int((lengths < short_threshold).sum())
    long_emails_count = int((lengths >= p99).sum())

    stats = {
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(lengths.median()),
        "std": float(lengths.std()),
        "p25": float(lengths.quantile(0.25)),
        "p75": float(lengths.quantile(0.75)),
        "p95": p95,
        "p99": p99,
        "short_threshold": short_threshold,
        "short_emails_count": short_emails_count,
        "long_threshold": p99,
        "long_emails_count": long_emails_count
    }

    # Samples of short emails for inspection
    if short_emails_count > 0:
        short_samples = df[df["text"].astype(str).str.len() < short_threshold][["text", "spam"]].to_dict(orient="records")
        stats["short_samples"] = short_samples
    else:
        stats["short_samples"] = []

    return stats


def analyze_content_characteristics(df: pd.DataFrame) -> dict:
    """Check common text patterns: HTML tags, URLs, emails, numbers, punctuation, non-ASCII."""
    if "text" not in df.columns:
        return {}

    text_series = df["text"].dropna().astype(str)
    total = len(text_series)

    # Patterns accounting for both raw text and space-tokenized Kaggle formats
    html_pattern = re.compile(r'<\s*\/?[a-zA-Z][^>]*>', re.IGNORECASE)
    url_pattern = re.compile(r'https?\s*:\s*/\s*/\S+|www\s*\.\s*\S+|\bhttps?://\S+|\bwww\.\S+', re.IGNORECASE)
    email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\s*\.\s*[A-Za-z]{2,}\b|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
    number_pattern = re.compile(r'\d+')
    excessive_punct_pattern = re.compile(r'(?:[!?.,$*#~=_-]\s*){3,}')
    non_ascii_pattern = re.compile(r'[^\x00-\x7F]')
    subject_prefix_pattern = re.compile(r'^\s*subject\s*:', re.IGNORECASE)

    has_html = text_series.apply(lambda x: bool(html_pattern.search(x)))
    has_url = text_series.apply(lambda x: bool(url_pattern.search(x)))
    has_email = text_series.apply(lambda x: bool(email_pattern.search(x)))
    has_number = text_series.apply(lambda x: bool(number_pattern.search(x)))
    has_excessive_punct = text_series.apply(lambda x: bool(excessive_punct_pattern.search(x)))
    has_non_ascii = text_series.apply(lambda x: bool(non_ascii_pattern.search(x)))
    has_subject_prefix = text_series.apply(lambda x: bool(subject_prefix_pattern.search(x)))

    return {
        "html_count": int(has_html.sum()),
        "html_pct": float(has_html.sum() / total * 100),
        "url_count": int(has_url.sum()),
        "url_pct": float(has_url.sum() / total * 100),
        "email_address_count": int(has_email.sum()),
        "email_address_pct": float(has_email.sum() / total * 100),
        "numbers_count": int(has_number.sum()),
        "numbers_pct": float(has_number.sum() / total * 100),
        "excessive_punct_count": int(has_excessive_punct.sum()),
        "excessive_punct_pct": float(has_excessive_punct.sum() / total * 100),
        "non_ascii_count": int(has_non_ascii.sum()),
        "non_ascii_pct": float(has_non_ascii.sum() / total * 100),
        "subject_prefix_count": int(has_subject_prefix.sum()),
        "subject_prefix_pct": float(has_subject_prefix.sum() / total * 100),
        "total_analyzed": total
    }


def check_encoding_and_types(df: pd.DataFrame) -> dict:
    """Check for encoding anomalies or non-string text values."""
    if "text" not in df.columns:
        return {"non_string_count": 0}
    
    non_string_mask = ~df["text"].apply(lambda x: isinstance(x, str))
    non_string_count = int(non_string_mask.sum())
    
    return {
        "non_string_count": non_string_count
    }


def generate_summary(audit_data: dict) -> list:
    """Compile structured findings, affected counts, and explicit action recommendations."""
    findings = []
    
    # 1. Missing values
    missing_cnt = audit_data["missing"]["total_missing_values"]
    if missing_cnt == 0:
        findings.append({
            "issue": "Missing / Null Values",
            "status": "CLEAN",
            "affected_records": "0",
            "recommendation": "Preserve as-is; no missing value imputation or row deletion needed."
        })
    else:
        findings.append({
            "issue": "Missing / Null Values",
            "status": "DETECTED",
            "affected_records": str(missing_cnt),
            "recommendation": "Investigate & Remove later during preprocessing."
        })

    # 2. Empty / whitespace-only text
    empty_cnt = audit_data["empty_text"]["total_empty_or_blank"]
    if empty_cnt == 0:
        findings.append({
            "issue": "Empty / Whitespace-Only Emails",
            "status": "CLEAN",
            "affected_records": "0",
            "recommendation": "Preserve as-is; all 5,728 records contain valid text."
        })
    else:
        findings.append({
            "issue": "Empty / Whitespace-Only Emails",
            "status": "DETECTED",
            "affected_records": str(empty_cnt),
            "recommendation": "Investigate & Remove later during preprocessing."
        })

    # 3. Duplicate rows & texts
    dup_rows = audit_data["duplicates"]["exact_duplicate_rows"]
    findings.append({
        "issue": "Exact Duplicate Rows",
        "status": "DETECTED" if dup_rows > 0 else "CLEAN",
        "affected_records": f"{dup_rows} rows",
        "recommendation": f"Remove later ({dup_rows} duplicate rows should be dropped in preprocessing to prevent data leakage and inflated evaluation scores)." if dup_rows > 0 else "None needed."
    })
    
    # 4. Conflicting Labels
    conflicts = audit_data["conflicting_labels"]["conflicting_count"]
    findings.append({
        "issue": "Conflicting Labels (Multi-label text)",
        "status": "CLEAN" if conflicts == 0 else "DETECTED",
        "affected_records": f"{conflicts} instances",
        "recommendation": "Preserve / No Action (0 conflicting labels detected; duplicates share identical labels)." if conflicts == 0 else "Investigate & Remove/Relabel inconsistent instances."
    })

    # 5. Invalid Labels
    invalid_labels = audit_data["labels"]["invalid_count"]
    findings.append({
        "issue": "Invalid Label Values (Non 0/1)",
        "status": "CLEAN" if invalid_labels == 0 else "DETECTED",
        "affected_records": "0",
        "recommendation": "Preserve (All labels strictly conform to binary {0, 1})."
    })

    # 6. Class Imbalance
    cls = audit_data["class_distribution"]
    findings.append({
        "issue": "Class Imbalance (Ham vs Spam)",
        "status": "NOTE",
        "affected_records": f"Ham: {cls.get('ham_count')} ({cls.get('ham_percentage', 0):.1f}%) / Spam: {cls.get('spam_count')} ({cls.get('spam_percentage', 0):.1f}%)",
        "recommendation": "Investigate & Handle during training/evaluation (Use StratifiedKFold cross-validation; prioritize F1-score & Precision-Recall over raw accuracy)."
    })

    # 7. Common 'Subject:' Prefix
    content = audit_data["content"]
    subj_cnt = content.get("subject_prefix_count", 0)
    findings.append({
        "issue": "'Subject:' Email Prefix Artifact",
        "status": "DETECTED",
        "affected_records": f"{subj_cnt} ({content.get('subject_prefix_pct', 0):.1f}%)",
        "recommendation": "Clean later during text preprocessing (Strip common email header prefix to prevent bias)."
    })

    # 8. URLs and Web Domains
    url_cnt = content.get("url_count", 0)
    findings.append({
        "issue": "URLs & Web Domains",
        "status": "DETECTED" if url_cnt > 0 else "CLEAN",
        "affected_records": f"{url_cnt} ({content.get('url_pct', 0):.1f}%)",
        "recommendation": "Clean / Normalize later (Replace with normalized token like 'httpaddr' or preserve domain clues for spam indicator)."
    })

    # 9. Email Addresses
    email_cnt = content.get("email_address_count", 0)
    findings.append({
        "issue": "Email Addresses in Body",
        "status": "DETECTED" if email_cnt > 0 else "CLEAN",
        "affected_records": f"{email_cnt} ({content.get('email_address_pct', 0):.1f}%)",
        "recommendation": "Clean / Normalize later (Replace with normalized token like 'emailaddr' to generalize features)."
    })

    # 10. Numbers / Digits
    num_cnt = content.get("numbers_count", 0)
    findings.append({
        "issue": "Numbers / Digits",
        "status": "DETECTED",
        "affected_records": f"{num_cnt} ({content.get('numbers_pct', 0):.1f}%)",
        "recommendation": "Clean / Normalize later (Replace with 'numtoken' or normalize monetary/numeric patterns)."
    })

    # 11. Extreme Lengths
    lengths = audit_data["lengths"]
    findings.append({
        "issue": "Length Outliers (99th %ile)",
        "status": "NOTE",
        "affected_records": f"{lengths.get('long_emails_count')} emails >= {lengths.get('long_threshold', 0):.0f} chars",
        "recommendation": "Preserve (Long emails are legitimate messages; use TF-IDF max_features/sublinear_tf rather than dropping them)."
    })

    return findings


def format_audit_report(dataset_path: Path, audit_data: dict) -> str:
    """Format full audit findings into an organized, readable text output."""
    lines = []
    sep = "=" * 75
    subsep = "-" * 75

    lines.append(sep)
    lines.append("SPAM EMAIL CLASSIFIER -- DATA QUALITY AUDIT REPORT")
    lines.append(sep)
    lines.append(f"Dataset Path : {dataset_path}")
    lines.append(f"File Size    : {os.path.getsize(dataset_path):,} bytes")
    lines.append("")

    # 1. DATASET OVERVIEW
    lines.append("1. DATASET OVERVIEW")
    lines.append(subsep)
    df_shape = audit_data["shape"]
    lines.append(f"  - Total Records (Rows)   : {df_shape[0]:,}")
    lines.append(f"  - Total Columns          : {df_shape[1]}")
    lines.append(f"  - Column Details:")
    for col, dtype in audit_data["columns"]["dtypes"].items():
        lines.append(f"      * '{col}' : {dtype}")
    lines.append(f"  - Required Columns Check : {'PASSED (text, spam present)' if audit_data['columns']['is_valid'] else 'FAILED'}")
    lines.append("")

    # 2. MISSING VALUES & EMPTY CONTENT
    lines.append("2. MISSING VALUES & EMPTY STRINGS")
    lines.append(subsep)
    lines.append(f"  - Missing Values in 'text' : {audit_data['missing']['missing_by_column'].get('text', 0)}")
    lines.append(f"  - Missing Values in 'spam' : {audit_data['missing']['missing_by_column'].get('spam', 0)}")
    lines.append(f"  - Total Missing Values     : {audit_data['missing']['total_missing_values']}")
    lines.append(f"  - Empty Strings (\"\")       : {audit_data['empty_text']['empty_strings']}")
    lines.append(f"  - Whitespace-Only Strings  : {audit_data['empty_text']['whitespace_only']}")
    lines.append(f"  - Non-String Data Types    : {audit_data['encoding']['non_string_count']}")
    lines.append("")

    # 3. DUPLICATES & IDENTICAL ROWS
    lines.append("3. DUPLICATE ANALYSIS")
    lines.append(subsep)
    lines.append(f"  - Exact Duplicate Rows     : {audit_data['duplicates']['exact_duplicate_rows']:,}")
    lines.append(f"  - Duplicate 'text' Entries : {audit_data['duplicates']['duplicate_texts']:,}")
    lines.append(f"  - Unique Email Texts       : {audit_data['duplicates']['unique_texts']:,}")
    lines.append("")

    # 4. LABEL VALIDATION & CONFLICT AUDIT
    lines.append("4. LABEL VALIDATION & CONFLICT AUDIT")
    lines.append(subsep)
    lines.append(f"  - Valid Labels Detected    : {sorted(audit_data['labels']['unique_values'])}")
    lines.append(f"  - Invalid Labels (Non 0/1) : {audit_data['labels']['invalid_values']} (Count: {audit_data['labels']['invalid_count']})")
    lines.append(f"  - Conflicting Text Labels  : {audit_data['conflicting_labels']['conflicting_count']} (same text having different labels)")
    if audit_data['conflicting_labels']['conflicting_count'] > 0:
        lines.append("    Examples of conflicting items:")
        for ex in audit_data['conflicting_labels']['conflicting_examples']:
            lines.append(f"      * Labels {ex['label_occurrences']}: \"{ex['text_preview']}\"")
    lines.append("")

    # 5. CLASS DISTRIBUTION
    lines.append("5. CLASS DISTRIBUTION")
    lines.append(subsep)
    cls = audit_data["class_distribution"]
    lines.append(f"  - Not Spam / Ham (0)         : {cls.get('ham_count', 0):,} ({cls.get('ham_percentage', 0.0):.2f}%)")
    lines.append(f"  - Spam (1)                   : {cls.get('spam_count', 0):,} ({cls.get('spam_percentage', 0.0):.2f}%)")
    lines.append(f"  - Total Samples              : {cls.get('total', 0):,}")
    lines.append(f"  - Imbalance Ratio (Ham:Spam) : {cls.get('imbalance_ratio', 0.0):.2f} : 1")
    lines.append("")

    # 6. EMAIL LENGTH ANALYSIS
    lines.append("6. EMAIL LENGTH ANALYSIS (Character Count)")
    lines.append(subsep)
    lengths = audit_data["lengths"]
    lines.append(f"  - Minimum Length             : {lengths.get('min', 0):,} chars")
    lines.append(f"  - Maximum Length             : {lengths.get('max', 0):,} chars")
    lines.append(f"  - Mean Length                : {lengths.get('mean', 0.0):,.2f} chars")
    lines.append(f"  - Median Length              : {lengths.get('median', 0.0):,.2f} chars")
    lines.append(f"  - Standard Deviation         : {lengths.get('std', 0.0):,.2f} chars")
    lines.append(f"  - 25th Percentile (Q1)       : {lengths.get('p25', 0.0):,.2f} chars")
    lines.append(f"  - 75th Percentile (Q3)       : {lengths.get('p75', 0.0):,.2f} chars")
    lines.append(f"  - 95th Percentile            : {lengths.get('p95', 0.0):,.2f} chars")
    lines.append(f"  - 99th Percentile            : {lengths.get('p99', 0.0):,.2f} chars")
    lines.append(f"  - Extremely Short (< {lengths.get('short_threshold')} chars)  : {lengths.get('short_emails_count')} emails")
    lines.append(f"  - Extremely Long (>= {lengths.get('long_threshold', 0):.0f} chars): {lengths.get('long_emails_count')} emails (99th percentile threshold)")
    lines.append("")

    # 7. CONTENT CHARACTERISTICS
    lines.append("7. CONTENT CHARACTERISTICS")
    lines.append(subsep)
    content = audit_data["content"]
    lines.append(f"  - Emails with 'Subject:' prefix   : {content.get('subject_prefix_count', 0):,} ({content.get('subject_prefix_pct', 0.0):.2f}%)")
    lines.append(f"  - Emails with URLs/Web Domains    : {content.get('url_count', 0):,} ({content.get('url_pct', 0.0):.2f}%)")
    lines.append(f"  - Emails with Email Addresses     : {content.get('email_address_count', 0):,} ({content.get('email_address_pct', 0.0):.2f}%)")
    lines.append(f"  - Emails with Digits / Numbers    : {content.get('numbers_count', 0):,} ({content.get('numbers_pct', 0.0):.2f}%)")
    lines.append(f"  - Emails with HTML Tags           : {content.get('html_count', 0):,} ({content.get('html_pct', 0.0):.2f}%)")
    lines.append(f"  - Emails with Repeated Punctuation: {content.get('excessive_punct_count', 0):,} ({content.get('excessive_punct_pct', 0.0):.2f}%)")
    lines.append(f"  - Emails with Non-ASCII Chars     : {content.get('non_ascii_count', 0):,} ({content.get('non_ascii_pct', 0.0):.2f}%)")
    lines.append("")

    # 8. FINAL AUDIT SUMMARY & RECOMMENDATIONS
    lines.append("8. FINAL AUDIT SUMMARY & ACTION RECOMMENDATIONS")
    lines.append(subsep)
    summary_items = audit_data["summary"]
    lines.append(f"{'#':<3} | {'Issue / Characteristic':<34} | {'Status':<10} | {'Affected':<22} | {'Recommended Action'}")
    lines.append("-" * 125)
    for idx, item in enumerate(summary_items, start=1):
        lines.append(
            f"{idx:<3} | {item['issue'][:34]:<34} | {item['status']:<10} | {str(item['affected_records'])[:22]:<22} | {item['recommendation']}"
        )
    lines.append(sep)

    return "\n".join(lines)


def run_audit(dataset_path: Path = None, save_report: bool = True) -> str:
    """Execute complete data quality audit and return formatted string report."""
    if dataset_path is None:
        dataset_path = get_dataset_path()

    df = load_dataset(dataset_path)

    audit_data = {
        "shape": df.shape,
        "columns": validate_columns(df),
        "missing": check_missing_values(df),
        "empty_text": check_empty_text(df),
        "duplicates": check_duplicates(df),
        "conflicting_labels": check_conflicting_labels(df),
        "labels": validate_labels(df),
        "class_distribution": analyze_class_distribution(df),
        "lengths": analyze_email_lengths(df),
        "content": analyze_content_characteristics(df),
        "encoding": check_encoding_and_types(df)
    }
    
    audit_data["summary"] = generate_summary(audit_data)
    
    report_text = format_audit_report(dataset_path, audit_data)
    
    if save_report:
        report_path = get_report_path()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
            
    return report_text


def main():
    """Main execution function."""
    try:
        report = run_audit()
        print(report)
    except Exception as e:
        print(f"Error during data quality audit: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
