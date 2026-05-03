"""
preprocessor.py  (v2 — finance-aware)
--------------------------------------
Text preprocessing pipeline for the FinSearch IR system.

Pipeline (applied identically to documents AND queries):
  0. Finance Normalization  ← NEW in v2: domain-specific term fusion
  1. Lowercase
  2. Tokenize               — keep only alphanumeric tokens
  3. Stop word removal
  4. Porter Stemming

What changed from v1
--------------------
v1: general-purpose pipeline only.
v2: adds a finance normalization layer (Step 0) applied BEFORE tokenization.

Why this matters
----------------
The regex tokenizer r"[a-z0-9]+" strips all non-alphanumeric characters,
which silently breaks financial terms in two ways:

  Problem 1 — Variant splitting:
    "401(k)"  →  ['401', 'k']   ← parens stripped, 'k' is noise
    "401k"    →  ['401k']       ← already fused, preserved correctly
    Result: same concept produces different tokens → no BM25 match

  Problem 2 — Multi-word term ambiguity:
    "mutual fund"  →  ['mutual', 'fund']
    "index fund"   →  ['index', 'fund']
    'fund' alone matches ANY document containing "fund",
    inflating BM25 scores for unrelated documents.

  Solution: fuse known financial terms into single tokens BEFORE tokenizing:
    "401(k)"      →  "401k"       →  ['401k']
    "mutual fund" →  "mutualfund" →  ['mutualfund']

All normalizations were derived from scanning the actual 5,000-document
dataset and 25 queries — only terms confirmed present are included.

Requires:
    pip install nltk
"""

import re
from nltk.stem import PorterStemmer


# ── Step 0: Finance-specific normalization rules ──────────────────────────────
# Format: (regex_pattern, replacement_string)
# Applied in ORDER — more specific patterns (e.g. "capital gains tax")
# appear BEFORE more general ones (e.g. "capital gains") to avoid
# partial replacements.
#
# Dataset evidence (occurrences across 5,000 docs + 25 queries):
#   401k variants : 345   interest rate  : 330   credit card  : 325
#   capital gains : 119   mutual fund    : 155   student loan : 111
#   stock market  : 139   cash flow      : 104   index fund   :  78
#   credit score  :  71   S&P 500        :  88   short sell   :  34

FINANCE_NORM = [
    # ── Retirement accounts ───────────────────────────────────────────────────
    # 401(k) / 401K / 401 (k) → 401k  [345 occurrences, 2 variants]
    (r"401\s*\(\s*k\s*\)",          "401k"),
    # 403(b) → 403b
    (r"403\s*\(\s*b\s*\)",          "403b"),
    # Roth IRA / ROTH IRA → rothira  [90 occurrences]
    (r"roth\s+ira",                 "rothira"),
    # SEP-IRA / SEP IRA → sepira  [8 occurrences]
    (r"sep\s*-?\s*ira",             "sepira"),

    # ── Settlement periods ────────────────────────────────────────────────────
    # T+1 / T+2 / T+3 → tplus1 / tplus2 / tplus3  [11 occurrences]
    # Without this: "T+2" → ['2'] — the 'T' is lost as stopword 't'
    (r"\bt\s*\+\s*(\d)\b",          r"tplus\1"),

    # ── Market indices ────────────────────────────────────────────────────────
    # S&P 500 / S&P → sp500  [88 occurrences]
    # Without this: "S&P" → [] — both letters become stopwords
    (r"s\s*&\s*p\s*500",            "sp500"),
    (r"s\s*&\s*p\b",                "sp500"),

    # ── Hyphenated time terms ─────────────────────────────────────────────────
    # 30-year / 15-year → 30year / 15year  [71 occurrences]
    # Preserves the number+unit relationship after tokenization
    (r"(\d+)\s*-\s*year",           r"\1year"),
    (r"(\d+)\s*-\s*month",          r"\1month"),
    (r"(\d+)\s*-\s*week",           r"\1week"),

    # ── High-frequency multi-word terms ──────────────────────────────────────
    # Note: "capital gains tax" must come BEFORE "capital gains"
    (r"interest\s+rates?",          "interestrate"),      # 330 occurrences
    (r"credit\s+cards?",            "creditcard"),        # 325 occurrences
    (r"student\s+loans?",           "studentloan"),       # 111 occurrences
    (r"capital\s+gains?\s+tax",     "capitalgainstax"),   # 26  (specific first)
    (r"capital\s+gains?",           "capitalgain"),       # 119 occurrences
    (r"mutual\s+funds?",            "mutualfund"),        # 155 occurrences
    (r"stock\s+market",             "stockmarket"),       # 139 occurrences
    (r"index\s+funds?",             "indexfund"),         # 78  occurrences
    (r"cash\s+flows?",              "cashflow"),          # 104 occurrences
    (r"credit\s+scores?",           "creditscore"),       # 71  occurrences
    (r"short\s+sell(?:ing)?",       "shortsell"),         # 34  occurrences
    (r"hedge\s+funds?",             "hedgefund"),         # 27  occurrences
    (r"net\s+worth",                "networth"),          # 33  occurrences
    (r"home\s+equity",              "homeequity"),        # 14  occurrences
    (r"p/e\s*ratio",                "peratio"),           # 5   occurrences
]

# ── Bundled English stop words ────────────────────────────────────────────────
# Sourced from NLTK English stopwords — bundled to avoid nltk.download().
STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "she's", "her",
    "hers", "herself", "it", "it's", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom",
    "this", "that", "that'll", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do",
    "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
    "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "don't", "should", "should've", "now", "d",
    "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn",
    "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't",
    "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma", "mightn",
    "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't",
    "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won",
    "won't", "wouldn", "wouldn't",
}

# ── Stemmer singleton ─────────────────────────────────────────────────────────
_stemmer = PorterStemmer()


# ── Pipeline functions ────────────────────────────────────────────────────────

def normalize_finance(text: str) -> str:
    """
    Step 0 — Domain normalization (applied BEFORE tokenization).

    Fuses financial term variants into single canonical tokens so that
    surface-form differences do not prevent BM25 matching.

    Parameters
    ----------
    text : str  Raw input text (document or query).

    Returns
    -------
    str         Text with financial terms normalized.

    Examples
    --------
    >>> normalize_finance("Max out your 401(k) contributions")
    'Max out your 401k contributions'

    >>> normalize_finance("Roth IRA vs traditional IRA")
    'rothira vs traditional IRA'

    >>> normalize_finance("Compare mutual funds vs index funds")
    'Compare mutualfund vs indexfund'

    >>> normalize_finance("S&P 500 performance")
    'sp500 performance'

    >>> normalize_finance("T+2 settlement period")
    'tplus2 settlement period'
    """
    for pattern, replacement in FINANCE_NORM:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def tokenize(text: str) -> list[str]:
    """
    Step 1+2 — Lowercase and split into alphanumeric tokens.

    Strips punctuation, symbols, and standalone characters.

    Examples
    --------
    >>> tokenize("401k contributions after normalize_finance")
    ['401k', 'contributions', 'after', 'normalize_finance']

    >>> tokenize("15year mortgage vs 30year loan")
    ['15year', 'mortgage', 'vs', '30year', 'loan']
    """
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Step 3 — Remove common English stop words.

    Example
    -------
    >>> remove_stopwords(['buy', 'from', 'an', 'aggressive', 'salesperson'])
    ['buy', 'aggressive', 'salesperson']
    """
    return [t for t in tokens if t not in STOP_WORDS]


def stem(tokens: list[str]) -> list[str]:
    """
    Step 4 — Apply Porter Stemming to reduce tokens to root forms.

    Example
    -------
    >>> stem(['buying', 'contributions', 'investing'])
    ['buy', 'contribut', 'invest']
    """
    return [_stemmer.stem(t) for t in tokens]


def preprocess(text: str) -> list[str]:
    """
    Full preprocessing pipeline:
        normalize_finance → tokenize → remove_stopwords → stem

    This is the main entry point used by the indexer and query processor.
    Applies identically to both documents and queries.

    Parameters
    ----------
    text : str  Raw document or query text.

    Returns
    -------
    list[str]   Cleaned, stemmed tokens ready for indexing or BM25 scoring.

    Examples
    --------
    >>> preprocess("Should you always max out contributions to your 401k?")
    ['alway', 'max', 'contribut', '401k']

    >>> preprocess("Max out your 401(k) contributions")
    ['max', 'contribut', '401k']

    >>> preprocess("Why does short selling require borrowing?")
    ['shortsel', 'requir', 'borrow']

    >>> preprocess("Paying Off Principal of Home vs. Investing In Mutual Fund")
    ['pay', 'princip', 'home', 'vs', 'invest', 'mutualfund']

    >>> preprocess("Is it ever a good idea to close credit cards?")
    ['ever', 'good', 'idea', 'close', 'creditcard']

    >>> preprocess("")   # edge case: empty string
    []
    """
    if not text or not text.strip():
        return []

    text   = normalize_finance(text)   # Step 0 — finance normalization (NEW)
    tokens = tokenize(text)            # Step 1+2 — lowercase + tokenize
    tokens = remove_stopwords(tokens)  # Step 3 — stop word removal
    tokens = stem(tokens)              # Step 4 — Porter Stemming
    return tokens


# ── Sanity check when run directly ───────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 65)
    print("STEP-BY-STEP PIPELINE TRACE")
    print("=" * 65)

    trace_cases = [
        "Max out your 401(k) contributions",
        "Paying Off Principal of Home vs. Investing In Mutual Fund",
        "Why does short selling require borrowing?",
        "Is it ever a good idea to close credit cards?",
        "To pay off a student loan, save lump sum or pay extra each month?",
        "As a 22-year-old, how risky should my 401(k) investments be?",
        "Why does Charles Schwab have a T+2 settlement period?",
        "S&P 500 index fund vs mutual fund returns",
        "capital gains tax on short selling stocks",
        "",       # edge: empty
        "!!! ???" # edge: symbols only
    ]

    for text in trace_cases:
        normed  = normalize_finance(text)
        toks    = tokenize(normed)
        no_sw   = remove_stopwords(toks)
        stemmed = stem(no_sw)
        display = (text[:50] + "...") if len(text) > 53 else text
        print(f"\n  Input    : {display!r}")
        if normed.lower() != text.lower():
            print(f"  Normed   : {normed!r}")
        print(f"  Tokens   : {toks}")
        print(f"  -StopWds : {no_sw}")
        print(f"  Stemmed  : {stemmed}")

    print("\n" + "=" * 65)
    print("QUERY vs DOCUMENT OVERLAP — KEY FIXES")
    print("=" * 65)

    overlap_tests = [
        ("401k query",
         "Should you always max out your 401k",
         "Max out your 401(k) contributions"),
        ("mutual fund query",
         "Paying Off Principal vs Investing In Mutual Fund",
         "mutual funds vs index funds comparison"),
        ("short selling query",
         "Why does short selling require borrowing",
         "short sell mechanics and borrowing costs"),
        ("student loan query",
         "pay off student loan lump sum or monthly",
         "student loans repayment strategies"),
        ("credit card query",
         "Pay off credit card debt or earn 401k match",
         "credit cards debt repayment priority"),
    ]

    for label, q_text, d_text in overlap_tests:
        q_v1 = set(stem(remove_stopwords(tokenize(q_text))))
        d_v1 = set(stem(remove_stopwords(tokenize(d_text))))
        q_v2 = set(preprocess(q_text))
        d_v2 = set(preprocess(d_text))

        overlap_v1 = q_v1 & d_v1
        overlap_v2 = q_v2 & d_v2
        gained     = overlap_v2 - overlap_v1

        print(f"\n  [{label}]")
        print(f"  Query : {q_text!r}")
        print(f"  Doc   : {d_text!r}")
        print(f"  V1 overlap : {sorted(overlap_v1)}")
        print(f"  V2 overlap : {sorted(overlap_v2)}")
        if gained:
            print(f"  ✅ Gained  : {sorted(gained)}")
        else:
            print(f"  → No change needed")