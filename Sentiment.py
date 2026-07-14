"""
LSEG News Sentiment Analysis — Streamlit Application.

Retrieves news headlines and full stories from LSEG Workspace via the
LSEG Data Library for Python (`lseg.data`), analyzes sentiment with
TextBlob, and presents interactive metrics, charts, and tables.

Prerequisites:
    - LSEG Workspace desktop application running and logged in.
    - `lseg-data` installed and (optionally) an app key configured in
      `.streamlit/secrets.toml` (see `.streamlit/secrets.toml.example`).

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from bs4 import BeautifulSoup
from textblob import TextBlob

try:
    import lseg.data as ld
except ImportError:  # pragma: no cover - handled gracefully in the UI
    ld = None  # type: ignore[assignment]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lseg_sentiment_app")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STORY_REQUEST_DELAY_SECONDS: float = 0.35
HEADLINES_CACHE_TTL_SECONDS: int = 600  # 10 minutes
MAX_HEADLINES_LIMIT: int = 100
SENTIMENT_ORDER: list[str] = ["Positive", "Neutral", "Negative"]
SENTIMENT_COLORS: dict[str, str] = {
    "Positive": "#2E8B57",
    "Neutral": "#8C8C8C",
    "Negative": "#C0392B",
}


# ---------------------------------------------------------------------------
# LSEG session and data retrieval
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Opening LSEG Workspace session...")
def get_lseg_session() -> Any:
    """
    Open (and cache) a single LSEG Workspace desktop session.

    The session is cached with ``st.cache_resource`` so it is created once
    per Streamlit server process and reused across reruns and users.

    An app key is read from Streamlit secrets if present
    (``[lseg] app_key = "..."``); otherwise the library falls back to its
    standard configuration (e.g. ``lseg-data.config.json``).

    Returns:
        The opened LSEG session object.

    Raises:
        RuntimeError: If the ``lseg-data`` package is not installed or the
            session cannot be opened (e.g. Workspace is not running).
    """
    if ld is None:
        raise RuntimeError(
            "The 'lseg-data' package is not installed. "
            "Install dependencies with: pip install -r requirements.txt"
        )

    app_key: Optional[str] = None
    try:
        app_key = st.secrets.get("lseg", {}).get("app_key")  # type: ignore[union-attr]
    except Exception:
        # No secrets file present — that is fine for a desktop session.
        app_key = None

    try:
        if app_key:
            session = ld.session.desktop.Definition(app_key=app_key).get_session()
            session.open()
            ld.session.set_default(session)
        else:
            session = ld.open_session()
    except Exception as exc:
        raise RuntimeError(
            "Could not open an LSEG Workspace session. Make sure the LSEG "
            "Workspace desktop application is running and you are logged in. "
            f"Underlying error: {exc}"
        ) from exc

    return session


@st.cache_data(ttl=HEADLINES_CACHE_TTL_SECONDS, show_spinner="Fetching headlines...")
def fetch_headlines(
    query: str,
    start_date_iso: str,
    end_date_iso: str,
    max_headlines: int,
) -> pd.DataFrame:
    """
    Retrieve news headlines from LSEG for a query and date range.

    Results are cached with ``st.cache_data`` (TTL: 10 minutes) keyed by the
    function arguments, so identical requests within the TTL do not hit the
    LSEG API again.

    Args:
        query: LSEG news query string (e.g. ``'"Goldman Sachs" and Language:LEN'``).
        start_date_iso: Inclusive start date, ISO format ``YYYY-MM-DD``.
        end_date_iso: Inclusive end date, ISO format ``YYYY-MM-DD``.
        max_headlines: Maximum number of headlines to retrieve (1–100).

    Returns:
        A DataFrame with (at minimum) columns ``headline``, ``storyId`` and
        ``versionCreated``. May be empty if no news is found.

    Raises:
        RuntimeError: If the LSEG API call fails.
    """
    # Ensure a session exists (cached, so effectively a no-op after first call).
    get_lseg_session()

    try:
        raw = ld.news.get_headlines(
            query,
            start=start_date_iso,
            end=end_date_iso,
            count=max_headlines,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to retrieve headlines from LSEG: {exc}") from exc

    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["headline", "storyId", "versionCreated"])

    df = raw.copy()

    # The publication timestamp is usually the index ("versionCreated").
    if "versionCreated" not in df.columns:
        df = df.reset_index()
        if "versionCreated" not in df.columns and len(df.columns) > 0:
            # Fall back: assume the former index column holds the timestamp.
            df = df.rename(columns={df.columns[0]: "versionCreated"})
    else:
        df = df.reset_index(drop=True)

    # Normalize expected column names defensively.
    if "headline" not in df.columns:
        for candidate in ("text", "title", "Headline"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "headline"})
                break

    required = {"headline", "storyId"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Unexpected headlines response from LSEG; missing columns: {sorted(missing)}"
        )

    df["versionCreated"] = pd.to_datetime(df["versionCreated"], errors="coerce", utc=True)
    return df[["headline", "storyId", "versionCreated"]]


def fetch_story(story_id: str) -> str:
    """
    Retrieve the full news story (HTML) for a given story ID.

    Args:
        story_id: The LSEG story identifier from a headlines result.

    Returns:
        The raw story content (typically HTML).

    Raises:
        RuntimeError: If the story cannot be retrieved.
    """
    try:
        story = ld.news.get_story(story_id)
    except Exception as exc:
        raise RuntimeError(f"get_story failed for '{story_id}': {exc}") from exc

    if story is None:
        raise RuntimeError(f"get_story returned no content for '{story_id}'.")

    return str(story)


# ---------------------------------------------------------------------------
# Text processing and sentiment
# ---------------------------------------------------------------------------

def clean_html(raw_html: str) -> str:
    """
    Strip HTML tags and normalize whitespace from article content.

    Args:
        raw_html: Raw (possibly HTML) article content.

    Returns:
        Plain text with tags removed and whitespace collapsed.
    """
    if not raw_html:
        return ""
    try:
        soup = BeautifulSoup(raw_html, "lxml")
        text = soup.get_text(separator=" ")
    except Exception:
        # If parsing fails for any reason, fall back to the raw string.
        text = raw_html
    return " ".join(text.split()).strip()


def classify_sentiment(
    polarity: float,
    positive_threshold: float,
    negative_threshold: float,
) -> str:
    """
    Map a polarity score to a sentiment label using user thresholds.

    Args:
        polarity: TextBlob polarity in [-1.0, 1.0].
        positive_threshold: Polarity at or above which text is Positive.
        negative_threshold: Polarity at or below which text is Negative.

    Returns:
        One of ``"Positive"``, ``"Neutral"``, ``"Negative"``.
    """
    if polarity >= positive_threshold:
        return "Positive"
    if polarity <= negative_threshold:
        return "Negative"
    return "Neutral"


def analyze_text(
    text: str,
    positive_threshold: float,
    negative_threshold: float,
) -> tuple[float, float, str]:
    """
    Compute TextBlob polarity, subjectivity, and a sentiment label.

    Args:
        text: The plain text to analyze.
        positive_threshold: Polarity threshold for the Positive label.
        negative_threshold: Polarity threshold for the Negative label.

    Returns:
        Tuple ``(polarity, subjectivity, sentiment_label)``.
    """
    blob = TextBlob(text)
    polarity = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)
    label = classify_sentiment(polarity, positive_threshold, negative_threshold)
    return polarity, subjectivity, label


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------

def run_sentiment_analysis(
    headlines_df: pd.DataFrame,
    positive_threshold: float,
    negative_threshold: float,
    progress_callback: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Fetch full stories for each headline and compute sentiment metrics.

    For each headline: the full story is retrieved via ``ld.news.get_story``
    and cleaned; if retrieval fails, the headline text is used as a fallback
    and the error is recorded — a single failure never aborts the run.
    A small delay is inserted between story requests to avoid API bursts.

    Args:
        headlines_df: DataFrame with ``headline``, ``storyId``, ``versionCreated``.
        positive_threshold: Polarity threshold for the Positive label.
        negative_threshold: Polarity threshold for the Negative label.
        progress_callback: Optional callable accepting ``(fraction, message)``
            for progress reporting (e.g. from a Streamlit progress bar).

    Returns:
        DataFrame with columns: ``headline``, ``published``, ``sentiment``,
        ``polarity``, ``subjectivity``, ``story_text``, ``used_fallback``,
        ``error``.
    """
    records: list[dict[str, Any]] = []
    total = len(headlines_df)

    for i, row in enumerate(headlines_df.itertuples(index=False)):
        headline: str = str(getattr(row, "headline", "") or "")
        story_id: str = str(getattr(row, "storyId", "") or "")
        published = getattr(row, "versionCreated", pd.NaT)

        story_text = ""
        error_message = ""
        used_fallback = False

        try:
            raw_story = fetch_story(story_id)
            story_text = clean_html(raw_story)
            if not story_text:
                raise RuntimeError("Story content was empty after HTML cleaning.")
        except Exception as exc:
            error_message = str(exc)
            used_fallback = True
            story_text = headline
            logger.warning("Falling back to headline for %s: %s", story_id, exc)

        try:
            polarity, subjectivity, label = analyze_text(
                story_text, positive_threshold, negative_threshold
            )
        except Exception as exc:
            polarity, subjectivity, label = np.nan, np.nan, "Neutral"
            error_message = (error_message + " | " if error_message else "") + (
                f"Sentiment analysis failed: {exc}"
            )

        records.append(
            {
                "headline": headline,
                "published": published,
                "sentiment": label,
                "polarity": polarity,
                "subjectivity": subjectivity,
                "story_text": story_text,
                "used_fallback": used_fallback,
                "error": error_message,
            }
        )

        if progress_callback is not None:
            progress_callback((i + 1) / max(total, 1), f"Analyzed {i + 1} of {total} articles")

        # Gentle pacing between story requests to reduce API bursts.
        if i < total - 1:
            time.sleep(STORY_REQUEST_DELAY_SECONDS)

    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_inputs(
    query: str,
    start_date: date,
    end_date: date,
    max_headlines: int,
    positive_threshold: float,
    negative_threshold: float,
) -> list[str]:
    """
    Validate user inputs and return a list of human-readable error messages.

    Args:
        query: The LSEG news query string.
        start_date: Analysis start date.
        end_date: Analysis end date.
        max_headlines: Maximum number of headlines to retrieve.
        positive_threshold: Positive sentiment threshold.
        negative_threshold: Negative sentiment threshold.

    Returns:
        A list of error messages; empty when all inputs are valid.
    """
    errors: list[str] = []
    if not query or not query.strip():
        errors.append("The news query cannot be empty.")
    if start_date > end_date:
        errors.append("The start date must be on or before the end date.")
    if end_date > date.today():
        errors.append("The end date cannot be in the future.")
    if not (1 <= max_headlines <= MAX_HEADLINES_LIMIT):
        errors.append(f"Max headlines must be between 1 and {MAX_HEADLINES_LIMIT}.")
    if negative_threshold >= positive_threshold:
        errors.append(
            "The negative threshold must be strictly below the positive threshold."
        )
    return errors


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def build_distribution_chart(results: pd.DataFrame) -> Any:
    """
    Build a bar chart of the sentiment class distribution.

    Args:
        results: The full results DataFrame.

    Returns:
        A Plotly figure.
    """
    counts = (
        results["sentiment"]
        .value_counts()
        .reindex(SENTIMENT_ORDER, fill_value=0)
        .rename_axis("Sentiment")
        .reset_index(name="Articles")
    )
    fig = px.bar(
        counts,
        x="Sentiment",
        y="Articles",
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        title="Sentiment Distribution",
    )
    fig.update_layout(showlegend=False)
    return fig


def build_time_series_chart(results: pd.DataFrame) -> Any:
    """
    Build a line chart of average polarity by publication date.

    Args:
        results: The full results DataFrame.

    Returns:
        A Plotly figure.
    """
    ts = results.dropna(subset=["published", "polarity"]).copy()
    ts["pub_date"] = pd.to_datetime(ts["published"], utc=True).dt.date
    daily = ts.groupby("pub_date", as_index=False)["polarity"].mean()
    fig = px.line(
        daily,
        x="pub_date",
        y="polarity",
        markers=True,
        title="Average Sentiment (Polarity) Over Time",
        labels={"pub_date": "Date", "polarity": "Average polarity"},
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray")
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def render_sidebar() -> dict[str, Any]:
    """
    Render sidebar inputs and return the collected parameter values.

    Returns:
        Dict with keys: ``company``, ``query``, ``start_date``, ``end_date``,
        ``max_headlines``, ``positive_threshold``, ``negative_threshold``,
        ``run_clicked``.
    """
    st.sidebar.header("Analysis Parameters")

    company = st.sidebar.text_input("Company name", value="Goldman Sachs")
    default_query = f'"{company}" and Language:LEN' if company.strip() else ""
    query = st.sidebar.text_input(
        "LSEG news query",
        value=default_query,
        help='Example: "Goldman Sachs" and Language:LEN — or use RIC syntax like GS.N',
    )

    today = date.today()
    start_date = st.sidebar.date_input("Start date", value=today - timedelta(days=7))
    end_date = st.sidebar.date_input("End date", value=today)

    max_headlines = st.sidebar.slider(
        "Max headlines", min_value=1, max_value=MAX_HEADLINES_LIMIT, value=25
    )

    st.sidebar.subheader("Sentiment thresholds")
    positive_threshold = st.sidebar.slider(
        "Positive threshold (polarity ≥)", 0.0, 1.0, 0.10, 0.01
    )
    negative_threshold = st.sidebar.slider(
        "Negative threshold (polarity ≤)", -1.0, 0.0, -0.10, 0.01
    )

    run_clicked = st.sidebar.button("Run analysis", type="primary", use_container_width=True)

    return {
        "company": company,
        "query": query,
        "start_date": start_date,
        "end_date": end_date,
        "max_headlines": int(max_headlines),
        "positive_threshold": float(positive_threshold),
        "negative_threshold": float(negative_threshold),
        "run_clicked": bool(run_clicked),
    }


def render_metrics(results: pd.DataFrame) -> None:
    """
    Render summary metric tiles for the results.

    Args:
        results: The full results DataFrame.
    """
    total = len(results)
    avg_polarity = float(results["polarity"].mean()) if total else 0.0
    counts = results["sentiment"].value_counts()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total articles", total)
    c2.metric("Average polarity", f"{avg_polarity:+.3f}")
    c3.metric("Positive", int(counts.get("Positive", 0)))
    c4.metric("Neutral", int(counts.get("Neutral", 0)))
    c5.metric("Negative", int(counts.get("Negative", 0)))


def render_results_table(results: pd.DataFrame) -> None:
    """
    Render a searchable, sentiment-filterable article table.

    Args:
        results: The full results DataFrame.
    """
    st.subheader("Articles")

    fcol1, fcol2 = st.columns([1, 2])
    with fcol1:
        selected_sentiments = st.multiselect(
            "Filter by sentiment",
            options=SENTIMENT_ORDER,
            default=SENTIMENT_ORDER,
        )
    with fcol2:
        search_term = st.text_input(
            "Search headlines and story text", value="", placeholder="e.g. earnings"
        )

    filtered = results[results["sentiment"].isin(selected_sentiments)]
    if search_term.strip():
        needle = search_term.strip().lower()
        mask = filtered["headline"].str.lower().str.contains(needle, na=False) | filtered[
            "story_text"
        ].str.lower().str.contains(needle, na=False)
        filtered = filtered[mask]

    st.caption(f"Showing {len(filtered)} of {len(results)} articles.")

    display_df = filtered.rename(
        columns={
            "headline": "Headline",
            "published": "Publication date",
            "sentiment": "Sentiment",
            "polarity": "Polarity",
            "subjectivity": "Subjectivity",
            "story_text": "Story text",
            "error": "Retrieval error",
        }
    )[
        [
            "Headline",
            "Publication date",
            "Sentiment",
            "Polarity",
            "Subjectivity",
            "Story text",
            "Retrieval error",
        ]
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Polarity": st.column_config.NumberColumn(format="%.3f"),
            "Subjectivity": st.column_config.NumberColumn(format="%.3f"),
            "Story text": st.column_config.TextColumn(width="large"),
        },
    )


def main() -> None:
    """Application entry point: render the UI and orchestrate the pipeline."""
    st.set_page_config(
        page_title="LSEG News Sentiment Analysis",
        page_icon="📰",
        layout="wide",
    )
    st.title("📰 LSEG News Sentiment Analysis")
    st.caption(
        "Retrieves headlines and full stories from LSEG Workspace, then scores "
        "sentiment with TextBlob. Requires a running LSEG Workspace desktop session."
    )

    if ld is None:
        st.error(
            "The 'lseg-data' package is not installed. "
            "Run: pip install -r requirements.txt"
        )
        st.stop()

    params = render_sidebar()

    if "results" not in st.session_state:
        st.session_state["results"] = None
    if "last_run_meta" not in st.session_state:
        st.session_state["last_run_meta"] = None

    if params["run_clicked"]:
        errors = validate_inputs(
            query=params["query"],
            start_date=params["start_date"],
            end_date=params["end_date"],
            max_headlines=params["max_headlines"],
            positive_threshold=params["positive_threshold"],
            negative_threshold=params["negative_threshold"],
        )
        if errors:
            for message in errors:
                st.error(message)
        else:
            try:
                headlines = fetch_headlines(
                    query=params["query"].strip(),
                    start_date_iso=params["start_date"].isoformat(),
                    end_date_iso=params["end_date"].isoformat(),
                    max_headlines=params["max_headlines"],
                )
            except RuntimeError as exc:
                st.error(str(exc))
                headlines = None

            if headlines is not None:
                if headlines.empty:
                    st.warning(
                        "No headlines were found for this query and date range. "
                        "Try broadening the query or widening the dates."
                    )
                else:
                    progress = st.progress(0.0, text="Starting analysis...")

                    def _update(fraction: float, message: str) -> None:
                        progress.progress(min(fraction, 1.0), text=message)

                    results = run_sentiment_analysis(
                        headlines_df=headlines,
                        positive_threshold=params["positive_threshold"],
                        negative_threshold=params["negative_threshold"],
                        progress_callback=_update,
                    )
                    progress.empty()

                    st.session_state["results"] = results
                    st.session_state["last_run_meta"] = {
                        "query": params["query"].strip(),
                        "start": params["start_date"].isoformat(),
                        "end": params["end_date"].isoformat(),
                        "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }

    results: Optional[pd.DataFrame] = st.session_state.get("results")

    if results is None or results.empty:
        st.info("Configure the parameters in the sidebar and click **Run analysis**.")
        return

    meta = st.session_state.get("last_run_meta") or {}
    if meta:
        st.caption(
            f"Query: `{meta.get('query', '')}` · Range: {meta.get('start', '')} → "
            f"{meta.get('end', '')} · Last run: {meta.get('run_at', '')}"
        )

    fallback_count = int(results["used_fallback"].sum())
    if fallback_count:
        st.warning(
            f"{fallback_count} article(s) could not be fully retrieved; the headline "
            "text was used as a fallback for sentiment scoring."
        )

    render_metrics(results)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(build_distribution_chart(results), use_container_width=True)
    with chart_col2:
        st.plotly_chart(build_time_series_chart(results), use_container_width=True)

    render_results_table(results)


if __name__ == "__main__":
    main()