"""
LSEG News Sentiment Analysis — Streamlit application.

Follows the structure of the LSEG Developer Portal article
"Introduction to News Sentiment Analysis with Eikon Data APIs - a Python
example", ported to the LSEG Data Library for Python (lseg.data):

    1. Get headlines into a dataframe   -> ld.news.get_headlines()
    2. Add Polarity / Subjectivity / Score columns
    3. Loop over storyId, fetch story   -> ld.news.get_story()
       clean with BeautifulSoup, score with TextBlob,
       bucket into positive / neutral / negative
    4. (Optional) Get minute price bars -> ld.get_history()
       and compute t+2m / t+5m / t+10m / t+30m returns per news item,
       then group mean returns by Score bucket

Requires a running, logged-in LSEG Workspace desktop session.
Run with: streamlit run app.py
"""

from __future__ import annotations

import datetime
import logging
import time
from datetime import date, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from bs4 import BeautifulSoup
from textblob import TextBlob

try:
    import lseg.data as ld
except ImportError:  # pragma: no cover - handled in the UI
    ld = None  # type: ignore[assignment]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lseg_sentiment_app")

STORY_REQUEST_DELAY_SECONDS: float = 0.35
HEADLINES_CACHE_TTL_SECONDS: int = 600
MAX_HEADLINES_LIMIT: int = 100
SCORE_ORDER: list[str] = ["positive", "neutral", "negative"]
SCORE_COLORS: dict[str, str] = {
    "positive": "#2E8B57",
    "neutral": "#8C8C8C",
    "negative": "#C0392B",
}
RETURN_HORIZONS_MIN: list[tuple[str, int]] = [
    ("twoM", 2),
    ("fiveM", 5),
    ("tenM", 10),
    ("thirtyM", 30),
]


# ---------------------------------------------------------------------------
# Session handling
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Opening LSEG Workspace session...")
def get_lseg_session() -> Any:
    """
    Open (and cache) a single LSEG Workspace desktop session.

    Returns:
        The opened LSEG session object.

    Raises:
        RuntimeError: If lseg-data is missing or the session cannot open.
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
        app_key = None

    try:
        if app_key:
            session = ld.session.desktop.Definition(app_key=app_key).get_session()
            session.open()
        else:
            session = ld.open_session()
        if session is None:
            raise RuntimeError("open_session() returned None.")
        ld.session.set_default(session)
    except Exception as exc:
        raise RuntimeError(
            "Could not open an LSEG Workspace session. Make sure the LSEG "
            "Workspace desktop application is running and you are logged in. "
            f"Underlying error: {exc}"
        ) from exc
    return session


def ensure_session_open() -> Any:
    """
    Return an OPEN LSEG session, rebuilding the cached one if it went stale.

    Returns:
        An open LSEG session object.

    Raises:
        RuntimeError: If a session cannot be (re)opened.
    """
    session = get_lseg_session()
    is_open = False
    try:
        state = getattr(session, "open_state", None)
        is_open = state is not None and str(state).lower().endswith("opened")
    except Exception:
        is_open = False

    if not is_open:
        logger.warning("Cached LSEG session is not open; rebuilding it.")
        try:
            session.close()
        except Exception:
            pass
        get_lseg_session.clear()
        session = get_lseg_session()

    ld.session.set_default(session)
    return session


# ---------------------------------------------------------------------------
# Step 1 — headlines dataframe (article's In [2])
# ---------------------------------------------------------------------------

@st.cache_data(ttl=HEADLINES_CACHE_TTL_SECONDS, show_spinner="Fetching headlines...")
def get_headlines_df(
    query: str,
    start_date_iso: str,
    end_date_iso: str,
    count: int,
) -> pd.DataFrame:
    """
    Retrieve headlines into a dataframe with columns
    ``versionCreated``, ``text``, ``storyId`` (article-style layout).

    Args:
        query: LSEG news query, e.g. 'R:GS.N AND Language:LEN'.
        start_date_iso: Start date, YYYY-MM-DD.
        end_date_iso: End date, YYYY-MM-DD.
        count: Number of headlines to retrieve (1-100).

    Returns:
        Headlines dataframe (possibly empty).

    Raises:
        RuntimeError: If the API call fails or the shape is unexpected.
    """
    ensure_session_open()
    try:
        df = ld.news.get_headlines(
            query, start=start_date_iso, end=end_date_iso, count=count
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to retrieve headlines from LSEG: {exc}") from exc

    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["versionCreated", "text", "storyId"])

    df = df.copy()
    # In lseg.data the timestamp is the index and the headline column is
    # named 'headline'; normalize to the article's 'text' convention.
    df = df.reset_index()
    rename_map: dict[str, str] = {}
    if "headline" in df.columns:
        rename_map["headline"] = "text"
    if "versionCreated" not in df.columns:
        rename_map[df.columns[0]] = "versionCreated"
    df = df.rename(columns=rename_map)

    missing = {"text", "storyId", "versionCreated"} - set(df.columns)
    if missing:
        raise RuntimeError(f"Unexpected headlines response; missing: {sorted(missing)}")

    df["versionCreated"] = pd.to_datetime(df["versionCreated"], errors="coerce", utc=True)
    return df[["versionCreated", "text", "storyId"]]


# ---------------------------------------------------------------------------
# Step 2 + 3 — sentiment loop (article's In [3] and In [4])
# ---------------------------------------------------------------------------

def classify_score(polarity: float, pos_threshold: float, neg_threshold: float) -> str:
    """
    Bucket a polarity into 'positive' / 'neutral' / 'negative'.

    Mirrors the article's logic (>= 0.05 positive, <= -0.05 negative by
    default) but with user-configurable thresholds.

    Args:
        polarity: TextBlob polarity in [-1, 1].
        pos_threshold: Polarity at or above which the score is positive.
        neg_threshold: Polarity at or below which the score is negative.

    Returns:
        One of 'positive', 'neutral', 'negative'.
    """
    if polarity >= pos_threshold:
        return "positive"
    if polarity <= neg_threshold:
        return "negative"
    return "neutral"


def score_sentiment(
    df: pd.DataFrame,
    pos_threshold: float,
    neg_threshold: float,
    progress_callback: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Iterate over the headlines dataframe, fetch each story, clean HTML,
    score sentiment with TextBlob, and write Polarity / Subjectivity /
    Score / storyText / error columns back onto the dataframe — exactly
    the article's In [4] loop, hardened for production.

    If a story fetch fails, the headline text is used as a fallback and
    the error is recorded; one failed article never stops the run.

    Args:
        df: Headlines dataframe (versionCreated, text, storyId).
        pos_threshold: Positive polarity threshold.
        neg_threshold: Negative polarity threshold.
        progress_callback: Optional callable (fraction, message).

    Returns:
        The dataframe with added sentiment columns.
    """
    df = df.copy()
    df["Polarity"] = np.nan
    df["Subjectivity"] = np.nan
    df["Score"] = ""
    df["storyText"] = ""
    df["error"] = ""

    total = len(df)
    for idx, story_id in enumerate(df["storyId"].values):  # for each row
        news_text = ""
        try:
            ensure_session_open()
            news_text = ld.news.get_story(str(story_id))  # get the news story
        except Exception as exc:
            df.iloc[idx, df.columns.get_loc("error")] = str(exc)
            logger.warning("get_story failed for %s: %s", story_id, exc)

        if news_text:
            soup = BeautifulSoup(str(news_text), "lxml")  # strip HTML
            plain_text = " ".join(soup.get_text(separator=" ").split())
        else:
            plain_text = str(df["text"].iloc[idx])  # headline fallback

        try:
            sent_a = TextBlob(plain_text)  # analyse text
            polarity = float(sent_a.sentiment.polarity)
            subjectivity = float(sent_a.sentiment.subjectivity)
            df.iloc[idx, df.columns.get_loc("Polarity")] = polarity
            df.iloc[idx, df.columns.get_loc("Subjectivity")] = subjectivity
            df.iloc[idx, df.columns.get_loc("Score")] = classify_score(
                polarity, pos_threshold, neg_threshold
            )
        except Exception as exc:
            df.iloc[idx, df.columns.get_loc("Score")] = "neutral"
            prev = df["error"].iloc[idx]
            df.iloc[idx, df.columns.get_loc("error")] = (
                (prev + " | " if prev else "") + f"Sentiment failed: {exc}"
            )

        df.iloc[idx, df.columns.get_loc("storyText")] = plain_text

        if progress_callback is not None:
            progress_callback((idx + 1) / max(total, 1), f"Analyzed {idx + 1}/{total}")
        if idx < total - 1:
            time.sleep(STORY_REQUEST_DELAY_SECONDS)  # pace API requests

    return df


# ---------------------------------------------------------------------------
# Step 4 — price impact (article's In [5]–[8]), optional
# ---------------------------------------------------------------------------

def add_price_impact(df: pd.DataFrame, ric: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute % returns at t+2/5/10/30 minutes after each news item and
    aggregate mean returns per Score bucket, mirroring the article's
    price-impact section. Uses ld.get_history for minute CLOSE bars.

    News outside market hours simply gets NaN returns (as in the article,
    those items are effectively discarded from the aggregation).

    Args:
        df: Scored dataframe with 'versionCreated' and 'Score'.
        ric: Instrument RIC, e.g. 'GS.N'.

    Returns:
        Tuple of (dataframe with twoM/fiveM/tenM/thirtyM columns,
        grouped mean dataframe by Score).

    Raises:
        RuntimeError: If minute price history cannot be retrieved.
    """
    ensure_session_open()

    start = pd.to_datetime(df["versionCreated"].min()).strftime("%Y-%m-%d")
    end = (
        pd.to_datetime(df["versionCreated"].max()) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    try:
        minute = ld.get_history(
            universe=ric,
            fields=["TRDPRC_1"],
            interval="minute",
            start=start,
            end=end,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to retrieve minute bars for '{ric}': {exc}") from exc

    if minute is None or len(minute) == 0:
        raise RuntimeError(f"No minute price history returned for '{ric}'.")

    close = minute.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None)

    df = df.copy()
    for col, _ in RETURN_HORIZONS_MIN:
        df[col] = np.nan

    for idx, news_date in enumerate(df["versionCreated"].values):
        s_time = pd.to_datetime(news_date)
        if s_time.tzinfo is not None:
            s_time = s_time.tz_convert(None) if s_time.tz is not None else s_time
        s_time = s_time.replace(second=0, microsecond=0, tzinfo=None)
        try:
            t0 = close.loc[s_time]
            for col, minutes in RETURN_HORIZONS_MIN:
                t_n = close.loc[s_time + datetime.timedelta(minutes=minutes)]
                df.iloc[idx, df.columns.get_loc(col)] = (t_n / t0 - 1) * 100
        except (KeyError, Exception):
            # Outside market hours or missing bar — skip, as in the article.
            pass

    grouped = (
        df.groupby("Score")[["Polarity", "Subjectivity"] + [c for c, _ in RETURN_HORIZONS_MIN]]
        .mean()
        .reindex(SCORE_ORDER)
    )
    return df, grouped


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_inputs(
    query: str,
    start_date: date,
    end_date: date,
    count: int,
    pos_threshold: float,
    neg_threshold: float,
) -> list[str]:
    """
    Validate user inputs.

    Returns:
        A list of error messages; empty when valid.
    """
    errors: list[str] = []
    if not query or not query.strip():
        errors.append("The news query cannot be empty.")
    if start_date > end_date:
        errors.append("The start date must be on or before the end date.")
    if end_date > date.today():
        errors.append("The end date cannot be in the future.")
    if not (1 <= count <= MAX_HEADLINES_LIMIT):
        errors.append(f"Max headlines must be between 1 and {MAX_HEADLINES_LIMIT}.")
    if neg_threshold >= pos_threshold:
        errors.append("The negative threshold must be strictly below the positive one.")
    return errors


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def render_sidebar() -> dict[str, Any]:
    """Render sidebar inputs and return their values."""
    st.sidebar.header("Analysis Parameters")

    company = st.sidebar.text_input("Company name", value="Goldman Sachs")
    ric = st.sidebar.text_input(
        "RIC (for query and price impact)", value="GS.N",
        help="Used to build the default article-style query R:<RIC> AND Language:LEN",
    )
    default_query = f"R:{ric} AND Language:LEN" if ric.strip() else ""
    query = st.sidebar.text_input(
        "LSEG news query", value=default_query,
        help="Article-style query, e.g. R:GS.N AND Language:LEN",
    )

    today = date.today()
    start_date = st.sidebar.date_input("Start date", value=today - timedelta(days=7))
    end_date = st.sidebar.date_input("End date", value=today)
    count = st.sidebar.slider("Max headlines", 1, MAX_HEADLINES_LIMIT, 100)

    st.sidebar.subheader("Score thresholds (article default: ±0.05)")
    pos_threshold = st.sidebar.slider("Positive if polarity ≥", 0.0, 1.0, 0.05, 0.01)
    neg_threshold = st.sidebar.slider("Negative if polarity ≤", -1.0, 0.0, -0.05, 0.01)

    do_price_impact = st.sidebar.checkbox(
        "Compute price impact (t+2/5/10/30 min returns)", value=True
    )
    run_clicked = st.sidebar.button("Run analysis", type="primary", use_container_width=True)

    return {
        "company": company,
        "ric": ric.strip(),
        "query": query,
        "start_date": start_date,
        "end_date": end_date,
        "count": int(count),
        "pos_threshold": float(pos_threshold),
        "neg_threshold": float(neg_threshold),
        "do_price_impact": bool(do_price_impact),
        "run_clicked": bool(run_clicked),
    }


def render_metrics(df: pd.DataFrame) -> None:
    """Render summary metric tiles."""
    counts = df["Score"].value_counts()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total articles", len(df))
    c2.metric("Average polarity", f"{float(df['Polarity'].mean()):+.3f}")
    c3.metric("Positive", int(counts.get("positive", 0)))
    c4.metric("Neutral", int(counts.get("neutral", 0)))
    c5.metric("Negative", int(counts.get("negative", 0)))


def render_charts(df: pd.DataFrame) -> None:
    """Render distribution and time-series charts."""
    col1, col2 = st.columns(2)
    with col1:
        counts = (
            df["Score"].value_counts().reindex(SCORE_ORDER, fill_value=0)
            .rename_axis("Score").reset_index(name="Articles")
        )
        fig = px.bar(
            counts, x="Score", y="Articles", color="Score",
            color_discrete_map=SCORE_COLORS, title="Sentiment Distribution",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        ts = df.dropna(subset=["versionCreated", "Polarity"]).copy()
        ts["pub_date"] = pd.to_datetime(ts["versionCreated"], utc=True).dt.date
        daily = ts.groupby("pub_date", as_index=False)["Polarity"].mean()
        fig2 = px.line(
            daily, x="pub_date", y="Polarity", markers=True,
            title="Average Polarity Over Time",
            labels={"pub_date": "Date", "Polarity": "Average polarity"},
        )
        fig2.add_hline(y=0.0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig2, use_container_width=True)


def render_table(df: pd.DataFrame) -> None:
    """Render the searchable, filterable article table."""
    st.subheader("Articles")
    fcol1, fcol2 = st.columns([1, 2])
    with fcol1:
        selected = st.multiselect("Filter by score", SCORE_ORDER, default=SCORE_ORDER)
    with fcol2:
        needle = st.text_input("Search headlines and story text", "")

    filtered = df[df["Score"].isin(selected)]
    if needle.strip():
        n = needle.strip().lower()
        mask = (
            filtered["text"].str.lower().str.contains(n, na=False)
            | filtered["storyText"].str.lower().str.contains(n, na=False)
        )
        filtered = filtered[mask]

    st.caption(f"Showing {len(filtered)} of {len(df)} articles.")
    display = filtered.rename(
        columns={
            "text": "Headline", "versionCreated": "Publication date",
            "Score": "Sentiment", "storyText": "Story text", "error": "Retrieval error",
        }
    )[["Headline", "Publication date", "Sentiment", "Polarity",
       "Subjectivity", "Story text", "Retrieval error"]]
    st.dataframe(
        display, use_container_width=True, hide_index=True,
        column_config={
            "Polarity": st.column_config.NumberColumn(format="%.3f"),
            "Subjectivity": st.column_config.NumberColumn(format="%.3f"),
            "Story text": st.column_config.TextColumn(width="large"),
        },
    )


def main() -> None:
    """Application entry point."""
    st.set_page_config(page_title="LSEG News Sentiment Analysis", page_icon="📰", layout="wide")
    st.title("📰 LSEG News Sentiment Analysis")
    st.caption(
        "Article-style pipeline: get_headlines → get_story → BeautifulSoup → "
        "TextBlob → Score buckets → optional price-impact aggregation."
    )

    if ld is None:
        st.error("The 'lseg-data' package is not installed. Run: pip install -r requirements.txt")
        st.stop()

    params = render_sidebar()
    st.session_state.setdefault("df", None)
    st.session_state.setdefault("grouped", None)

    if params["run_clicked"]:
        errors = validate_inputs(
            params["query"], params["start_date"], params["end_date"],
            params["count"], params["pos_threshold"], params["neg_threshold"],
        )
        if errors:
            for message in errors:
                st.error(message)
        else:
            try:
                df = get_headlines_df(
                    params["query"].strip(),
                    params["start_date"].isoformat(),
                    params["end_date"].isoformat(),
                    params["count"],
                )
            except RuntimeError as exc:
                st.error(str(exc))
                df = None

            if df is not None:
                if df.empty:
                    st.warning("No headlines found. Broaden the query or widen the dates.")
                else:
                    progress = st.progress(0.0, text="Scoring sentiment...")
                    df = score_sentiment(
                        df, params["pos_threshold"], params["neg_threshold"],
                        lambda f, m: progress.progress(min(f, 1.0), text=m),
                    )
                    progress.empty()

                    grouped = None
                    if params["do_price_impact"] and params["ric"]:
                        try:
                            with st.spinner("Computing price impact..."):
                                df, grouped = add_price_impact(df, params["ric"])
                        except RuntimeError as exc:
                            st.warning(f"Price impact skipped: {exc}")

                    st.session_state["df"] = df
                    st.session_state["grouped"] = grouped

    df: Optional[pd.DataFrame] = st.session_state.get("df")
    if df is None or df.empty:
        st.info("Configure the parameters in the sidebar and click **Run analysis**.")
        return

    failures = int((df["error"] != "").sum())
    if failures:
        st.warning(f"{failures} article(s) had retrieval issues; headline text was used as fallback.")

    render_metrics(df)
    render_charts(df)

    grouped: Optional[pd.DataFrame] = st.session_state.get("grouped")
    if grouped is not None:
        st.subheader("Mean returns after news, by Score bucket (%)")
        st.caption(
            "Average t+2/5/10/30-minute returns per sentiment bucket. News "
            "outside market hours is excluded, as in the reference article."
        )
        st.dataframe(grouped.style.format("{:.4f}"), use_container_width=True)

    render_table(df)


if __name__ == "__main__":
    main()
