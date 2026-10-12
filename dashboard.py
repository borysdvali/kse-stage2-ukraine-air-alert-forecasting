"""Local research dashboard for saved, historical KSE Stage 2 forecasts.

Run with: streamlit run dashboard.py --server.address 127.0.0.1
The dashboard does not fit models or bypass scripts/predict_alert.py checks.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime, time, timezone
from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from dashboard_data import (  # noqa: E402
    BOUNDARY_PATH, BOUNDARY_SOURCE, REGION_LABELS, display_rows,
    format_forecast_time, map_geojson, run_historical_forecast, select_model,
)

st.set_page_config(page_title="KSE Stage 2 · Historical alert forecasts", layout="wide")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@st.cache_data(show_spinner=False)
def forecast_cached(model_path: str, as_of_iso: str, model_sha: str, raw_sha: str) -> list[dict]:
    # Hashes are cache-key inputs. The existing prediction command still checks
    # source identity, model validity, cadence and full horizon on cache misses.
    del model_sha, raw_sha
    return run_historical_forecast(Path(model_path), datetime.fromisoformat(as_of_iso))


@st.cache_data(show_spinner=False)
def boundaries_cached(path: str, file_sha: str) -> dict:
    del file_sha
    return json.loads(Path(path).read_text(encoding="utf-8"))


def show_region(row: dict, start_text: str, end_text: str) -> None:
    st.subheader(f"{row['Region']} · {row['Region ID']}")
    if row["Probability (%)"] is None:
        st.warning("No new-alert probability is issued for this region.")
        st.write(f"Archive state: **{row['Archive state']}**.")
        st.caption("An abstention is not a zero-percent forecast or an all-clear.")
    else:
        st.metric("Estimated chance of an alert starting in the next hour",
                  f"{row['Probability (%)']:.1f}%")
        st.write(f"Historical window: **{start_text}–{end_text}**. ")
        st.caption("This is for a reconstructed-clear region; it does not say an alert will or will not occur.")
    left, right = st.columns(2)
    with left:
        st.write("**Alert type:** any air alert")
        st.caption("Missile, propeller-UAV and jet-UAV probabilities are not separately validated.")
    with right:
        st.write("**Duration:** not estimated for this clear-region forecast")
        st.caption("The exploratory duration model predicts total duration only at alert onset, not remaining time.")


def show_map(rows: list[dict]) -> None:
    st.subheader("Regional map")
    if not BOUNDARY_PATH.exists():
        st.info("Map boundaries are not installed. Run `python scripts/prepare_dashboard_map.py` once, then refresh.")
        return
    geometry = boundaries_cached(str(BOUNDARY_PATH), sha256(BOUNDARY_PATH))
    joined = map_geojson(geometry, rows)
    layer = pdk.Layer(
        "GeoJsonLayer", joined, id="regions", pickable=True, auto_highlight=True,
        stroked=True, filled=True, get_fill_color="properties.fill_color",
        get_line_color=[105, 115, 130, 150], line_width_min_pixels=0.6,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(latitude=48.8, longitude=31.5, zoom=5, pitch=0),
        map_style=None,
        tooltip={"html": "<b>{properties.display_name}</b><br/>{properties.forecast_text}<br/>{properties.archive_state}"},
    )
    event = st.pydeck_chart(deck, key="regional-map", height=500, on_select="rerun",
                            selection_mode="single-object")
    selected = event.selection.get("objects", {}).get("regions", []) if event else []
    if selected:
        properties = selected[0].get("properties", {})
        st.write(f"**{properties.get('display_name', 'Region')}** — "
                 f"{properties.get('forecast_text', 'No onset forecast')} "
                 f"({properties.get('archive_state', 'Unknown state')}).")
    st.caption("Darker blue = higher estimated probability; amber = active or unresolved; gray = unsupported/no model. Color never means ‘safe’.")
    st.caption(f"Administrative boundaries: [geoBoundaries gbOpen UKR ADM1]({BOUNDARY_SOURCE}), 2017 reference geometry, ODbL 1.0. Boundaries are not current front lines.")


def show_comparison(rows: list[dict]) -> None:
    st.subheader("Compare regions")
    selected = st.multiselect(
        "Regions to show", options=sorted(REGION_LABELS), default=sorted(REGION_LABELS),
        format_func=lambda region_id: REGION_LABELS[region_id],
    )
    frame = pd.DataFrame(row for row in rows if row["Region ID"] in selected)
    if frame.empty:
        st.info("Select one or more regions to compare.")
        return
    frame = frame.sort_values(["Probability (%)", "Region"], ascending=[False, True], na_position="last")
    st.dataframe(
        frame[["Region", "Archive state", "Probability (%)"]],
        hide_index=True, width="stretch",
        column_config={"Probability (%)": st.column_config.NumberColumn(
            "Next-hour probability", format="%.1f%%", help="Blank means no onset forecast, not 0%."
        )},
    )
    st.caption("Rows with active, unresolved or insufficient states have no onset probability. Sort the table by clicking a column header.")


def show_review(rows: list[dict], raw: list[dict], model: Path, as_of: datetime) -> None:
    st.subheader("Review details")
    first = raw[0]
    count = sum(row["Probability (%)"] is not None for row in rows)
    st.write(f"**Archive:** {first['source']} · **forecast origin:** {as_of.isoformat()} · "
             f"**horizon:** {first['horizon_minutes']} minutes")
    st.write(f"**Model fit cutoff:** {first['fit_cutoff_utc']} · **regional forecasts:** {count}/{len(rows)} · "
             f"**abstentions:** {len(rows) - count}")
    st.success("The prediction command accepted this model, time, cadence and source snapshot.")
    st.caption(f"Saved model: `{model.relative_to(ROOT)}`. Only locally generated, trusted joblib models should be loaded.")

    metrics_path = ROOT / "reports" / "learning_baseline" / first["source"] / "metrics.json"
    if metrics_path.exists():
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        holdout = report.get("metrics", {}).get("holdout_frozen", {})
        model_score = holdout.get("logistic", {}).get("brier")
        baseline_score = holdout.get("region_hour_baseline", {}).get("brier")
        if model_score is not None and baseline_score is not None:
            st.write("**Archived holdout evaluation** (aggregate, not a score for this individual forecast)")
            st.write(f"Brier score: model **{model_score:.5f}** vs region/hour baseline **{baseline_score:.5f}**; lower is better.")
            st.caption("The saved report's holdout window and source coverage may differ from the time selected above.")
    else:
        st.caption("Detailed evaluation report not present locally; generated reports are excluded from GitHub.")

    with st.expander("Method and limitations"):
        st.write("The baseline estimates whether any alert starts in a reconstructed-clear region during the next hour. Features use archived events timestamped no later than the forecast origin; outcome windows must end before model fitting.")
        st.write("Historical reception and edit histories are unavailable, so event timestamps do not prove true point-in-time availability. Missing archive records are not verified no-alert ground truth.")
        st.write("The model does not distinguish missile from UAV threats, predict a route, or estimate when an active alert will end. Follow official alerts and instructions, not this research display.")


def main() -> None:
    st.title("Ukraine air-alert forecast · historical research view")
    st.warning("Retrospective archive replay — not a live warning system. Follow official alerts and guidance.")

    with st.sidebar:
        st.header("Forecast selection")
        source = st.selectbox("Archive", ["volunteer", "official"],
                              help="Models and archive coverage are source-specific.")
        max_date = date(2026, 9, 30) if source == "volunteer" else date(2026, 9, 6)
        chosen_date = st.date_input("Forecast date (UTC)", value=date(2026, 9, 3),
                                    min_value=date(2026, 1, 1), max_value=max_date)
        hour = st.number_input("Forecast hour (UTC)", min_value=0, max_value=23, value=12, step=1)
        st.caption("Time selection is in UTC; the result is also shown in Kyiv time.")
    as_of = datetime.combine(chosen_date, time(int(hour), 0), tzinfo=timezone.utc)
    try:
        model = select_model(source, as_of)
        raw_path = ROOT / "data" / "raw" / f"{source}_data_en.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing {raw_path.relative_to(ROOT)}. Follow the README data setup.")
        with st.spinner("Replaying the archived alert state..."):
            raw = forecast_cached(str(model), as_of.isoformat(), sha256(model), sha256(raw_path))
        rows = display_rows(raw)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        st.error(str(exc))
        st.stop()

    start_text, end_text = format_forecast_time(raw[0]["forecast_at_utc"], raw[0]["horizon_minutes"])
    st.caption(f"Forecast origin: {start_text} · horizon ends: {end_text} · {source} archive")
    region_tab, map_tab, compare_tab, review_tab = st.tabs(
        ["Region focus", "Map", "Regional comparison", "Review details"]
    )
    with region_tab:
        by_id = {row["Region ID"]: row for row in rows}
        ordered = sorted(by_id, key=lambda region_id: by_id[region_id]["Region"])
        default = ordered.index("UA-05") if "UA-05" in ordered else 0
        region_id = st.selectbox("Region", ordered, index=default,
                                 format_func=lambda value: by_id[value]["Region"])
        show_region(by_id[region_id], start_text, end_text)
    with map_tab:
        show_map(rows)
    with compare_tab:
        show_comparison(rows)
    with review_tab:
        show_review(rows, raw, model, as_of)


if __name__ == "__main__":
    main()
