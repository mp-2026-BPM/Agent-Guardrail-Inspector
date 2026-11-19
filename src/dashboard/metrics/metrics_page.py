from __future__ import annotations

import os
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

import panel as pn
import polars as pl

from src.agents import CUSTOMER_SCENARIO_LABELS
from src.config import CoffeeShopConfig
from src.trace_processing.eventlog_conversion import ObjectCentricEventlog

from ..nav import header_nav
from .complexity_section import ComplexitySection
from .conversation_composition_section import ConversationCompositionSection
from .feedback_section import FeedbackSection
from .overview_section import OverviewSection
from .guardrail_section import GuardrailSection
from .trace_cache import CACHE_FILENAME, ensure_trace_cache
from .visualization_section import VisualizationSection
from .styling_helpers import section_header


_TIMESTAMP_COL = "time:timestamp"
# Naive-UTC sibling of `time:timestamp`. Read by the guardrail-extension
# resolver so it doesn't double-shift the already UTC→local converted column.
_TIMESTAMP_UTC_NAIVE_COL = "time:timestamp_utc_naive"
_GUARDRAIL_LOG_PATH = Path(
    CoffeeShopConfig.__dataclass_fields__["guardrail_log_path"].default
)
# Set to a shared data folder (e.g. experiment_data/2026-11) to show its
# snapshot instead of the local MLflow store: the page reads the folder's
# generated_event_log/_all_traces.csv and guardrail_log/events.jsonl as they
# are, without syncing against MLflow.
_DATA_DIR_ENV = "METRICS_DATA_DIR"

# Scenario labels mirror the interaction observatory's dropdown
# (src/dashboard/interaction/interaction_page.py) — both dashboards use
# the "{index}: {label}" shorthand from CUSTOMER_SCENARIO_LABELS. -1 is
# the "no preset scenario" sentinel used at trace-tag time.
_SCENARIO_LABELS = {
    i: f"{i}: {label}" for i, label in enumerate(CUSTOMER_SCENARIO_LABELS)
}
_SCENARIO_LABELS[-1] = "Custom / Unspecified"
# Setup filter's option label for cases whose MLflow trace carried no `setup`
# tag (older traces, or a conversation that crashed before tagging). Kept as a
# named constant because it's the string that both populates the checkbox and
# encodes "null" in the applied-state tuple — one place to change.
_UNKNOWN_SETUP_LABEL = "(unknown)"
# Model filter's option label for cases whose events carried no `model`
# attribute (e.g. crashed before any `call_llm` event fired, or exports from
# before the `model` column existed). Same string as `_UNKNOWN_SETUP_LABEL`,
# but scoped independently so future divergence is a one-line change.
_UNKNOWN_MODEL_LABEL = "(unknown)"


def create_metrics_dashboard():
    """Create the Metrics Dashboard page."""
    pn.extension("plotly", sizing_mode="stretch_width")

    data_dir = os.getenv(_DATA_DIR_ENV)
    if data_dir:
        LOG_DIR = Path(data_dir) / "generated_event_log"
        snapshot = LOG_DIR / CACHE_FILENAME
        cache_path = snapshot if snapshot.exists() else None
        guardrail_log_path = Path(data_dir) / "guardrail_log" / "events.jsonl"
    else:
        LOG_DIR = Path("generated_event_log")
        cache_path = ensure_trace_cache(LOG_DIR)
        guardrail_log_path = _GUARDRAIL_LOG_PATH
    if cache_path is None:
        return _empty_template(LOG_DIR)

    combined, case_metadata = _load_combined_eventlog([cache_path])
    if combined.is_empty() or _TIMESTAMP_COL not in combined.columns:
        return _empty_template(LOG_DIR)

    ts_series = combined[_TIMESTAMP_COL].drop_nulls()
    if ts_series.is_empty():
        return _empty_template(LOG_DIR)

    full_start: datetime = ts_series.min()
    full_end: datetime = ts_series.max()
    # DatetimePicker's start/end must differ; if a single timestamp exists,
    # pad by 1s. Extend end past `now` so anchored-at-now presets ("Last 10
    # min") don't push the picker's value outside its bounds when data is
    # stale.
    if full_end <= full_start:
        full_end = full_start + timedelta(seconds=1)
    pick_min = full_start
    pick_max = max(full_end, datetime.now())

    # Discover scenario / setup options from data. Options are data-driven so
    # a new setup added later shows up automatically; scenario labels come
    # from _SCENARIO_LABELS with a fallback to the raw int for anything
    # outside the known range.
    scenarios_present = (
        sorted(
            int(v)
            for v in case_metadata["case_scenario_index"]
            .drop_nulls()
            .unique()
            .to_list()
        )
        if not case_metadata.is_empty()
        else []
    )
    scenario_options = {
        _SCENARIO_LABELS.get(v, f"Scenario {v}"): v for v in scenarios_present
    }
    setups_present_raw = (
        case_metadata["case_setup"].unique().to_list()
        if not case_metadata.is_empty()
        else []
    )
    setup_options = {}
    for s in setups_present_raw:
        if s is None:
            setup_options[_UNKNOWN_SETUP_LABEL] = None
        else:
            setup_options[s] = s
    models_present_raw = (
        case_metadata["case_model"].unique().to_list()
        if not case_metadata.is_empty() and "case_model" in case_metadata.columns
        else []
    )
    model_options: dict[str, str | None] = {}
    for m in sorted(models_present_raw, key=lambda x: (x is None, x or "")):
        if m is None:
            model_options[_UNKNOWN_MODEL_LABEL] = None
        else:
            model_options[m] = m

    start_picker = pn.widgets.DatetimePicker(
        name="From",
        value=full_start,
        start=pick_min,
        end=pick_max,
        enable_time=True,
        enable_seconds=True,
        military_time=True,
        sizing_mode="stretch_width",
    )
    end_picker = pn.widgets.DatetimePicker(
        name="To",
        value=full_end,
        start=pick_min,
        end=pick_max,
        enable_time=True,
        enable_seconds=True,
        military_time=True,
        sizing_mode="stretch_width",
    )

    def _preset_button(label: str, on_click) -> pn.widgets.Button:
        btn = pn.widgets.Button(
            name=label, button_type="light", height=28, sizing_mode="stretch_width"
        )
        btn.on_click(on_click)
        return btn

    def _set_range(start: datetime, end: datetime) -> None:
        """Write both pickers, clamped to the picker bounds so we never blow
        param validation. Live-update watchers fire naturally on assignment."""
        start = max(pick_min, min(pick_max, start))
        end = max(pick_min, min(pick_max, end))
        if end < start:
            end = start
        start_picker.value = start
        end_picker.value = end

    def _preset_all(_e=None):
        _set_range(full_start, full_end)

    def _preset_today(_e=None):
        now = datetime.now()
        _set_range(now.replace(hour=0, minute=0, second=0, microsecond=0), now)

    def _preset_last_24h(_e=None):
        now = datetime.now()
        _set_range(now - timedelta(hours=24), now)

    def _preset_last_hour(_e=None):
        now = datetime.now()
        _set_range(now - timedelta(hours=1), now)

    def _preset_last_10min(_e=None):
        now = datetime.now()
        _set_range(now - timedelta(minutes=10), now)

    preset_row_top = pn.Row(
        _preset_button("Last 10 min", _preset_last_10min),
        _preset_button("Last hour", _preset_last_hour),
        sizing_mode="stretch_width",
        margin=(0, 0, 4, 0),
    )
    preset_row_mid = pn.Row(
        _preset_button("Last 24h", _preset_last_24h),
        _preset_button("Today", _preset_today),
        sizing_mode="stretch_width",
        margin=(0, 0, 4, 0),
    )
    preset_row_bot = pn.Row(
        _preset_button("All", _preset_all),
        sizing_mode="stretch_width",
        margin=(0, 0, 6, 0),
    )

    scenario_group = pn.widgets.CheckBoxGroup(
        name="",
        options=scenario_options,
        value=[],
        inline=False,
        sizing_mode="stretch_width",
    )
    setup_group = pn.widgets.CheckBoxGroup(
        name="",
        options=setup_options,
        value=[],
        inline=False,
        sizing_mode="stretch_width",
    )
    model_group = pn.widgets.CheckBoxGroup(
        name="",
        options=model_options,
        value=[],
        inline=False,
        sizing_mode="stretch_width",
    )

    trace_count_label = pn.pane.HTML(
        _format_count_label(
            *_case_counts(case_metadata, full_start, full_end, [], [], [])
        ),
        sizing_mode="stretch_width",
        margin=(0, 0, 4, 0),
    )
    span_hint_label = pn.pane.HTML(
        _format_span_hint(case_metadata, full_start, full_end, [], [], []),
        sizing_mode="stretch_width",
        margin=(0, 0, 8, 0),
    )
    apply_button = pn.widgets.Button(
        name="Apply filters",
        button_type="primary",
        disabled=True,
        sizing_mode="stretch_width",
    )

    time_card = pn.Card(
        preset_row_top,
        preset_row_mid,
        preset_row_bot,
        start_picker,
        end_picker,
        title="Time",
        collapsed=False,
        sizing_mode="stretch_width",
        styles={"margin-bottom": "6px"},
    )
    scenario_card = pn.Card(
        scenario_group
        if scenario_options
        else pn.pane.HTML(
            '<div style="font-size:11px;color:#999;">No scenarios found.</div>'
        ),
        title="Scenario",
        collapsed=False,
        sizing_mode="stretch_width",
        styles={"margin-bottom": "6px"},
    )
    setup_card = pn.Card(
        setup_group
        if setup_options
        else pn.pane.HTML(
            '<div style="font-size:11px;color:#999;">No configurations found.</div>'
        ),
        title="Configuration",
        collapsed=False,
        sizing_mode="stretch_width",
        styles={"margin-bottom": "6px"},
    )
    model_card = pn.Card(
        model_group
        if model_options
        else pn.pane.HTML(
            '<div style="font-size:11px;color:#999;">No models found.</div>'
        ),
        title="Model",
        collapsed=False,
        sizing_mode="stretch_width",
        styles={"margin-bottom": "6px"},
    )

    # Applied state = filter values currently reflected by the metrics pane.
    # Staged state = whatever the widgets say right now. Apply button is
    # enabled iff the two diverge.
    applied: dict = {
        "start": full_start,
        "end": full_end,
        "scenarios": [],
        "setups": [],
        "models": [],
    }

    def _staged() -> dict:
        return {
            "start": start_picker.value,
            "end": end_picker.value,
            "scenarios": list(scenario_group.value),
            "setups": list(setup_group.value),
            "models": list(model_group.value),
        }

    def _restage(_event=None) -> None:
        s = _staged()
        trace_count_label.object = _format_count_label(
            *_case_counts(
                case_metadata,
                s["start"],
                s["end"],
                s["scenarios"],
                s["setups"],
                s["models"],
            )
        )
        span_hint_label.object = _format_span_hint(
            case_metadata,
            s["start"],
            s["end"],
            s["scenarios"],
            s["setups"],
            s["models"],
        )
        apply_button.disabled = _same_filter(s, applied)

    render_cache: OrderedDict[
        tuple[
            datetime,
            datetime,
            tuple[int, ...],
            tuple[str | None, ...],
            tuple[str | None, ...],
        ],
        tuple[ObjectCentricEventlog, int],
    ] = OrderedDict()

    def _render_from_applied() -> pn.viewable.Viewable:
        key = _filter_signature(
            applied["start"],
            applied["end"],
            applied["scenarios"],
            applied["setups"],
            applied["models"],
        )
        cached = render_cache.get(key)
        if cached is not None:
            render_cache.move_to_end(key)
            ocel, event_count = cached
            return _render_metrics_from_ocel(
                ocel,
                event_count,
                applied["start"],
                applied["end"],
            )
        view, entry = _build_metrics(
            combined,
            case_metadata,
            applied["start"],
            applied["end"],
            applied["scenarios"],
            applied["setups"],
            applied["models"],
            guardrail_log_path,
        )
        if entry is not None:
            render_cache[key] = entry
            while len(render_cache) > _RENDER_CACHE_SIZE:
                render_cache.popitem(last=False)
        return view

    metrics_pane = pn.Column(
        _render_from_applied(),
        sizing_mode="stretch_width",
        styles={"padding": "4px 0"},
    )

    def _on_apply(_event=None) -> None:
        applied.update(_staged())
        metrics_pane[:] = [_render_from_applied()]
        _restage()

    start_picker.param.watch(_restage, "value")
    end_picker.param.watch(_restage, "value")
    scenario_group.param.watch(_restage, "value")
    setup_group.param.watch(_restage, "value")
    model_group.param.watch(_restage, "value")
    apply_button.on_click(_on_apply)

    total_cases = case_metadata.height if not case_metadata.is_empty() else 0
    sidebar = pn.Column(
        pn.pane.HTML(
            '<div style="font-size:14px;font-weight:600;margin-bottom:6px;">Metrics Dashboard</div>',
            sizing_mode="stretch_width",
        ),
        pn.pane.HTML(
            f'<div style="font-size:11px;color:#999;margin-bottom:6px;">'
            f"Across {total_cases} trace(s) in MLflow store</div>",
            sizing_mode="stretch_width",
        ),
        time_card,
        scenario_card,
        setup_card,
        model_card,
        trace_count_label,
        span_hint_label,
        apply_button,
        width=320,
        styles={"padding": "10px 12px 10px 16px"},
    )

    return pn.template.FastListTemplate(
        title="Coffee Shop Agent Observatory",
        sidebar=[sidebar],
        header=[header_nav(active="/metrics")],
        main=[metrics_pane],
        accent_base_color="#795548",
        header_background="#4E342E",
        theme="default",
    )


def _load_combined_eventlog(csv_files: list[Path]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Read every CSV in `csv_files` and return `(combined, case_metadata)`.

    `combined` is the concatenated event log with a parsed datetime
    `time:timestamp`. `case_metadata` is one row per case with columns
    `case_id`, `case_setup`, `case_scenario_index`, `first_t`, `last_t` — the
    small lookup table the filter joins against. Files that fail to read are
    skipped silently; empty inputs yield empty frames.

    OpenTelemetry `start_time_unix_nano` is UTC, and `log_generator.py`
    writes the CSV strings as naive-UTC (no offset marker). The rest of the
    dashboard — preset buttons via `datetime.now()`, `DatetimePicker` values,
    `_apply_filters` — all use naive-local datetimes. To keep the comparison
    honest we convert UTC → local here on load. Naive datetimes throughout
    the dashboard match the user's clock; a CET user sees CET times, and
    "Last 10 min" arithmetic just works.

    We also keep the pre-conversion naive-UTC datetime as
    `time:timestamp_utc_naive` for the guardrail-extension resolver, whose
    loader assumes every naive datetime is UTC and would double-shift the
    already-converted column.
    """
    frames: list[pl.DataFrame] = []
    for path in csv_files:
        try:
            df = pl.read_csv(str(path), infer_schema_length=10_000)
        except Exception:
            continue
        if _TIMESTAMP_COL not in df.columns:
            continue
        frames.append(df)
    if not frames:
        return pl.DataFrame(), pl.DataFrame()
    local_tz = _local_tz_name()
    combined = (
        pl.concat(frames, how="diagonal_relaxed")
        .with_columns(
            pl.col(_TIMESTAMP_COL)
            .str.to_datetime(strict=False)
            .alias(_TIMESTAMP_UTC_NAIVE_COL)
        )
        .with_columns(
            pl.col(_TIMESTAMP_COL)
            .str.to_datetime(strict=False)
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone(local_tz)
            .dt.replace_time_zone(None)
        )
    )
    case_metadata = _build_case_metadata(combined)
    return combined, case_metadata


def _local_tz_name() -> str:
    """Best-effort IANA name for the local timezone (e.g. 'Europe/Berlin').

    Tries three sources in order:
    1. `datetime.now().astimezone().tzinfo.key` — set when the OS resolved
       into a `ZoneInfo`, typically on macOS and modern Linux distros with
       `TZ` set. On WSL and some containers this returns a plain
       `datetime.timezone` with no `.key` attribute.
    2. `/etc/timezone` — the canonical file on Debian-family systems and
       WSL. One line, the IANA name.
    3. `/etc/localtime` symlink target under `/usr/share/zoneinfo/…`.

    Falls back to UTC when none of the above yields a name. Comparisons
    still work in that case because both sides end up naive-UTC.
    """
    try:
        from datetime import datetime as _dt

        tz = _dt.now().astimezone().tzinfo
        key = getattr(tz, "key", None)
        if key:
            return key
    except Exception:
        pass
    try:
        etc = Path("/etc/timezone")
        if etc.exists():
            name = etc.read_text().strip()
            if name:
                return name
    except Exception:
        pass
    try:
        localtime = Path("/etc/localtime")
        if localtime.is_symlink():
            target = str(localtime.resolve())
            marker = "/zoneinfo/"
            idx = target.find(marker)
            if idx >= 0:
                return target[idx + len(marker) :]
    except Exception:
        pass
    return "UTC"


def _build_case_metadata(eventlog: pl.DataFrame) -> pl.DataFrame:
    """One row per case_id with setup, scenario, model, and first/last timestamps.

    Case-level attributes (`case_setup`, `case_scenario_index`) are already
    broadcast onto every event row by the extractor, so `.first()` per case
    is authoritative. `model` is event-level (populated only on `call_llm`
    rows), so we aggregate distinct non-null values per case and collapse:
    zero → null (unknown bucket), one → the model string, two-or-more →
    `"(mixed: A, B)"`. Missing columns fall back to null / -1 so older caches
    without the extractor's schema-v4 columns keep loading.
    """
    if eventlog.is_empty() or "case_id" not in eventlog.columns:
        return pl.DataFrame(
            schema={
                "case_id": pl.Utf8,
                "case_setup": pl.Utf8,
                "case_scenario_index": pl.Int64,
                "case_model": pl.Utf8,
                "first_t": pl.Datetime,
                "last_t": pl.Datetime,
            }
        )
    setup_expr = (
        pl.col("case_setup").first().alias("case_setup")
        if "case_setup" in eventlog.columns
        else pl.lit(None, dtype=pl.Utf8).alias("case_setup")
    )
    scenario_expr = (
        pl.col("case_scenario_index")
        .first()
        .cast(pl.Int64)
        .alias("case_scenario_index")
        if "case_scenario_index" in eventlog.columns
        else pl.lit(-1, dtype=pl.Int64).alias("case_scenario_index")
    )
    model_expr = (
        pl.col("model").drop_nulls().unique().sort().alias("_models")
        if "model" in eventlog.columns
        else pl.lit([], dtype=pl.List(pl.Utf8)).alias("_models")
    )
    aggregated = (
        eventlog.drop_nulls(_TIMESTAMP_COL)
        .group_by("case_id")
        .agg(
            setup_expr,
            scenario_expr,
            model_expr,
            pl.col(_TIMESTAMP_COL).min().alias("first_t"),
            pl.col(_TIMESTAMP_COL).max().alias("last_t"),
        )
    )
    return aggregated.with_columns(
        pl.when(pl.col("_models").list.len() == 0)
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(pl.col("_models").list.len() == 1)
        .then(pl.col("_models").list.first())
        .otherwise(pl.lit("(mixed: ") + pl.col("_models").list.join(", ") + pl.lit(")"))
        .alias("case_model")
    ).drop("_models")


def _apply_filters(
    case_metadata: pl.DataFrame,
    start: datetime,
    end: datetime,
    scenarios: list[int],
    setups: list[str | None],
    models: list[str | None],
) -> pl.DataFrame:
    """Return the subset of `case_metadata` matching all filter groups.

    Empty scenarios / setups / models mean "no filter" (all pass). Non-empty
    means "whitelist". Time keeps the existing 'fully-contained case'
    semantics.
    """
    if case_metadata.is_empty():
        return case_metadata
    filt = (pl.col("first_t") >= start) & (pl.col("last_t") <= end)
    if scenarios:
        filt = filt & pl.col("case_scenario_index").is_in(scenarios)
    if setups:
        has_unknown = None in setups
        concrete = [s for s in setups if s is not None]
        setup_filt = pl.lit(False)
        if concrete:
            setup_filt = setup_filt | pl.col("case_setup").is_in(concrete)
        if has_unknown:
            setup_filt = setup_filt | pl.col("case_setup").is_null()
        filt = filt & setup_filt
    if models:
        has_unknown = None in models
        concrete = [m for m in models if m is not None]
        model_filt = pl.lit(False)
        if concrete:
            model_filt = model_filt | pl.col("case_model").is_in(concrete)
        if has_unknown:
            model_filt = model_filt | pl.col("case_model").is_null()
        filt = filt & model_filt
    return case_metadata.filter(filt)


def _null_safe(x):
    """Sort key that orders None before any value."""
    return (x is None, x)


def _same_filter(a: dict, b: dict) -> bool:
    return (
        a["start"] == b["start"]
        and a["end"] == b["end"]
        and sorted(a["scenarios"]) == sorted(b["scenarios"])
        and sorted(a["setups"], key=_null_safe) == sorted(b["setups"], key=_null_safe)
        and sorted(a["models"], key=_null_safe) == sorted(b["models"], key=_null_safe)
    )


# 8 covers the typical preset-hopping workflow (toggle a filter, apply,
# toggle back) without holding the whole product-space of filter states.
_RENDER_CACHE_SIZE = 8


def _filter_signature(
    start: datetime,
    end: datetime,
    scenarios: list[int],
    setups: list[str | None],
    models: list[str | None],
) -> tuple:
    """Hashable, order-insensitive key for the render cache."""
    return (
        start,
        end,
        tuple(sorted(scenarios)),
        tuple(sorted(setups, key=_null_safe)),
        tuple(sorted(models, key=_null_safe)),
    )


def _case_counts(
    case_metadata: pl.DataFrame,
    start: datetime,
    end: datetime,
    scenarios: list[int],
    setups: list[str | None],
    models: list[str | None],
) -> tuple[int, int]:
    """Return (contained, partial) case counts for the current filter set.

    Contained cases pass all filters (time fully-contained AND scenario AND
    setup AND model). Partial counts cases that overlap the time window
    without full containment; scenario/setup/model are still applied so the
    "excluded" number reflects what a widened time window would rescue."""
    if case_metadata.is_empty():
        return 0, 0
    contained = _apply_filters(
        case_metadata, start, end, scenarios, setups, models
    ).height
    overlap_filt = (pl.col("first_t") <= end) & (pl.col("last_t") >= start)
    if scenarios:
        overlap_filt = overlap_filt & pl.col("case_scenario_index").is_in(scenarios)
    if setups:
        has_unknown = None in setups
        concrete = [s for s in setups if s is not None]
        s_filt = pl.lit(False)
        if concrete:
            s_filt = s_filt | pl.col("case_setup").is_in(concrete)
        if has_unknown:
            s_filt = s_filt | pl.col("case_setup").is_null()
        overlap_filt = overlap_filt & s_filt
    if models:
        has_unknown = None in models
        concrete = [m for m in models if m is not None]
        m_filt = pl.lit(False)
        if concrete:
            m_filt = m_filt | pl.col("case_model").is_in(concrete)
        if has_unknown:
            m_filt = m_filt | pl.col("case_model").is_null()
        overlap_filt = overlap_filt & m_filt
    overlapping = case_metadata.filter(overlap_filt).height
    return contained, overlapping - contained


def _format_count_label(contained: int, partial: int) -> str:
    plural = "trace" if contained == 1 else "traces"
    verb = "matches" if contained == 1 else "match"
    excluded_line = ""
    if partial > 0:
        excluded_plural = "trace" if partial == 1 else "traces"
        excluded_line = (
            f'<div style="font-size:11px;color:#999;padding-top:2px;">'
            f"{partial} partial {excluded_plural} excluded "
            f"(started before or ended after the window)</div>"
        )
    return (
        f'<div style="padding:4px 0;">'
        f'<div style="font-size:12px;color:#444;">'
        f"<b>{contained}</b> {plural} {verb} current filters</div>"
        f"{excluded_line}"
        f"</div>"
    )


def _format_span_hint(
    case_metadata: pl.DataFrame,
    start: datetime,
    end: datetime,
    scenarios: list[int],
    setups: list[str | None],
    models: list[str | None],
) -> str:
    """Show the actual time-span the currently-staged filter selects. Helps
    users understand "why is my Last-10-min preset empty" without needing
    them to zoom the picker."""
    filtered = _apply_filters(case_metadata, start, end, scenarios, setups, models)
    if filtered.is_empty():
        return (
            '<div style="font-size:11px;color:#999;padding-top:2px;">'
            "Selected conversations: none.</div>"
        )
    span_start = filtered["first_t"].min()
    span_end = filtered["last_t"].max()
    return (
        f'<div style="font-size:11px;color:#666;padding-top:2px;">'
        f"Selected conversations span "
        f"<b>{span_start:%Y-%m-%d %H:%M:%S}</b> → "
        f"<b>{span_end:%Y-%m-%d %H:%M:%S}</b></div>"
    )


def _build_metrics(
    eventlog: pl.DataFrame,
    case_metadata: pl.DataFrame,
    start: datetime,
    end: datetime,
    scenarios: list[int],
    setups: list[str | None],
    models: list[str | None],
    guardrail_log_path: Path = _GUARDRAIL_LOG_PATH,
) -> tuple[pn.viewable.Viewable, tuple | None]:
    """Returns `(view, cache_entry)`. `cache_entry` is `(ocel, event_count)`
    for the render cache, or `None` when the filter yielded nothing."""
    keep_ids = _apply_filters(case_metadata, start, end, scenarios, setups, models)[
        "case_id"
    ]
    if keep_ids.is_empty():
        return pn.pane.Alert(
            "No traces match the current filters.", alert_type="warning"
        ), None
    filtered = eventlog.filter(pl.col("case_id").is_in(keep_ids))
    if filtered.is_empty():
        return pn.pane.Alert(
            "No traces match the current filters.", alert_type="warning"
        ), None
    try:
        ocel = ObjectCentricEventlog.from_eventlog(
            filtered, guardrail_log_path=guardrail_log_path
        )
    except Exception as e:
        return pn.pane.Alert(f"Error processing events: {e}", alert_type="danger"), None
    view = _render_metrics_from_ocel(ocel, filtered.height, start, end)
    return view, (ocel, filtered.height)


def _render_metrics_from_ocel(
    ocel: ObjectCentricEventlog,
    event_count: int,
    start: datetime,
    end: datetime,
) -> pn.viewable.Viewable:
    """Compose the section panels around a pre-built OCEL.

    Separate from `_build_metrics` because Panel viewables can't safely be
    reused across mount cycles — we regenerate the tree each Apply even when
    the OCEL is cached.
    """
    range_label = _RangeLabel(start, end, event_count)
    return pn.Column(
        range_label.panel(),
        OverviewSection(ocel, range_label.fake_path()).panel(),
        ConversationCompositionSection(ocel).panel(),
        ComplexitySection(ocel).panel(),
        FeedbackSection(ocel).panel(),
        GuardrailSection(ocel).panel(),
        _lazy_visualization_panel(ocel),
        sizing_mode="stretch_width",
        styles={"padding": "4px 0"},
    )


def _lazy_visualization_panel(ocel) -> pn.viewable.Viewable:
    """Defer pm4py discovery + graphviz rendering behind a button click.

    Building `VisualizationSection` runs three pm4py discoveries and three
    graphviz `dot` invocations — the dominant cost of an Apply.
    """
    slot = pn.Column(
        pn.pane.HTML(
            '<div style="font-size:12px;color:#666;padding:4px 0;">'
            "Process visualization (Conversation DFG, OC-DFG, OC-PN) is "
            "generated on demand — it takes a few seconds.</div>",
            sizing_mode="stretch_width",
        ),
        sizing_mode="stretch_width",
    )
    button = pn.widgets.Button(
        name="Generate visualization",
        button_type="primary",
        width=220,
    )

    def _on_click(_event) -> None:
        button.disabled = True
        button.name = "Generating…"
        try:
            panel = VisualizationSection(ocel).panel()
        except Exception as e:
            panel = pn.pane.Alert(f"Visualization failed: {e}", alert_type="warning")
            button.disabled = False
            button.name = "Retry visualization"
            slot[:] = [panel, button]
        else:
            button.visible = False
            slot[:] = [panel]

    button.on_click(_on_click)
    slot.append(button)
    return pn.Column(
        section_header("Process Visualization"),
        slot,
        sizing_mode="stretch_width",
    )


class _RangeLabel:
    """Lightweight stand-in so OverviewSection (which expects a Path with
    `.name`) can render the active timeframe instead of a filename."""

    def __init__(self, start: datetime, end: datetime, event_count: int):
        self._start = start
        self._end = end
        self._event_count = event_count

    def fake_path(self) -> Path:
        # OverviewSection only reads `.name`; pass a Path whose name describes
        # the filter window so the existing header line stays informative.
        label = f"{self._start:%Y-%m-%d %H:%M:%S} → {self._end:%Y-%m-%d %H:%M:%S}"
        return Path(label)

    def panel(self) -> pn.pane.HTML:
        return pn.pane.HTML(
            f'<div style="font-size:11px;color:#666;padding:2px 0 6px 0;">'
            f"<b>Window:</b> {self._start:%Y-%m-%d %H:%M:%S} → "
            f"{self._end:%Y-%m-%d %H:%M:%S}  ·  "
            f"{self._event_count:,} events in window</div>",
            sizing_mode="stretch_width",
        )


def _empty_template(log_dir: Path) -> pn.template.FastListTemplate:
    return pn.template.FastListTemplate(
        title="Coffee Shop Metrics",
        sidebar=[
            pn.pane.Alert(
                f"No MLflow traces found. Run a conversation in the "
                f"Interaction Observatory or via `poetry run simulate` "
                f"first. (Cache directory: **{log_dir.resolve()}**, "
                f"file: `{CACHE_FILENAME}`)",
                alert_type="warning",
            )
        ],
        header=[header_nav(active="/metrics")],
        main=[],
        accent_base_color="#795548",
        header_background="#4E342E",
        theme="default",
    )
