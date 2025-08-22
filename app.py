# Fantasy Draft Helper — Streamlit app
# ---------------------------------------------------
# Use this single-file app with your 2025 fantasy CSVs.
# Place these files in your repo (root or ./data):
#   QB_RANKINGS.csv, RB_RANKINGS.csv, WR_RANKINGS.csv,
#   TE_RANKINGS.csv, DST_RANKINGS.csv, K_RANKINGS.csv
# Then run:  streamlit run app.py   (or whatever you name this file)
#
# Features:
# - Load 6 position ranking CSVs and normalize columns
# - Show Best Available (all positions) + By Position views
# - Quickly mark players as drafted (others or yours)
# - Pre-load keepers (paste names) — Tee Higgins preselected for you
# - Queue/star players you like
# - Draft board with undo
# - Save/Load state to/from JSON (so you can persist mid-draft)
#
# Notes:
# - If your CSV column names differ, the loader tries to auto-detect
#   common variants ("Player", "Name", "Team", "Bye", "Rank", etc.).
# - If an overall rank column isn’t found, the app builds a cross-position
#   "overall_approx" using each position’s rank percentile.
# - This is offline-only and does not depend on ESPN/Sleeper APIs.

from __future__ import annotations
import json
import os
import re
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

# ----------------------------
# Page setup & styles
# ----------------------------
st.set_page_config(
    page_title="Fantasy Draft Helper 2025",
    page_icon="🏈",
    layout="wide",
)

HIDE_INDEX_CSS = """
<style>
    .block-container {padding-top: 1.6rem;}
    .stDataFrame td, .stDataFrame th {font-size: 0.95rem;}
    .small-note {color: #7a7a7a; font-size: 0.88rem;}
    .tag {display: inline-block; padding: 2px 8px; border-radius: 999px; border: 1px solid #e3e3e3; margin-right: 6px; font-size: 0.8rem;}
    .tag.good {background: #eef8ee;}
    .tag.warn {background: #fff6ea;}
    .pill {display:inline-block; padding: 3px 10px; border-radius: 9999px; border:1px solid #e5e7eb; font-size:0.82rem;}
    .queue-row {background: rgba(255, 235, 59, 0.12);} /* subtle yellow */
</style>
"""
st.markdown(HIDE_INDEX_CSS, unsafe_allow_html=True)

# ----------------------------
# Helpers
# ----------------------------

# Common synonyms for column detection
NAME_COLS = ["player", "name", "player name", "player_name"]
TEAM_COLS = ["team", "tm"]
BYE_COLS  = ["bye", "bye week", "bye_week", "byeweek"]
RANK_COLS = ["overall", "overall rank", "overall_rank", "ovr", "rk", "rank", "ecr", "adp", "consensus", "top200", "overallrank"]
POS_RANK_COLS = ["posrank", "pos_rank", "position rank", "position_rank", "prk", "rk_pos"]

POS_FILES = {
    "QB": "QB_RANKINGS.csv",
    "RB": "RB_RANKINGS.csv",
    "WR": "WR_RANKINGS.csv",
    "TE": "TE_RANKINGS.csv",
    "DST": "DST_RANKINGS.csv",
    "K": "K_RANKINGS.csv",
}

# Try both repo root and ./data
SEARCH_DIRS = [".", "./data", "./Data", "./datasets", "./rankings"]


def _find_first_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = [c.lower().strip() for c in cols]
    for cand in candidates:
        if cand in cols_lower:
            # Return the original column name with matching index
            return cols[cols_lower.index(cand)]
    return None


def _coalesce_columns(df: pd.DataFrame, wanted: List[str], new_name: str) -> Optional[str]:
    col = _find_first_col(list(df.columns), wanted)
    if col and col != new_name:
        df.rename(columns={col: new_name}, inplace=True)
        return new_name
    return col


def _safe_read_csv(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin-1")
        except Exception:
            return None


def discover_files() -> Dict[str, str]:
    found = {}
    for pos, fname in POS_FILES.items():
        for d in SEARCH_DIRS:
            path = os.path.join(d, fname)
            if os.path.exists(path):
                found[pos] = path
                break
    return found


def normalize_one_position(path: str, pos: str) -> pd.DataFrame:
    raw = _safe_read_csv(path)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["player", "team", "bye", "pos", "pos_rank", "overall", "source_file"])  # empty

    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]

    # Coalesce common columns
    name_col = _coalesce_columns(df, NAME_COLS, "player")
    team_col = _coalesce_columns(df, TEAM_COLS, "team")
    bye_col  = _coalesce_columns(df, BYE_COLS,  "bye")

    # Create position & source metadata
    df["pos"] = pos
    df["source_file"] = os.path.basename(path)

    # Position rank: use explicit if present, else index+1
    pos_rank_col = _find_first_col(list(df.columns), POS_RANK_COLS)
    if pos_rank_col is None:
        df["pos_rank"] = range(1, len(df) + 1)
    else:
        if pos_rank_col != "pos_rank":
            df.rename(columns={pos_rank_col: "pos_rank"}, inplace=True)
        # Clean numeric
        df["pos_rank"] = pd.to_numeric(df["pos_rank"], errors="coerce")

    # Overall (cross-position) if present
    overall_col = _find_first_col(list(df.columns), RANK_COLS)
    if overall_col is not None and overall_col != "overall":
        df.rename(columns={overall_col: "overall"}, inplace=True)
    if "overall" in df.columns:
        df["overall"] = pd.to_numeric(df["overall"], errors="coerce")

    # Ensure essential cols exist
    if "player" not in df.columns:
        # Try to build from first column if totally unknown
        first = df.columns[0]
        df.rename(columns={first: "player"}, inplace=True)
    if "team" not in df.columns:
        df["team"] = ""
    if "bye" not in df.columns:
        df["bye"] = ""

    # Trim & clean
    df["player"] = df["player"].astype(str).str.strip()
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df["bye"] = df["bye"].astype(str).str.strip()

    # Drop rows without player names
    df = df[df["player"].str.len() > 0].copy()

    return df


def load_all_positions(files: Dict[str, str]) -> pd.DataFrame:
    frames = []
    for pos, path in files.items():
        frames.append(normalize_one_position(path, pos))
    if not frames:
        return pd.DataFrame(columns=["player", "team", "bye", "pos", "pos_rank", "overall", "source_file"])  # empty
    all_df = pd.concat(frames, ignore_index=True)

    # Build a cross-position approximation if overall not widely present
    if all_df["overall"].notna().sum() < len(all_df) * 0.5:
        all_df["pos_count"] = all_df.groupby("pos")["player"].transform("count")
        all_df["pos_percentile"] = all_df["pos_rank"] / all_df["pos_count"]
        # Lower is better; fallback to percentile “overall_approx”
        all_df["overall_approx"] = all_df["pos_percentile"]
    else:
        # Normalize given overall into 0..1 for cross-position sort consistency
        # Smaller is better
        max_overall = all_df["overall"].max()
        all_df["overall_approx"] = all_df["overall"] / max_overall

    # Player stable id (handles dup names across positions)
    all_df["player_id"] = (
        all_df["player"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        + "|" + all_df["pos"].str.upper()
    )

    # Ensure numeric
    all_df["pos_rank"] = pd.to_numeric(all_df["pos_rank"], errors="coerce")

    # Sort default: best to worst
    all_df.sort_values(["overall_approx", "pos", "pos_rank"], inplace=True, ignore_index=True)
    return all_df


# ----------------------------
# Session state init
# ----------------------------

def init_state():
    if "players" not in st.session_state:
        files = discover_files()
        st.session_state.players = load_all_positions(files)
    if "drafted" not in st.session_state:
        st.session_state.drafted: List[dict] = []
    if "queue" not in st.session_state:
        st.session_state.queue: set[str] = set()
    if "pick_no" not in st.session_state:
        st.session_state.pick_no = 1
    if "preloaded_tee" not in st.session_state:
        st.session_state.preloaded_tee = False
    if "num_teams" not in st.session_state:
        st.session_state.num_teams = 10
    if "default_picker" not in st.session_state:
        st.session_state.default_picker = "Other Team"


init_state()
players = st.session_state.players

# ----------------------------
# Utilities for draft actions
# ----------------------------

PLAYER_COLUMNS_VIEW = [
    "player", "team", "pos", "pos_rank", "bye"
]


def is_drafted(player_id: str) -> bool:
    return any(pick["player_id"] == player_id for pick in st.session_state.drafted)


def add_to_queue(player_id: str):
    st.session_state.queue.add(player_id)


def remove_from_queue(player_id: str):
    st.session_state.queue.discard(player_id)


def mark_drafted(player_row: pd.Series, picker: str, method: str = "manual"):
    pid = player_row["player_id"]
    if is_drafted(pid):
        return
    st.session_state.drafted.append({
        "pick_no": st.session_state.pick_no,
        "player_id": pid,
        "player": player_row.get("player", ""),
        "team": player_row.get("team", ""),
        "pos": player_row.get("pos", ""),
        "picker": picker,
        "method": method,
    })
    st.session_state.pick_no += 1
    # If they were in queue, remove
    remove_from_queue(pid)


def undo_last_pick():
    if st.session_state.drafted:
        last = st.session_state.drafted.pop()
        st.session_state.pick_no = max(1, last["pick_no"])  # revert counter to last


def save_state_json() -> str:
    payload = {
        "drafted": st.session_state.drafted,
        "pick_no": st.session_state.pick_no,
        "queue": list(st.session_state.queue),
        "num_teams": st.session_state.num_teams,
        "default_picker": st.session_state.default_picker,
    }
    return json.dumps(payload, indent=2)


def load_state_json(text: str):
    try:
        data = json.loads(text)
        st.session_state.drafted = data.get("drafted", [])
        st.session_state.pick_no = int(data.get("pick_no", 1))
        st.session_state.queue = set(data.get("queue", []))
        st.session_state.num_teams = int(data.get("num_teams", 10))
        st.session_state.default_picker = data.get("default_picker", "Other Team")
        st.success("State loaded.")
    except Exception as e:
        st.error(f"Failed to load state: {e}")


# ----------------------------
# Sidebar: Settings & Keepers
# ----------------------------

with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.num_teams = st.number_input("Number of teams", min_value=4, max_value=20, step=1, value=st.session_state.num_teams)
    st.session_state.default_picker = st.text_input("Default picker label for drafts", value=st.session_state.default_picker)

    st.divider()
    st.subheader("📥 Load / 💾 Save")
    c1, c2 = st.columns(2)
    with c1:
        state_text = save_state_json()
        st.download_button("💾 Download state JSON", data=state_text, file_name="draft_state.json", mime="application/json")
    with c2:
        uploaded = st.file_uploader("Upload saved state JSON", type=["json"], label_visibility="collapsed")
        if uploaded is not None:
            load_state_json(uploaded.read().decode("utf-8"))

    st.divider()
    st.subheader("🧲 Pre-load keepers")
    st.caption("Paste one player per line. Example: `Tee Higgins`\nYou can include position in parentheses, e.g., `Tee Higgins (WR)`.")
    keepers_text = st.text_area("Keeper names", value="Tee Higgins", height=100)
    keeper_picker = st.text_input("Label for keeper picks (your team name)", value="Me (keeper)")
    if st.button("Mark keepers as drafted"):
        names = [n.strip() for n in keepers_text.splitlines() if n.strip()]
        hits, misses = 0, []
        for name in names:
            # Optional position in parentheses
            m = re.match(r"^(.*?)(\s*\((QB|RB|WR|TE|DST|K)\))?$", name, flags=re.I)
            qname = m.group(1).strip() if m else name
            qpos = (m.group(3) or "").upper() if m else ""

            # Candidate matches
            cand = players[players["player"].str.lower() == qname.lower()]
            if qpos:
                cand = cand[cand["pos"].str.upper() == qpos]
            if cand.empty:
                # fallback: substring contains
                cand = players[players["player"].str.lower().str.contains(qname.lower())]
                if qpos:
                    cand = cand[cand["pos"].str.upper() == qpos]
            if cand.empty:
                misses.append(name)
                continue
            # Choose the best-ranked among candidates not already drafted
            cand = cand[~cand["player_id"].isin([d["player_id"] for d in st.session_state.drafted])]
            if cand.empty:
                misses.append(name)
                continue
            row = cand.sort_values(["overall_approx", "pos_rank"]).iloc[0]
            mark_drafted(row, picker=keeper_picker, method="keeper")
            hits += 1
        if hits:
            st.success(f"Marked {hits} keeper(s) drafted.")
        if misses:
            st.warning("Not found or already drafted: " + ", ".join(misses))

    st.divider()
    st.subheader("🧹 Reset")
    reset_ok = st.checkbox("I understand this clears all picks", value=False)
    if st.button("Reset draft state", disabled=not reset_ok):
        st.session_state.drafted = []
        st.session_state.queue = set()
        st.session_state.pick_no = 1
        st.success("Draft state reset.")


# Preload Tee Higgins once if present and not already drafted
if not st.session_state.preloaded_tee and not any(d["method"] == "keeper" for d in st.session_state.drafted):
    tee = players[(players["player"].str.lower() == "tee higgins") & (players["pos"] == "WR")]
    if not tee.empty:
        mark_drafted(tee.iloc[0], picker="Me (keeper)", method="keeper")
        st.session_state.preloaded_tee = True


# ----------------------------
# Main layout
# ----------------------------

st.title("🏈 Fantasy Draft Helper 2025")

# Quick actions row
qc1, qc2, qc3, qc4 = st.columns([2,2,2,1])
with qc1:
    quick_name = st.text_input("Quick draft by name (press Enter)", placeholder="e.g., Puka Nacua or \"Puka Nacua (WR)\"")
with qc2:
    quick_picker = st.text_input("Picker label", value=st.session_state.default_picker)
with qc3:
    st.write("\n")
    if st.button("➕ Draft by name") and quick_name.strip():
        # Reuse keeper matching logic
        m = re.match(r"^(.*?)(\s*\((QB|RB|WR|TE|DST|K)\))?$", quick_name.strip(), flags=re.I)
        qname = m.group(1).strip() if m else quick_name.strip()
        qpos = (m.group(3) or "").upper() if m else ""
        cand = players[players["player"].str.lower() == qname.lower()]
        if qpos:
            cand = cand[cand["pos"].str.upper() == qpos]
        if cand.empty:
            cand = players[players["player"].str.lower().str.contains(qname.lower())]
            if qpos:
                cand = cand[cand["pos"].str.upper() == qpos]
        cand = cand[~cand["player_id"].isin([d["player_id"] for d in st.session_state.drafted])]
        if cand.empty:
            st.warning("No matching available player found.")
        else:
            row = cand.sort_values(["overall_approx", "pos_rank"]).iloc[0]
            mark_drafted(row, picker=quick_picker, method="manual")
            st.success(f"Drafted: {row['player']} ({row['pos']})")
with qc4:
    st.write("\n")
    if st.button("↩️ Undo last pick"):
        undo_last_pick()
        st.info("Last pick undone.")

st.write("<span class='small-note'>Tip: Add position in parentheses to disambiguate (e.g., **Kenneth Walker (RB)**). Use the **Load/Save** panel to persist mid-draft.</span>", unsafe_allow_html=True)


# Tabs
T1, T2, T3, T4 = st.tabs(["⭐ Best Available", "📊 By Position", "🧾 Draft Board", "📌 Queue & Search"])

# ---------- Tab 1: Best Available ----------
with T1:
    st.subheader("Best Available (All Positions)")
    hide_drafted = st.toggle("Hide drafted", value=True)
    limit = st.slider("Rows to show", min_value=25, max_value=400, step=25, value=100)

    df = players.copy()
    if hide_drafted:
        drafted_ids = {d["player_id"] for d in st.session_state.drafted}
        df = df[~df["player_id"].isin(drafted_ids)]

    # Decorate with queue flag
    df["queued"] = df["player_id"].isin(st.session_state.queue)

    view_cols = ["player", "team", "pos", "pos_rank", "bye", "queued"]
    view = df[view_cols].head(limit).reset_index(drop=True)
    st.dataframe(view, use_container_width=True, height=500)

    # Queue management
    st.markdown("**Queue actions**")
    cqa1, cqa2 = st.columns(2)
    with cqa1:
        add_names = st.text_input("Add to queue (comma-separated names)", placeholder="e.g., Jahmyr Gibbs, Sam LaPorta")
        if st.button("Add to queue") and add_names.strip():
            new_names = [n.strip() for n in add_names.split(",") if n.strip()]
            added = 0
            for nm in new_names:
                cand = players[players["player"].str.lower() == nm.lower()]
                if cand.empty:
                    cand = players[players["player"].str.lower().str.contains(nm.lower())]
                cand = cand[~cand["player_id"].isin([d["player_id"] for d in st.session_state.drafted])]
                if not cand.empty:
                    add_to_queue(cand.iloc[0]["player_id"])
                    added += 1
            st.success(f"Added {added} to queue.")
    with cqa2:
        rem_names = st.text_input("Remove from queue (comma-separated names)")
        if st.button("Remove from queue") and rem_names.strip():
            del_names = [n.strip() for n in rem_names.split(",") if n.strip()]
            removed = 0
            for nm in del_names:
                cand = players[players["player"].str.lower().str.contains(nm.lower())]
                for pid in cand["player_id"].tolist():
                    if pid in st.session_state.queue:
                        remove_from_queue(pid)
                        removed += 1
            st.info(f"Removed {removed} from queue.")


# ---------- Tab 2: By Position ----------
with T2:
    st.subheader("By Position")
    pos_choice = st.segmented_control("Position", options=["ALL", "QB", "RB", "WR", "TE", "DST", "K"], default="RB")

    df = players.copy()
    if pos_choice != "ALL":
        df = df[df["pos"] == pos_choice]

    drafted_ids = {d["player_id"] for d in st.session_state.drafted}
    df["drafted"] = df["player_id"].isin(drafted_ids)
    df["queued"] = df["player_id"].isin(st.session_state.queue)

    show_drafted = st.checkbox("Show drafted players in table", value=False)
    view_df = df if show_drafted else df[~df["drafted"]]

    # Editable selection column
    edit_cols = ["player", "team", "pos", "pos_rank", "bye", "queued"]
    view_df = view_df[edit_cols].copy()
    view_df.insert(0, "Select", False)

    edited = st.data_editor(
        view_df,
        use_container_width=True,
        height=520,
        column_config={
            "Select": st.column_config.CheckboxColumn(help="Select players to draft"),
            "queued": st.column_config.CheckboxColumn(help="Starred in your queue (toggle in 'Queue & Search')")
        },
        disabled=["player", "team", "pos", "pos_rank", "bye", "queued"],
        hide_index=True,
    )

    # Map edited rows back to base df by (player, pos)
    to_draft = []
    for _, row in edited.iterrows():
        if row.get("Select"):
            # Find matching row in base df
            base = players[(players["player"] == row["player"]) & (players["pos"] == row["pos"])]
            if not base.empty:
                to_draft.append(base.iloc[0])

    st.write("")
    colA, colB = st.columns([1,1])
    with colA:
        picker = st.text_input("Picker label for selected rows", value=st.session_state.default_picker, key="picker_bypos")
    with colB:
        if st.button("Draft selected players"):
            if not to_draft:
                st.warning("No players selected.")
            else:
                drafted_now = 0
                for base_row in to_draft:
                    if not is_drafted(base_row["player_id"]):
                        mark_drafted(base_row, picker=picker, method="manual")
                        drafted_now += 1
                if drafted_now:
                    st.success(f"Drafted {drafted_now} player(s).")


# ---------- Tab 3: Draft Board ----------
with T3:
    st.subheader("Draft Board")
    if not st.session_state.drafted:
        st.info("No picks yet. Use the other tabs to draft players.")
    else:
        board = pd.DataFrame(st.session_state.drafted)
        board = board.sort_values("pick_no").reset_index(drop=True)
        board["Pick"] = board["pick_no"]
        board = board[["Pick", "player", "team", "pos", "picker", "method"]]
        st.dataframe(board, use_container_width=True, height=520)

    d1, d2 = st.columns([1,1])
    with d1:
        if st.button("↩️ Undo last pick (here)"):
            undo_last_pick()
            st.info("Last pick undone.")
    with d2:
        csv = pd.DataFrame(st.session_state.drafted).to_csv(index=False)
        st.download_button("Download board CSV", data=csv, file_name="draft_board.csv", mime="text/csv")


# ---------- Tab 4: Queue & Search ----------
with T4:
    st.subheader("Queue & Search")

    qcol1, qcol2 = st.columns([2,1])
    with qcol1:
        q = st.text_input("Search players (name contains)", placeholder="e.g., 'waddle' or 'Bengals WR'")
    with qcol2:
        only_available = st.toggle("Only available", value=True)

    df = players.copy()
    if q.strip():
        ql = q.lower()
        # Support simple filters like "bengals wr"
        tokens = ql.split()
        mask = pd.Series([True] * len(df))
        for t in tokens:
            mask = mask & (
                df["player"].str.lower().str.contains(t) |
                df["team"].str.lower().str.contains(t) |
                df["pos"].str.lower().str.contains(t)
            )
        df = df[mask]

    drafted_ids = {d["player_id"] for d in st.session_state.drafted}
    if only_available:
        df = df[~df["player_id"].isin(drafted_ids)]

    df["queued"] = df["player_id"].isin(st.session_state.queue)

    view_cols = ["player", "team", "pos", "pos_rank", "bye", "queued"]
    st.dataframe(df[view_cols].head(400).reset_index(drop=True), use_container_width=True, height=520)

    st.markdown("**Queue quick actions**")
    qqa1, qqa2 = st.columns(2)
    with qqa1:
        nm = st.text_input("Add 1 player to queue")
        if st.button("Add to queue (1)") and nm.strip():
            cand = players[players["player"].str.lower().str.contains(nm.lower())]
            cand = cand[~cand["player_id"].isin(drafted_ids)]
            if not cand.empty:
                add_to_queue(cand.iloc[0]["player_id"])
                st.success("Added to queue.")
            else:
                st.warning("No available match.")
    with qqa2:
        nm2 = st.text_input("Remove 1 from queue")
        if st.button("Remove from queue (1)") and nm2.strip():
            cand = players[players["player"].str.lower().str.contains(nm2.lower())]
            removed = 0
            for pid in cand["player_id"].tolist():
                if pid in st.session_state.queue:
                    remove_from_queue(pid)
                    removed += 1
            if removed:
                st.info(f"Removed {removed}.")
            else:
                st.warning("No queued match.")


# Footer hint
st.write("\n")
st.caption("Built for manual-board drafts: track keepers, hide taken players, and always see the best available.")
