# Fantasy Draft Helper — Streamlit app (with League Setup & Keeper Line 0)
# ---------------------------------------------------------------------
# New features per request:
# - League Setup tab: enter 10 team names & draft order (slots 1..10).
# - For each team, specify an optional keeper ("Name" or "Name (POS)").
# - Draft Board shows a locked **Pick 0** row per team for keepers.
#     • If a keeper is provided and matched, that player is auto-marked drafted.
#     • If no keeper yet, placeholder remains on Pick 0 and does NOT affect availability.
# - Players marked as keepers are removed from Best Available / By Position.
# - Draft order panel renders on the Draft Board.
# - Optional auto-assign of "picker" by draft order (linear or snake).
# - State save/load includes league config & keepers.
#
# How to run:
#   1) Put CSVs in repo root or ./data: QB_RANKINGS.csv, RB_RANKINGS.csv, WR_RANKINGS.csv,
#      TE_RANKINGS.csv, DST_RANKINGS.csv, K_RANKINGS.csv
#   2) pip install streamlit pandas
#   3) streamlit run app.py

from __future__ import annotations
import json
import os
import re
from typing import Dict, List, Optional, Tuple

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

CSS = """
<style>
  .block-container {padding-top: 1.2rem;}
  .stDataFrame td, .stDataFrame th {font-size: 0.95rem;}
  .small-note {color: #7a7a7a; font-size: 0.88rem;}
  .tag {display:inline-block; padding:2px 8px; border-radius:999px; border:1px solid #e3e3e3; margin-right:6px; font-size:0.8rem;}
  .pill {display:inline-block; padding:3px 10px; border-radius: 9999px; border:1px solid #e5e7eb; font-size:0.82rem;}
  .keeper-row {background: rgba(76, 175, 80, 0.08);} /* subtle green */
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ----------------------------
# CSV discovery & normalization
# ----------------------------
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

SEARCH_DIRS = [".", "./data", "./Data", "./datasets", "./rankings"]

def _find_first_col(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = [c.lower().strip() for c in cols]
    for cand in candidates:
        if cand in cols_lower:
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

    _coalesce_columns(df, NAME_COLS, "player")
    _coalesce_columns(df, TEAM_COLS, "team")
    _coalesce_columns(df, BYE_COLS,  "bye")

    df["pos"] = pos
    df["source_file"] = os.path.basename(path)

    pos_rank_col = _find_first_col(list(df.columns), POS_RANK_COLS)
    if pos_rank_col is None:
        df["pos_rank"] = range(1, len(df) + 1)
    else:
        if pos_rank_col != "pos_rank":
            df.rename(columns={pos_rank_col: "pos_rank"}, inplace=True)
        df["pos_rank"] = pd.to_numeric(df["pos_rank"], errors="coerce")

    overall_col = _find_first_col(list(df.columns), RANK_COLS)
    if overall_col is not None and overall_col != "overall":
        df.rename(columns={overall_col: "overall"}, inplace=True)
    if "overall" in df.columns:
        df["overall"] = pd.to_numeric(df["overall"], errors="coerce")

    if "player" not in df.columns:
        df.rename(columns={df.columns[0]: "player"}, inplace=True)
    if "team" not in df.columns:
        df["team"] = ""
    if "bye" not in df.columns:
        df["bye"] = ""

    df["player"] = df["player"].astype(str).str.strip()
    df["team"] = df["team"].astype(str).str.upper().str.strip()
    df["bye"] = df["bye"].astype(str).str.strip()

    df = df[df["player"].str.len() > 0].copy()
    return df

def load_all_positions(files: Dict[str, str]) -> pd.DataFrame:
    frames = []
    for pos, path in files.items():
        frames.append(normalize_one_position(path, pos))
    if not frames:
        return pd.DataFrame(columns=["player", "team", "bye", "pos", "pos_rank", "overall", "source_file"])  # empty
    all_df = pd.concat(frames, ignore_index=True)

    if all_df["overall"].notna().sum() < len(all_df) * 0.5:
        all_df["pos_count"] = all_df.groupby("pos")["player"].transform("count")
        all_df["pos_percentile"] = all_df["pos_rank"] / all_df["pos_count"]
        all_df["overall_approx"] = all_df["pos_percentile"]
    else:
        max_overall = all_df["overall"].max()
        all_df["overall_approx"] = all_df["overall"] / max_overall

    all_df["player_id"] = (
        all_df["player"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
        + "|" + all_df["pos"].str.upper()
    )

    all_df["pos_rank"] = pd.to_numeric(all_df["pos_rank"], errors="coerce")
    all_df.sort_values(["overall_approx", "pos", "pos_rank"], inplace=True, ignore_index=True)
    return all_df

# ----------------------------
# Session State & League Model
# ----------------------------
def init_state():
    if "players" not in st.session_state:
        files = discover_files()
        st.session_state.players = load_all_positions(files)
    if "drafted" not in st.session_state:
        st.session_state.drafted: List[dict] = []  # includes keepers with pick_no==0
    if "pick_no" not in st.session_state:
        st.session_state.pick_no = 1
    if "league" not in st.session_state:
        st.session_state.league = {
            "num_teams": 10,
            "teams": [
                {"slot": i+1, "team_name": f"Team {i+1}", "keeper_text": "", "keeper_pid": None}
                for i in range(10)
            ],
            "snake": True,
            "auto_assign_picker": False,
        }

init_state()
players: pd.DataFrame = st.session_state.players

def drafted_ids_set() -> set:
    return {d["player_id"] for d in st.session_state.drafted if d.get("player_id")}

# ----- Keeper matching & application -----
def parse_name_pos(s: str) -> Tuple[str, Optional[str]]:
    m = re.match(r"^(.*?)(\s*\((QB|RB|WR|TE|DST|K)\))?$", s.strip(), flags=re.I)
    if not m:
        return s.strip(), None
    name = m.group(1).strip()
    pos = (m.group(3) or None)
    return name, (pos.upper() if pos else None)

def find_available_player(name: str, pos: Optional[str]) -> Optional[pd.Series]:
    df = players.copy()
    df = df[~df["player_id"].isin(drafted_ids_set())]
    cand = df[df["player"].str.lower() == name.lower()]
    if pos:
        cand = cand[cand["pos"].str.upper() == pos]
    if cand.empty:
        cand = df[df["player"].str.lower().str.contains(name.lower())]
        if pos:
            cand = cand[cand["pos"].str.upper() == pos]
    if cand.empty:
        return None
    return cand.sort_values(["overall_approx", "pos_rank"]).iloc[0]

# ----- Draft mechanics -----
def is_drafted(player_id: str) -> bool:
    return any(p.get("player_id") == player_id for p in st.session_state.drafted)

def compute_picker_for_pick(pick_no: int) -> str:
    teams = sorted(st.session_state.league["teams"], key=lambda t: t["slot"])
    N = len(teams)
    if N == 0:
        return "Other Team"
    round_idx = (pick_no - 1) // N
    idx_in_round = (pick_no - 1) % N
    if st.session_state.league.get("snake", True) and (round_idx % 2 == 1):
        team = teams[::-1][idx_in_round]
    else:
        team = teams[idx_in_round]
    return team.get("team_name", f"Team {idx_in_round+1}")

def mark_drafted(row: pd.Series, picker: Optional[str] = None, method: str = "manual", pick_no: Optional[int] = None):
    pid = row["player_id"]
    if is_drafted(pid):
        return
    if pick_no is None:
        pick_no = st.session_state.pick_no
    if picker is None:
        picker = compute_picker_for_pick(pick_no) if st.session_state.league.get("auto_assign_picker", False) else "Other Team"
    st.session_state.drafted.append({
        "pick_no": pick_no,
        "player_id": pid,
        "player": row.get("player", ""),
        "team": row.get("team", ""),
        "pos": row.get("pos", ""),
        "picker": picker,
        "method": method,
    })
    if pick_no > 0:
        st.session_state.pick_no = max(st.session_state.pick_no, pick_no + 1)

def undo_last_pick():
    # Do not remove keeper placeholders or keeper picks at pick 0
    while st.session_state.drafted:
        last = st.session_state.drafted[-1]
        if last.get("pick_no", 0) == 0:
            st.session_state.drafted.pop()
            st.session_state.drafted.insert(0, last)  # keep keepers locked at top
            break
        st.session_state.drafted.pop()
        st.session_state.pick_no = max(1, last.get("pick_no", 1))
        break

# ----- Save / Load -----
def save_state_json() -> str:
    payload = {
        "drafted": st.session_state.drafted,
        "pick_no": st.session_state.pick_no,
        "league": st.session_state.league,
    }
    return json.dumps(payload, indent=2)

def load_state_json(text: str):
    data = json.loads(text)
    st.session_state.drafted = data.get("drafted", [])
    st.session_state.pick_no = int(data.get("pick_no", 1))
    st.session_state.league = data.get("league", st.session_state.league)
    st.success("State loaded.")

# ----------------------------
# Apply League Setup & Keepers
# ----------------------------
def apply_league_setup(teams_df: pd.DataFrame):
    # Update league teams
    teams = []
    for _, r in teams_df.iterrows():
        teams.append({
            "slot": int(r["slot"]),
            "team_name": str(r["team_name"]).strip() or f"Team {int(r['slot'])}",
            "keeper_text": str(r.get("keeper", "")).strip(),
            "keeper_pid": None,
        })
    teams.sort(key=lambda t: t["slot"])
    st.session_state.league["teams"] = teams

    # Clear any existing pick 0 rows
    st.session_state.drafted = [p for p in st.session_state.drafted if p.get("pick_no", 0) != 0]

    # Add fresh placeholders
    for t in teams:
        st.session_state.drafted.append({
            "pick_no": 0,
            "player_id": None,
            "player": "— TBD —" if not t["keeper_text"] else t["keeper_text"],
            "team": "",
            "pos": "",
            "picker": t["team_name"],
            "method": "keeper",
        })

    # Try to resolve and lock any provided keepers
    for t in teams:
        kt = t["keeper_text"].strip()
        if not kt:
            continue
        name, pos = parse_name_pos(kt)
        row = find_available_player(name, pos)
        if row is None:
            continue
        # assign to that team's pick 0 placeholder
        for p in st.session_state.drafted:
            if p.get("pick_no") == 0 and p.get("picker") == t["team_name"] and p.get("player_id") is None:
                p.update({
                    "player_id": row["player_id"],
                    "player": row["player"],
                    "team": row["team"],
                    "pos": row["pos"],
                })
                t["keeper_pid"] = row["player_id"]
                break

# ----------------------------
# UI — Sidebar
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.league["snake"] = st.toggle("Snake draft", value=st.session_state.league.get("snake", True))
    st.session_state.league["auto_assign_picker"] = st.toggle("Auto-assign picker by draft order", value=st.session_state.league.get("auto_assign_picker", False))

    st.divider()
    st.subheader("📥 Load / 💾 Save State")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("💾 Download JSON", data=save_state_json(), file_name="draft_state.json", mime="application/json")
    with c2:
        uploaded = st.file_uploader("Upload JSON", type=["json"], label_visibility="collapsed")
        if uploaded is not None:
            load_state_json(uploaded.read().decode("utf-8"))

    st.divider()
    st.subheader("🧹 Reset")
    if st.button("Reset ALL (keep CSVs)"):
        st.session_state.clear()
        st.rerun()

# ----------------------------
# UI — Main Tabs
# ----------------------------
st.title("🏈 Fantasy Draft Helper 2025")

T0, T1, T2, T3, T4 = st.tabs(["🏟️ League Setup", "⭐ Best Available", "📊 By Position", "🧾 Draft Board", "📌 Queue & Search"])

# ---------- Tab 0: League Setup ----------
with T0:
    st.subheader("League Setup (Teams, Draft Order, Keeper per team)")
    teams = st.session_state.league["teams"]
    df_init = pd.DataFrame({
        "slot": [t["slot"] for t in teams],
        "team_name": [t["team_name"] for t in teams],
        "keeper": [t["keeper_text"] for t in teams],
    })

    st.caption("Fill in team names and (optionally) each team's keeper as `Name` or `Name (POS)`.")
    edited = st.data_editor(
        df_init,
        num_rows="dynamic",
        use_container_width=True,
        height=420,
        column_config={
            "slot": st.column_config.NumberColumn("Draft Slot", min_value=1, max_value=24, step=1, help="1 = first pick"),
            "team_name": st.column_config.TextColumn("Team Name"),
            "keeper": st.column_config.TextColumn("Keeper (optional)")
        },
    )

    cA, cB, cC = st.columns([1,1,1])
    with cA:
        if st.button("Apply league & keepers"):
            apply_league_setup(edited)
            st.success("League updated. Keepers placed at Pick 0 placeholders. Any matched keepers are now locked and removed from availability.")
    with cB:
        if st.button("Clear keepers (placeholders stay)"):
            # Remove pick 0 rows and re-add placeholders without keeper
            for t in st.session_state.league["teams"]:
                t["keeper_text"] = ""
                t["keeper_pid"] = None
            st.session_state.drafted = [p for p in st.session_state.drafted if p.get("pick_no", 0) != 0]
            for t in sorted(st.session_state.league["teams"], key=lambda x: x["slot"]):
                st.session_state.drafted.append({
                    "pick_no": 0,
                    "player_id": None,
                    "player": "— TBD —",
                    "team": "",
                    "pos": "",
                    "picker": t["team_name"],
                    "method": "keeper",
                })
            st.success("Keepers cleared.")
    with cC:
        st.info("Note: Pick 0 rows are non-removable and always on top of the Draft Board.")

# ---------- Tab 1: Best Available ----------
with T1:
    st.subheader("Best Available (All Positions)")
    hide_drafted = st.toggle("Hide drafted", value=True, key="hide_drafted_all")
    limit = st.slider("Rows to show", min_value=25, max_value=400, step=25, value=100, key="limit_all")

    df = players.copy()
    dids = drafted_ids_set()
    if hide_drafted:
        df = df[~df["player_id"].isin(dids)]

    view = df[["player", "team", "pos", "pos_rank", "bye"]].head(limit).reset_index(drop=True)
    st.dataframe(view, use_container_width=True, height=520)

    st.caption("Matched keepers (Pick 0) are removed from this list automatically.")

# ---------- Tab 2: By Position ----------
with T2:
    st.subheader("By Position")
    pos_choice = st.segmented_control("Position", options=["ALL", "QB", "RB", "WR", "TE", "DST", "K"], default="RB")

    df = players.copy()
    if pos_choice != "ALL":
        df = df[df["pos"] == pos_choice]
    dids = drafted_ids_set()
    df = df[~df["player_id"].isin(dids)]

    view_df = df[["player", "team", "pos", "pos_rank", "bye"]].copy()
    view_df.insert(0, "Select", False)

    edited = st.data_editor(
        view_df,
        use_container_width=True,
        height=520,
        column_config={
            "Select": st.column_config.CheckboxColumn(help="Select players to draft")
        },
        disabled=["player", "team", "pos", "pos_rank", "bye"],
        hide_index=True,
    )

    to_draft_rows = []
    for _, r in edited.iterrows():
        if r.get("Select"):
            base = players[(players["player"] == r["player"]) & (players["pos"] == r["pos"])].head(1)
            if not base.empty:
                to_draft_rows.append(base.iloc[0])

    colA, colB = st.columns([1,1])
    with colA:
        picker = st.text_input("Picker label for selected rows (ignored if Auto-assign on)", value="Other Team", key="picker_bypos")
    with colB:
        if st.button("Draft selected players"):
            if not to_draft_rows:
                st.warning("No players selected.")
            else:
                cnt = 0
                for base_row in to_draft_rows:
                    mk_picker = None if st.session_state.league.get("auto_assign_picker", False) else picker
                    mark_drafted(base_row, picker=mk_picker, method="manual")
                    cnt += 1
                st.success(f"Drafted {cnt} player(s).")

# ---------- Tab 3: Draft Board ----------
with T3:
    st.subheader("Draft Board")

    # Draft order panel
    teams_sorted = sorted(st.session_state.league["teams"], key=lambda t: t["slot"])
    st.markdown("**Draft Order (Slot → Team):** " + ", ".join([f"{t['slot']}: {t['team_name']}" for t in teams_sorted]))
    st.caption("Auto-assign picker uses this order (snake if enabled). Keepers are locked at Pick 0 and listed below.")

    if not st.session_state.drafted:
        st.info("No picks yet. Use the other tabs to draft players.")
    else:
        board = pd.DataFrame(st.session_state.drafted).sort_values(["pick_no", "picker"]).reset_index(drop=True)
        board["Pick"] = board["pick_no"]
        board_display = board[["Pick", "player", "team", "pos", "picker", "method"]]
        st.dataframe(board_display, use_container_width=True, height=520)

    d1, d2, d3 = st.columns([1,1,1])
    with d1:
        if st.button("↩️ Undo last pick"):
            undo_last_pick()
            st.info("Last non-keeper pick undone.")
    with d2:
        csv = pd.DataFrame(st.session_state.drafted).to_csv(index=False)
        st.download_button("Download board CSV", data=csv, file_name="draft_board.csv", mime="text/csv")
    with d3:
        qname = st.text_input("Quick draft by name (opt. 'Name (POS)')", key="quick_name")
        if st.button("➕ Draft by name") and qname.strip():
            name, pos = parse_name_pos(qname)
            row = find_available_player(name, pos)
            if row is None:
                st.warning("No matching available player found.")
            else:
                mk_picker = None if st.session_state.league.get("auto_assign_picker", False) else st.text_input("Picker override", value="Other Team", key="picker_override")
                mark_drafted(row, picker=mk_picker, method="manual")
                st.success(f"Drafted: {row['player']} ({row['pos']})")

# ---------- Tab 4: Queue & Search ----------
with T4:
    st.subheader("Queue & Search")
    q1, q2 = st.columns([2,1])
    with q1:
        q = st.text_input("Search players (name, team, or pos contains)", placeholder="e.g., 'bengals wr' or 'gibbs'")
    with q2:
        only_available = st.toggle("Only available", value=True)

    df = players.copy()
    if q.strip():
        tokens = q.lower().split()
        mask = pd.Series([True] * len(df))
        for t in tokens:
            mask = mask & (
                df["player"].str.lower().str.contains(t) |
                df["team"].str.lower().str.contains(t) |
                df["pos"].str.lower().str.contains(t)
            )
        df = df[mask]

    if only_available:
        df = df[~df["player_id"].isin(drafted_ids_set())]

    st.dataframe(df[["player", "team", "pos", "pos_rank", "bye"]].head(400).reset_index(drop=True), use_container_width=True, height=520)

# Footer
st.write("\n")
st.caption("League-aware. Keepers locked at Pick 0. Use auto-assign picker for rapid-fire drafting.")
