# Fantasy Draft Helper — Streamlit app (Grid Draft Board + League Setup)
# -------------------------------------------------------------------
# New in this version:
# • League Setup tab to enter team names (10 teams), draft order (slots 1..10),
#   and an optional keeper for each team (one keeper per team).
# • Keepers appear on a dedicated **Row 0** at the top of the Draft Board grid.
# • Draft Board now renders like a real sticker board: columns are team names in
#   draft order, rows are rounds (1..N). Snake draft supported.
# • Each grid cell shows ONLY the player name.
# • Best Available view hides keepers and drafted players.
# • Quick draft by name, undo, and save/load state JSON.
#
# Usage:
#   - Place CSVs in repo root or ./data:
#       QB_RANKINGS.csv, RB_RANKINGS.csv, WR_RANKINGS.csv,
#       TE_RANKINGS.csv, DST_RANKINGS.csv, K_RANKINGS.csv
#   - pip install streamlit pandas
#   - streamlit run app.py

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
st.set_page_config(page_title="Fantasy Draft Helper 2025", page_icon="🏈", layout="wide")

CSS = """
<style>
  .block-container {padding-top: 1.0rem;}
  .stDataFrame td, .stDataFrame th {font-size: 0.95rem;}
  .small-note {color: #7a7a7a; font-size: 0.88rem;}
  .keeper {background: rgba(76,175,80,0.10);} /* light green */
  .board {border:1px solid #e5e7eb; border-radius: 8px;}
</style>
"""
sto = st.markdown(CSS, unsafe_allow_html=True)

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


def _coalesce(df: pd.DataFrame, wanted: List[str], new_name: str) -> Optional[str]:
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


def normalize_one(path: str, pos: str) -> pd.DataFrame:
    raw = _safe_read_csv(path)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["player","team","bye","pos","pos_rank","overall","player_id"])  # empty

    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]

    _coalesce(df, NAME_COLS, "player")
    _coalesce(df, TEAM_COLS, "team")
    _coalesce(df, BYE_COLS,  "bye")

    pos_rank_col = _find_first_col(list(df.columns), POS_RANK_COLS)
    if pos_rank_col is None:
        df["pos_rank"] = range(1, len(df)+1)
    else:
        if pos_rank_col != "pos_rank":
            df.rename(columns={pos_rank_col: "pos_rank"}, inplace=True)
        df["pos_rank"] = pd.to_numeric(df["pos_rank"], errors="coerce")

    overall_col = _find_first_col(list(df.columns), RANK_COLS)
    if overall_col is not None and overall_col != "overall":
        df.rename(columns={overall_col: "overall"}, inplace=True)
    if "overall" in df.columns:
        df["overall"] = pd.to_numeric(df["overall"], errors="coerce")

    df["pos"] = pos
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
    df["player_id"] = df["player"].str.lower().str.replace(r"\s+"," ", regex=True).str.strip() + "|" + df["pos"].str.upper()

    return df[["player","team","bye","pos","pos_rank","overall","player_id"]]


def load_all_positions(files: Dict[str,str]) -> pd.DataFrame:
    frames = []
    for pos, path in files.items():
        frames.append(normalize_one(path, pos))
    if not frames:
        return pd.DataFrame(columns=["player","team","bye","pos","pos_rank","overall","player_id"])  # empty
    all_df = pd.concat(frames, ignore_index=True)
    # Sort: prefer explicit overall if present, else pos_rank
    all_df.sort_values(["overall","pos_rank"], inplace=True, na_position="last")
    return all_df.reset_index(drop=True)

# ----------------------------
# Session state & league model
# ----------------------------
if "players" not in st.session_state:
    st.session_state.players = load_all_positions(discover_files())
if "drafted" not in st.session_state:
    st.session_state.drafted: List[dict] = []  # includes keeper pick_no==0 rows
if "league" not in st.session_state:
    st.session_state.league = {
        "num_teams": 10,
        "teams": [
            {"slot": i+1, "team_name": f"Team {i+1}", "keeper_text": "", "keeper_pid": None}
            for i in range(10)
        ],
        "snake": True,
        "total_rounds": 16,
    }
if "next_pick" not in st.session_state:
    st.session_state.next_pick = 1  # next overall pick number (>=1)

players: pd.DataFrame = st.session_state.players

# ----------------------------
# Utility functions
# ----------------------------

def drafted_ids_set() -> set:
    return {d["player_id"] for d in st.session_state.drafted if d.get("player_id")}


def parse_name_pos(s: str) -> Tuple[str, Optional[str]]:
    m = re.match(r"^(.*?)(\s*\((QB|RB|WR|TE|DST|K)\))?$", s.strip(), flags=re.I)
    if not m:
        return s.strip(), None
    name = m.group(1).strip()
    pos = (m.group(3) or None)
    return name, (pos.upper() if pos else None)


def find_available_player(name: str, pos: Optional[str]) -> Optional[pd.Series]:
    df = players[~players["player_id"].isin(drafted_ids_set())]
    cand = df[df["player"].str.lower() == name.lower()]
    if pos:
        cand = cand[cand["pos"].str.upper() == pos]
    if cand.empty:
        cand = df[df["player"].str.lower().str.contains(name.lower())]
        if pos:
            cand = cand[cand["pos"].str.upper() == pos]
    if cand.empty:
        return None
    return cand.sort_values(["overall","pos_rank"]).iloc[0]


def team_order() -> List[dict]:
    return sorted(st.session_state.league["teams"], key=lambda t: t["slot"])[: st.session_state.league["num_teams"]]


def team_name_by_pick(pick_no: int) -> str:
    order = team_order()
    n = len(order)
    rnd = (pick_no - 1) // n  # 0-index round
    idx = (pick_no - 1) % n
    if st.session_state.league.get("snake", True) and (rnd % 2 == 1):
        return order[::-1][idx]["team_name"]
    return order[idx]["team_name"]


def mark_drafted(row: pd.Series, pick_no: Optional[int] = None):
    pid = row["player_id"]
    if pid in drafted_ids_set():
        return
    if pick_no is None:
        pick_no = st.session_state.next_pick
    st.session_state.drafted.append({
        "pick_no": pick_no,
        "player_id": pid,
        "player": row["player"],
        "team": row.get("team", ""),
        "pos": row.get("pos", ""),
        "picker": team_name_by_pick(pick_no),
        "method": "manual",
    })
    if pick_no >= st.session_state.next_pick:
        st.session_state.next_pick = pick_no + 1


def undo_last_pick():
    # Remove last non-keeper pick (pick_no >= 1)
    picks = [p for p in st.session_state.drafted if p.get("pick_no", 0) >= 1]
    if not picks:
        return
    last_pick_no = max(p["pick_no"] for p in picks)
    st.session_state.drafted = [p for p in st.session_state.drafted if not (p.get("pick_no") == last_pick_no)]
    st.session_state.next_pick = min(st.session_state.next_pick, last_pick_no)


def save_state_json() -> str:
    payload = {
        "drafted": st.session_state.drafted,
        "league": st.session_state.league,
        "next_pick": st.session_state.next_pick,
    }
    return json.dumps(payload, indent=2)


def load_state_json(text: str):
    data = json.loads(text)
    st.session_state.drafted = data.get("drafted", [])
    st.session_state.league = data.get("league", st.session_state.league)
    st.session_state.next_pick = int(data.get("next_pick", 1))
    st.success("State loaded.")

# ----------------------------
# Keeper logic
# ----------------------------

def apply_league_setup(df_input: pd.DataFrame):
    # Update league state from edited dataframe
    teams: List[dict] = []
    for _, r in df_input.iterrows():
        teams.append({
            "slot": int(r["slot"]),
            "team_name": str(r["team_name"]).strip() or f"Team {int(r['slot'])}",
            "keeper_text": str(r.get("keeper", "")).strip(),
            "keeper_pid": None,
        })
    teams.sort(key=lambda t: t["slot"])  # ensure order
    st.session_state.league["teams"] = teams

    # Rebuild pick 0 rows: remove existing pick 0s
    st.session_state.drafted = [p for p in st.session_state.drafted if p.get("pick_no", 1) != 0]

    # Add pick 0 rows (placeholders)
    for t in teams:
        st.session_state.drafted.append({
            "pick_no": 0,
            "player_id": None,
            "player": "—" if not t["keeper_text"] else t["keeper_text"],
            "team": "",
            "pos": "",
            "picker": t["team_name"],
            "method": "keeper",
        })

    # Try to resolve actual keeper players; mark as taken at pick 0
    for t in teams:
        kt = t["keeper_text"].strip()
        if not kt:
            continue
        name, pos = parse_name_pos(kt)
        row = find_available_player(name, pos)
        if row is None:
            continue
        t["keeper_pid"] = row["player_id"]
        # update placeholder row matching this team
        for p in st.session_state.drafted:
            if p.get("pick_no") == 0 and p.get("picker") == t["team_name"]:
                p.update({
                    "player_id": row["player_id"],
                    "player": row["player"],
                    "team": row.get("team", ""),
                    "pos": row.get("pos", ""),
                })
                break

# ----------------------------
# Draft board grid rendering
# ----------------------------

def build_draft_grid(total_rounds: int) -> pd.DataFrame:
    order = team_order()
    columns = [t["team_name"] for t in order]
    n = len(columns)

    # Row 0: keepers row (strings)
    keepers = ["" for _ in range(n)]
    for i, t in enumerate(order):
        # find pick 0 entry for this team
        p0 = None
        for p in st.session_state.drafted:
            if p.get("pick_no") == 0 and p.get("picker") == t["team_name"]:
                p0 = p
                break
        keepers[i] = p0["player"] if (p0 and p0.get("player")) else "—"

    data = {col: [keepers[idx]] for idx, col in enumerate(columns)}

    # Build a map from (round_idx, col_idx) -> player name
    # round_idx: 1..total_rounds (row 1 is Round 1)
    grid = [["" for _ in range(n)] for _ in range(total_rounds)]

    picks = [p for p in st.session_state.drafted if p.get("pick_no", 0) >= 1]
    if picks:
        by_pick = {p["pick_no"]: p for p in picks}
        max_pick = max(by_pick.keys())
        for pick_no in range(1, max_pick + 1):
            if pick_no not in by_pick:
                continue
            # Determine round and team column for this pick
            rnd = (pick_no - 1) // n + 1  # 1-indexed round
            idx = (pick_no - 1) % n       # 0..n-1 position in round
            if rnd > total_rounds:
                break
            # Column index depends on snake
            if st.session_state.league.get("snake", True) and (rnd % 2 == 0):
                col_idx = n - 1 - idx
            else:
                col_idx = idx
            player_name = by_pick[pick_no].get("player", "")
            grid[rnd - 1][col_idx] = player_name

    # Fill rows into data dict
    for r in range(total_rounds):
        for j, col in enumerate(columns):
            data[col].append(grid[r][j])

    # Row labels
    index = ["Row 0 (Keepers)"] + [f"Round {i}" for i in range(1, total_rounds + 1)]
    df = pd.DataFrame(data, index=index)
    return df

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.league["snake"] = st.toggle("Snake draft", value=st.session_state.league.get("snake", True))
    st.session_state.league["total_rounds"] = st.number_input("Total rounds", min_value=5, max_value=30, step=1, value=st.session_state.league.get("total_rounds", 16))

    st.divider()
    st.subheader("📥 Load / 💾 Save")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("💾 Download JSON", data=save_state_json(), file_name="draft_state.json", mime="application/json")
    with c2:
        up = st.file_uploader("Upload JSON", type=["json"], label_visibility="collapsed")
        if up is not None:
            load_state_json(up.read().decode("utf-8"))

    st.divider()
    if st.button("↩️ Undo last pick"):
        undo_last_pick()
        st.info("Last pick undone.")

# ----------------------------
# Main UI
# ----------------------------
st.title("🏈 Fantasy Draft Helper 2025")

T0, T1, T2 = st.tabs(["🏟️ League Setup", "⭐ Best Available", "📋 Draft Board (Grid)"])

# ---------- Tab 0: League Setup ----------
with T0:
    st.subheader("League Setup (Team names, Draft order, Keeper per team)")
    order = team_order()
    init_df = pd.DataFrame({
        "slot": [t["slot"] for t in order],
        "team_name": [t["team_name"] for t in order],
        "keeper": [t["keeper_text"] for t in order],
    })
    st.caption("Enter team names and optional keepers as `Name` or `Name (POS)`. Click **Apply** to lock them to Row 0 and remove matched players from availability.")
    edited = st.data_editor(
        init_df,
        use_container_width=True,
        height=420,
        column_config={
            "slot": st.column_config.NumberColumn("Draft Slot", min_value=1, max_value=24, step=1),
            "team_name": st.column_config.TextColumn("Team Name"),
            "keeper": st.column_config.TextColumn("Keeper (optional)")
        },
    )

    cA, cB = st.columns([1,1])
    with cA:
        if st.button("Apply league & keepers"):
            apply_league_setup(edited)
            st.success("League updated. Keepers placed in Row 0 and removed from Best Available.")
    with cB:
        if st.button("Clear keepers (keep placeholders)"):
            # wipe pick 0 rows and re-add empty placeholders
            for t in st.session_state.league["teams"]:
                t["keeper_text"] = ""
                t["keeper_pid"] = None
            st.session_state.drafted = [p for p in st.session_state.drafted if p.get("pick_no", 1) != 0]
            for t in team_order():
                st.session_state.drafted.append({
                    "pick_no": 0, "player_id": None, "player": "—", "team": "", "pos": "",
                    "picker": t["team_name"], "method": "keeper"
                })
            st.info("Keepers cleared.")

# ---------- Tab 1: Best Available ----------
with T1:
    st.subheader("Best Available (All Positions)")
    hide_drafted = True
    df = players.copy()
    if hide_drafted:
        df = df[~df["player_id"].isin(drafted_ids_set())]
    st.dataframe(df[["player","team","pos","pos_rank","bye"]].head(200).reset_index(drop=True), use_container_width=True, height=520)

    st.markdown("**Quick draft by name**")
    c1, c2 = st.columns([2,1])
    with c1:
        qname = st.text_input("Name (optional 'Name (POS)')", key="qd_name")
    with c2:
        if st.button("➕ Draft next pick") and qname.strip():
            name, pos = parse_name_pos(qname)
            row = find_available_player(name, pos)
            if row is None:
                st.warning("No matching available player.")
            else:
                mark_drafted(row)
                st.success(f"Drafted: {row['player']}")

# ---------- Tab 2: Draft Board (Grid) ----------
with T2:
    st.subheader("Draft Board (Sticker Grid)")
    st.caption("Columns are draft order. Row 0 shows keepers. Each cell = player name only. Snake drafting applied to rows.")

    grid_df = build_draft_grid(total_rounds=st.session_state.league.get("total_rounds", 16))
    st.dataframe(grid_df, use_container_width=True, height=600, hide_index=False)

    st.markdown("**Manual pick controls**")
    c1, c2 = st.columns([2,1])
    with c1:
        manual_name = st.text_input("Draft by name (optional 'Name (POS)')", key="man_name")
    with c2:
        next_pick = st.number_input("Pick #", min_value=1, step=1, value=st.session_state.next_pick)
    c3, c4 = st.columns([1,1])
    with c3:
        if st.button("Draft at Pick #") and manual_name.strip():
            name, pos = parse_name_pos(manual_name)
            row = find_available_player(name, pos)
            if row is None:
                st.warning("No matching available player.")
            else:
                mark_drafted(row, pick_no=int(next_pick))
                st.success(f"Drafted at Pick {int(next_pick)}: {row['player']}")
    with c4:
        if st.button("↩️ Undo last pick (here)"):
            undo_last_pick()
            st.info("Last pick undone.")

# Footer
st.write("\n")
st.caption("Grid board with Row 0 keepers. Save your state JSON between sessions. Happy drafting!")
