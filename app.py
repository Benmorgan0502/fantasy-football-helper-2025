# Fantasy Draft Helper — FULL Streamlit App (Sticker-Grid Board + League Setup + Keepers + Best Available + By Position)
# -------------------------------------------------------------------------------------------------
# This file unifies all features requested across the thread into a single, clean app.
#
# ✅ Features
#   • Load 2025 CSV rankings (QB/RB/WR/TE/DST/K) and normalize columns
#   • League Setup tab: enter team names, draft order (slots 1..N), and one keeper per team
#   • Draft Board renders like a real sticker board (columns = team names, Row 0 = keepers, then rounds)
#       - Snake draft supported for mapping picks → columns per round
#       - Cells show ONLY the player name (no team or position)
#   • Best Available view (cross-position), hides drafted + keepers automatically
#       - Sorting uses explicit Overall if present, else positional percentile fallback
#       - Quick Draft by Name (supports "Name (POS)") drafts at the next pick
#   • By Position view with multi-select and "Draft selected" actions
#   • Undo (last non-keeper pick), Save/Load state JSON, Total rounds control
#
# 🔧 Usage
#   1) Put CSVs in repo root or ./data:
#       QB_RANKINGS.csv, RB_RANKINGS.csv, WR_RANKINGS.csv,
#       TE_RANKINGS.csv, DST_RANKINGS.csv, K_RANKINGS.csv
#   2) pip install streamlit pandas
#   3) streamlit run app.py
#
# Notes
#   - If your CSVs have different column names, the loader auto-detects common variants.
#   - Keepers live on Row 0 (one per team); keepers that match real players are considered drafted
#     and removed from availability; blank placeholders show "—" until you set them.
#   - "Draft at Pick #" and "Draft next pick" both honor snake mapping.

from __future__ import annotations
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# -------------------------------------
# Page setup
# -------------------------------------
st.set_page_config(page_title="Fantasy Draft Helper 2025", page_icon="🏈", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.0rem;}
      .stDataFrame td, .stDataFrame th {font-size: 0.95rem;}
      .small-note {color: #7a7a7a; font-size: 0.88rem;}
      .board {border:1px solid #e5e7eb; border-radius: 8px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------
# File discovery & normalization
# -------------------------------------
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

    # Cross-position sort key: prefer explicit overall; otherwise fallback to positional percentile
    if all_df["overall"].notna().sum() < len(all_df) * 0.5:
        all_df["pos_count"] = all_df.groupby("pos")["player"].transform("count")
        all_df["pos_percentile"] = all_df["pos_rank"] / all_df["pos_count"].replace(0, 1)
        all_df["overall_approx"] = all_df["pos_percentile"]
    else:
        max_overall = all_df["overall"].max()
        if pd.isna(max_overall) or max_overall == 0:
            all_df["overall_approx"] = all_df["pos_rank"]  # fallback
        else:
            all_df["overall_approx"] = all_df["overall"] / max_overall

    all_df.sort_values(["overall_approx", "pos", "pos_rank"], inplace=True, ignore_index=True)
    return all_df

# -------------------------------------
# Session state
# -------------------------------------
if "players" not in st.session_state:
    st.session_state.players = load_all_positions(discover_files())
if "drafted" not in st.session_state:
    st.session_state.drafted: List[dict] = []  # includes pick_no==0 keeper rows
if "league" not in st.session_state:
    st.session_state.league = {
        "teams": [
            {"slot": i+1, "team_name": f"Team {i+1}", "keeper_text": "", "keeper_pid": None}
            for i in range(10)
        ],
        "snake": True,
        "total_rounds": 16,
    }
if "next_pick" not in st.session_state:
    st.session_state.next_pick = 1

players: pd.DataFrame = st.session_state.players

# -------------------------------------
# Helpers
# -------------------------------------

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
    return cand.sort_values(["overall_approx", "pos_rank"]).iloc[0]


def team_order() -> List[dict]:
    teams = st.session_state.league.get("teams", [])
    teams = [t for t in teams if str(t.get("team_name","")) != ""]
    teams.sort(key=lambda t: t["slot"])
    return teams


def team_name_by_pick(pick_no: int) -> str:
    order = team_order()
    n = max(1, len(order))
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
    # Remove last pick with pick_no >= 1 (do not remove keepers at pick 0)
    picks = [p for p in st.session_state.drafted if p.get("pick_no", 0) >= 1]
    if not picks:
        return
    last_pick_no = max(p["pick_no"] for p in picks)
    st.session_state.drafted = [p for p in st.session_state.drafted if p.get("pick_no") != last_pick_no]
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

# -------------------------------------
# League Setup & Keepers
# -------------------------------------

def apply_league_setup(df_input: pd.DataFrame):
    # Update teams from editor
    teams: List[dict] = []
    for _, r in df_input.iterrows():
        slot_val = int(r["slot"]) if pd.notna(r["slot"]) else 1
        team_nm = str(r.get("team_name", "")).strip() or f"Team {slot_val}"
        keeper_text = str(r.get("keeper", "")).strip()
        teams.append({
            "slot": slot_val,
            "team_name": team_nm,
            "keeper_text": keeper_text,
            "keeper_pid": None,
        })
    teams.sort(key=lambda t: t["slot"])  # ensure draft order
    st.session_state.league["teams"] = teams

    # Remove any existing pick 0 rows and rebuild placeholders
    st.session_state.drafted = [p for p in st.session_state.drafted if p.get("pick_no", 1) != 0]
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

    # Try to resolve any provided keeper names to actual players
    for t in teams:
        kt = t["keeper_text"].strip()
        if not kt:
            continue
        name, pos = parse_name_pos(kt)
        row = find_available_player(name, pos)
        if row is None:
            continue
        t["keeper_pid"] = row["player_id"]
        # update placeholder row for this team
        for p in st.session_state.drafted:
            if p.get("pick_no") == 0 and p.get("picker") == t["team_name"]:
                p.update({
                    "player_id": row["player_id"],
                    "player": row["player"],
                    "team": row.get("team", ""),
                    "pos": row.get("pos", ""),
                })
                break

# -------------------------------------
# Draft board grid
# -------------------------------------

def build_draft_grid(total_rounds: int) -> pd.DataFrame:
    order = team_order()
    columns = [t["team_name"] for t in order]
    n = len(columns) if len(columns) > 0 else 1

    # Row 0 (keepers)
    keepers_row = []
    for team in order:
        p0 = next((p for p in st.session_state.drafted if p.get("pick_no") == 0 and p.get("picker") == team["team_name"]), None)
        keepers_row.append(p0.get("player", "—") if p0 else "—")

    # Initialize data dict with keepers row as first element for each column
    data = {col: [keepers_row[idx]] for idx, col in enumerate(columns)} if columns else {"Team 1": ["—"]}

    # Fill round rows
    total_rounds = int(total_rounds)
    grid = [["" for _ in range(n)] for _ in range(total_rounds)]

    # Map picks to cells
    picks = [p for p in st.session_state.drafted if p.get("pick_no", 0) >= 1]
    if picks:
        by_pick = {p["pick_no"]: p for p in picks}
        max_pick = max(by_pick.keys())
        for pick_no in range(1, max_pick + 1):
            if pick_no not in by_pick:
                continue
            rnd = (pick_no - 1) // n + 1  # 1-indexed
            idx = (pick_no - 1) % n       # 0..n-1 in the round
            if rnd > total_rounds:
                break
            if st.session_state.league.get("snake", True) and (rnd % 2 == 0):
                col_idx = n - 1 - idx
            else:
                col_idx = idx
            player_name = by_pick[pick_no].get("player", "")
            grid[rnd - 1][col_idx] = player_name

    # Append rounds to data
    if columns:
        for r in range(total_rounds):
            for j, col in enumerate(columns):
                data[col].append(grid[r][j])
    else:
        # Single default column if no teams defined yet
        col = "Team 1"
        for r in range(total_rounds):
            data[col].append(grid[r][0])

    index = ["Row 0 (Keepers)"] + [f"Round {i}" for i in range(1, total_rounds + 1)]
    return pd.DataFrame(data, index=index)

# -------------------------------------
# Sidebar
# -------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.league["snake"] = st.toggle("Snake draft", value=st.session_state.league.get("snake", True))
    st.session_state.league["total_rounds"] = st.number_input(
        "Total rounds", min_value=5, max_value=30, step=1, value=st.session_state.league.get("total_rounds", 16)
    )

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

    st.caption("Tip: Save JSON after each round in case you need to restore.")

# -------------------------------------
# Main UI Tabs
# -------------------------------------
st.title("🏈 Fantasy Draft Helper 2025")

T0, T1, T2, T3 = st.tabs(["🏟️ League Setup", "⭐ Best Available", "📊 By Position", "📋 Draft Board (Grid)"])

# ---------- Tab 0: League Setup ----------
with T0:
    st.subheader("League Setup (Team names, Draft order, Keeper per team)")
    order = team_order()
    # If no teams yet, use current league config
    if not order:
        order = st.session_state.league["teams"]
    init_df = pd.DataFrame({
        "slot": [t.get("slot", i+1) for i, t in enumerate(order)],
        "team_name": [t.get("team_name", f"Team {i+1}") for i, t in enumerate(order)],
        "keeper": [t.get("keeper_text", "") for t in order],
    })

    st.caption("Enter team names and optional keepers as `Name` or `Name (POS)`. Click **Apply** to lock Row 0 and remove matched players from availability.")
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
            # Wipe pick 0 rows and re-add empty placeholders
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

    hide_drafted = st.toggle("Hide drafted", value=True, key="hide_drafted_all")
    limit = st.slider("Rows to show", min_value=25, max_value=400, step=25, value=100, key="limit_all")

    df = players.copy()
    if hide_drafted:
        df = df[~df["player_id"].isin(drafted_ids_set())]

    view = df[["player", "team", "pos", "pos_rank", "bye"]].head(limit).reset_index(drop=True)
    st.dataframe(view, use_container_width=True, height=520)

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

# ---------- Tab 2: By Position ----------
with T2:
    st.subheader("By Position")

    dids = drafted_ids_set()
    df = players[~players["player_id"].isin(dids)].copy()

    pos_choice = st.selectbox("Position", options=["ALL", "QB", "RB", "WR", "TE", "DST", "K"], index=2)
    if pos_choice != "ALL":
        df = df[df["pos"] == pos_choice]

    # Select & draft multiple
    view_df = df[["player", "team", "pos", "pos_rank", "bye"]].copy()
    view_df.insert(0, "Select", False)

    edited = st.data_editor(
        view_df,
        use_container_width=True,
        height=520,
        column_config={
            "Select": st.column_config.CheckboxColumn(help="Select players to draft"),
        },
        disabled=["player", "team", "pos", "pos_rank", "bye"],
        hide_index=True,
    )

    to_draft_rows: List[pd.Series] = []
    for _, r in edited.iterrows():
        if r.get("Select"):
            base = players[(players["player"] == r["player"]) & (players["pos"] == r["pos"])].head(1)
            if not base.empty and base.iloc[0]["player_id"] not in dids:
                to_draft_rows.append(base.iloc[0])

    if st.button("Draft selected"):
        if not to_draft_rows:
            st.warning("No players selected.")
        else:
            cnt = 0
            for base_row in to_draft_rows:
                mark_drafted(base_row)
                cnt += 1
            st.success(f"Drafted {cnt} player(s).")

# ---------- Tab 3: Draft Board (Grid) ----------
with T3:
    st.subheader("Draft Board (Sticker Grid)")
    st.caption("Columns are draft order. Row 0 shows keepers. Each cell = player name only. Snake drafting applied per round.")

    grid_df = build_draft_grid(total_rounds=st.session_state.league.get("total_rounds", 16))
    st.dataframe(grid_df, use_container_width=True, height=620, hide_index=False)

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
st.caption("All-in-one: League setup, Row 0 keepers, sticker-grid draft board, best available, and by-position drafting. Save often and crush your draft.")
