# Fantasy Draft Helper — Streamlit app (Draft Board as Grid)
# ---------------------------------------------------
# Updates:
# - Draft Board now renders like an actual draft board:
#   • Team names as column headers
#   • Row 0 shows keepers
#   • Subsequent rows fill with drafted players by round and slot
#   • Cells show just player names (no team/pos)

import os
import re
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Fantasy Draft Helper 2025", page_icon="🏈", layout="wide")

NAME_COLS = ["player", "name", "player name"]
TEAM_COLS = ["team", "tm"]
BYE_COLS = ["bye", "bye week"]
RANK_COLS = ["overall", "rank", "ecr", "adp"]
POS_RANK_COLS = ["pos_rank", "position rank"]

POS_FILES = {
    "QB": "QB_RANKINGS.csv",
    "RB": "RB_RANKINGS.csv",
    "WR": "WR_RANKINGS.csv",
    "TE": "TE_RANKINGS.csv",
    "DST": "DST_RANKINGS.csv",
    "K": "K_RANKINGS.csv",
}
SEARCH_DIRS = [".", "./data"]

# ----------------------------
# Helpers to load rankings
# ----------------------------
def _find_col(cols, candidates):
    cols_lower = [c.lower().strip() for c in cols]
    for cand in candidates:
        if cand in cols_lower:
            return cols[cols_lower.index(cand)]
    return None

def normalize(path, pos):
    try:
        df = pd.read_csv(path)
    except:
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]

    name_col = _find_col(df.columns, NAME_COLS)
    if name_col:
        df.rename(columns={name_col: "player"}, inplace=True)
    else:
        df.rename(columns={df.columns[0]: "player"}, inplace=True)

    team_col = _find_col(df.columns, TEAM_COLS)
    if team_col:
        df.rename(columns={team_col: "team"}, inplace=True)
    else:
        df["team"] = ""

    bye_col = _find_col(df.columns, BYE_COLS)
    if bye_col:
        df.rename(columns={bye_col: "bye"}, inplace=True)
    else:
        df["bye"] = ""

    pos_rank_col = _find_col(df.columns, POS_RANK_COLS)
    if pos_rank_col:
        df.rename(columns={pos_rank_col: "pos_rank"}, inplace=True)
        df["pos_rank"] = pd.to_numeric(df["pos_rank"], errors="coerce")
    else:
        df["pos_rank"] = range(1, len(df) + 1)

    overall_col = _find_col(df.columns, RANK_COLS)
    if overall_col:
        df.rename(columns={overall_col: "overall"}, inplace=True)
        df["overall"] = pd.to_numeric(df["overall"], errors="coerce")

    df["pos"] = pos
    df["player"] = df["player"].astype(str).str.strip()
    df["player_id"] = df["player"].str.lower() + "|" + df["pos"]

    return df[["player", "team", "bye", "pos", "pos_rank", "overall", "player_id"]]

def load_all():
    frames = []
    for pos, fname in POS_FILES.items():
        for d in SEARCH_DIRS:
            path = os.path.join(d, fname)
            if os.path.exists(path):
                frames.append(normalize(path, pos))
                break
    if not frames:
        return pd.DataFrame(columns=["player", "team", "bye", "pos", "pos_rank", "overall", "player_id"])
    all_df = pd.concat(frames, ignore_index=True)
    all_df.sort_values(["overall", "pos_rank"], inplace=True, na_position="last")
    return all_df.reset_index(drop=True)

# ----------------------------
# Session state init
# ----------------------------
if "players" not in st.session_state:
    st.session_state.players = load_all()
if "drafted" not in st.session_state:
    st.session_state.drafted = []
if "teams" not in st.session_state:
    st.session_state.teams = [f"Team {i+1}" for i in range(10)]
if "draft_order" not in st.session_state:
    st.session_state.draft_order = [i for i in range(10)]
if "keepers" not in st.session_state:
    st.session_state.keepers = {t: None for t in st.session_state.teams}

# ----------------------------
# Functions
# ----------------------------
def is_drafted(pid):
    return any(p["player_id"] == pid for p in st.session_state.drafted)

def mark_drafted(row, picker, method="manual", pick_no=None):
    if is_drafted(row["player_id"]):
        return
    entry = {
        "pick_no": pick_no if pick_no is not None else len(st.session_state.drafted)+1,
        "player_id": row["player_id"],
        "player": row["player"],
        "team": row["team"],
        "pos": row["pos"],
        "picker": picker,
        "method": method,
    }
    st.session_state.drafted.append(entry)

def build_draft_grid():
    num_teams = len(st.session_state.teams)
    # Build keeper row
    keeper_row = []
    for team in st.session_state.teams:
        keeper = st.session_state.keepers.get(team)
        keeper_row.append(keeper if keeper else "—")

    # Build draft rows
    rounds = (len(st.session_state.drafted)//num_teams)+1
    grid = []
    grid.append(keeper_row)
    picks_by_team = {t: [] for t in st.session_state.teams}
    for pick in st.session_state.drafted:
        if pick["pick_no"] == 0:
            continue
        picks_by_team[pick["picker"]].append(pick["player"])
    for r in range(rounds):
        row = []
        for team in st.session_state.teams:
            row.append(picks_by_team[team][r] if r < len(picks_by_team[team]) else "")
        grid.append(row)
    return pd.DataFrame(grid, columns=st.session_state.teams)

# ----------------------------
# Tabs
# ----------------------------
T1, T2, T3 = st.tabs(["⚙️ League Setup", "⭐ Best Available", "🧾 Draft Board"])

with T1:
    st.subheader("League Setup")
    team_names = []
    for i in range(10):
        c1, c2, c3 = st.columns([2,1,2])
        with c1:
            name = st.text_input(f"Team {i+1} name", value=st.session_state.teams[i], key=f"team_{i}")
            team_names.append(name)
        with c2:
            order = st.number_input("Draft slot", min_value=1, max_value=10, value=i+1, key=f"order_{i}")
            st.session_state.draft_order[i] = order
        with c3:
            keeper = st.text_input("Keeper name", value=st.session_state.keepers.get(st.session_state.teams[i]) or "", key=f"keeper_{i}")
            if keeper:
                cand = st.session_state.players[st.session_state.players["player"].str.lower() == keeper.lower()]
                if not cand.empty:
                    row = cand.iloc[0]
                    mark_drafted(row, name, method="keeper", pick_no=0)
                    st.session_state.keepers[name] = keeper
    st.session_state.teams = team_names
    st.success("League setup saved. Keepers assigned.")

with T2:
    st.subheader("Best Available")
    drafted_ids = {d["player_id"] for d in st.session_state.drafted}
    avail = st.session_state.players[~st.session_state.players["player_id"].isin(drafted_ids)]
    st.dataframe(avail[["player","team","pos","pos_rank","bye"]].head(50), use_container_width=True)

with T3:
    st.subheader("Draft Board (Grid)")
    grid = build_draft_grid()
    st.dataframe(grid, use_container_width=True, height=600)
    st.caption("Row 0 shows keepers. Following rows are draft rounds. Cells show drafted player names.")

with T4:
    st.subheader("By Position")
    drafted_ids = {d["player_id"] for d in st.session_state.drafted}
    avail = st.session_state.players[~st.session_state.players["player_id"].isin(drafted_ids)]
    pos_choice = st.selectbox("Position", options=["ALL", "QB", "RB", "WR", "TE", "DST", "K"], index=2)
    if pos_choice != "ALL":
        avail = avail[avail["pos"] == pos_choice]
    st.dataframe(avail[["player","team","pos","pos_rank","bye"]].reset_index(drop=True), use_container_width=True, height=600)
