from __future__ import annotations

import asyncio
import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles


app = FastAPI()
BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web"
DATA_DIR = BASE_DIR / "data"
HISTORY_FILE = DATA_DIR / "history.jsonl"
HISTORY_MAX = 5000
HISTORY: List[Dict[str, Any]] = []
HISTORY_LOCK = asyncio.Lock()


def _load_history() -> None:
    try:
        if not HISTORY_FILE.exists():
            return
        # jsonl: 一行一条记录
        for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                HISTORY.append(json.loads(s))
            except Exception:
                continue
        if len(HISTORY) > HISTORY_MAX:
            del HISTORY[: len(HISTORY) - HISTORY_MAX]
    except Exception:
        # 历史读取失败不影响服务启动
        return


def _write_history_line(record: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def append_history(record: Dict[str, Any]) -> None:
    async with HISTORY_LOCK:
        HISTORY.append(record)
        if len(HISTORY) > HISTORY_MAX:
            del HISTORY[: len(HISTORY) - HISTORY_MAX]
    # 文件写入放到线程，避免阻塞事件循环
    try:
        await asyncio.to_thread(_write_history_line, record)
    except Exception:
        pass


_load_history()


def _id(nbytes: int = 6) -> str:
    return secrets.token_urlsafe(nbytes)


# ----------------------------
# 扑克牌与牌型评估（德州扑克）
# ----------------------------

RANKS = "23456789TJQKA"
SUITS = "shdc"  # spades/hearts/diamonds/clubs


@dataclass(frozen=True)
class Card:
    rank: str  # one of RANKS
    suit: str  # one of SUITS

    def code(self) -> str:
        return f"{self.rank}{self.suit}"


def new_deck() -> List[Card]:
    return [Card(r, s) for r in RANKS for s in SUITS]


def rank_value(r: str) -> int:
    return RANKS.index(r) + 2


def eval_5(cards: List[Card]) -> Tuple[int, List[int]]:
    """
    返回可比较的手牌强度：
    - category: 0..8（越大越强）
      8同花顺 7四条 6葫芦 5同花 4顺子 3三条 2两对 1一对 0高牌
    - tiebreak: 若干整数（从大到小），用于同类比大小
    """
    rs = sorted([rank_value(c.rank) for c in cards], reverse=True)
    ss = [c.suit for c in cards]
    is_flush = len(set(ss)) == 1

    # 处理顺子（A2345）
    uniq = sorted(set(rs), reverse=True)
    is_straight = False
    straight_high = 0
    if len(uniq) == 5 and uniq[0] - uniq[4] == 4:
        is_straight = True
        straight_high = uniq[0]
    elif set(rs) == {14, 5, 4, 3, 2}:
        is_straight = True
        straight_high = 5

    # 计数
    counts: Dict[int, int] = {}
    for v in rs:
        counts[v] = counts.get(v, 0) + 1
    groups = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)  # (rank, cnt)
    cnts = sorted(counts.values(), reverse=True)

    if is_straight and is_flush:
        return 8, [straight_high]
    if cnts == [4, 1]:
        four = groups[0][0]
        kicker = max([v for v in rs if v != four])
        return 7, [four, kicker]
    if cnts == [3, 2]:
        trips = groups[0][0]
        pair = groups[1][0]
        return 6, [trips, pair]
    if is_flush:
        return 5, sorted(rs, reverse=True)
    if is_straight:
        return 4, [straight_high]
    if cnts == [3, 1, 1]:
        trips = groups[0][0]
        kickers = sorted([v for v in rs if v != trips], reverse=True)
        return 3, [trips] + kickers
    if cnts == [2, 2, 1]:
        pair1 = groups[0][0]
        pair2 = groups[1][0]
        hi, lo = max(pair1, pair2), min(pair1, pair2)
        kicker = max([v for v in rs if v != pair1 and v != pair2])
        return 2, [hi, lo, kicker]
    if cnts == [2, 1, 1, 1]:
        pair = groups[0][0]
        kickers = sorted([v for v in rs if v != pair], reverse=True)
        return 1, [pair] + kickers
    return 0, sorted(rs, reverse=True)


def eval_7(cards: List[Card]) -> Tuple[int, List[int]]:
    assert len(cards) == 7
    best = (-1, [])
    # 7 选 5：21 种，直接枚举
    for i in range(7):
        for j in range(i + 1, 7):
            five = [cards[k] for k in range(7) if k not in (i, j)]
            val = eval_5(five)
            if val > best:
                best = val
    return best


HAND_CATEGORY_NAME = {
    8: "同花顺",
    7: "四条",
    6: "葫芦",
    5: "同花",
    4: "顺子",
    3: "三条",
    2: "两对",
    1: "一对",
    0: "高牌",
}


def best_hand_any(cards: List[Card]) -> Tuple[Tuple[int, List[int]], List[Card]]:
    """
    德州（Hold'em）规则：可从“手牌+公共牌”中任取 5 张组成最终牌型。
    cards: 已知的全部牌（>=5）
    return: (score_tuple, best_five_cards)
    """
    if len(cards) < 5:
        raise ValueError("cards_not_enough")
    best_score: Tuple[int, List[int]] = (-1, [])
    best_five: List[Card] = []
    for five in combinations(cards, 5):
        score = eval_5(list(five))
        if score > best_score:
            best_score = score
            best_five = list(five)
    return best_score, best_five


def best_hand_omaha(hole: List[Card], board: List[Card]) -> Tuple[Tuple[int, List[int]], List[Card]]:
    """
    奥马哈规则：最终 5 张中必须“2 张来自手牌 + 3 张来自公共牌”。
    hole 可为 3~6（或 4 张经典奥马哈），board >= 3 时才可成 5 张。
    return: (score_tuple, best_five_cards)
    """
    if len(hole) < 2:
        raise ValueError("hole_not_enough")
    if len(board) < 3:
        raise ValueError("board_not_enough")
    best_score: Tuple[int, List[int]] = (-1, [])
    best_five: List[Card] = []
    for h2 in combinations(hole, 2):
        for b3 in combinations(board, 3):
            five = list(h2) + list(b3)
            score = eval_5(five)
            if score > best_score:
                best_score = score
                best_five = five
    return best_score, best_five


# ----------------------------
# 游戏状态
# ----------------------------


@dataclass
class Player:
    pid: str
    nickname: str
    join_index: int
    # 同一玩家可能在同一台电脑打开多个页面（多个 WebSocket 连接共享同一个 pid）
    # 不能用单个 ws，否则后连接会覆盖前连接，导致“只有一个页面收得到消息”
    wss: Set[WebSocket] = field(default_factory=set, repr=False)
    hole: List[Card] = field(default_factory=list)
    chips_by_round: Dict[int, int] = field(default_factory=dict)  # round -> chip number
    chip_marks_by_round: Dict[int, bool] = field(default_factory=dict)  # round -> marked
    pending: bool = False  # 中途加入：本局观战，下一局进入
    spectator: bool = False  # 观战：不参与游戏，但可看所有人手牌


@dataclass
class Room:
    rid: str
    host_pid: str
    players: Dict[str, Player] = field(default_factory=dict)
    started: bool = False
    # 牌局编号（同一房间可进行多局）。用于隔离日志，避免新一局仍下发旧局日志。
    hand_no: int = 0
    # 本局参与的玩家 pid 集合（允许房间中存在“观战/待加入”的玩家）
    hand_pids: Set[str] = field(default_factory=set)
    # 游戏设置
    hole_count: int = 2  # 2~6
    rule: str = "holdem"  # holdem|omaha（omaha 强制用 2 张手牌 + 3 张公共牌）
    fast_pick: bool = False  # 实时筹码：本轮拿筹码不按顺序
    round_idx: int = 0  # 1..4
    stage: str = "lobby"  # lobby|preflop|flop|turn|river|showdown（兼容 drafting）

    deck: List[Card] = field(default_factory=list)
    board: List[Card] = field(default_factory=list)

    public_chips: Set[int] = field(default_factory=set)  # 当前轮公共筹码（数字1..n）
    current_turn_pid: Optional[str] = None
    # 第一轮行动顺序：对玩家随机排序后，按该顺序选择“本轮尚未拿筹码者”中最靠前者行动
    first_round_order: List[str] = field(default_factory=list)
    join_counter: int = 0
    logs: List[Dict[str, Any]] = field(default_factory=list)  # {seq, round, text}
    log_seq: int = 0

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


ROOMS: Dict[str, Room] = {}


# 阶段中文名（前后端共用）
STAGE_NAME_MAP = {
    "lobby": "等待中",
    # 兼容旧版本（曾用 drafting/showdown）
    "drafting": "翻牌前",
    "preflop": "翻牌前",
    "flop": "翻牌",
    "turn": "转牌",
    "river": "河牌",
    "showdown": "亮牌",
}


def room_public_state(room: Room, viewer_pid: Optional[str] = None) -> Dict[str, Any]:
    # 注意：亮牌阶段公开所有人的底牌；其他阶段只向本人展示自己的底牌
    reveal_all = room.stage == "showdown"
    viewer_is_spectator = bool(viewer_pid and viewer_pid in room.players and room.players[viewer_pid].spectator)
    players = []
    for p in room.players.values():
        # 未开局时，所有人默认都会参与（等待房主开始）
        in_hand = (not room.started) or (p.pid in room.hand_pids)
        hole_visible = reveal_all or viewer_is_spectator or viewer_pid == p.pid
        hand_info: Optional[Dict[str, Any]] = None
        # 翻牌圈及之后：仅对“该视角可见手牌”的玩家计算当前牌型
        if room.started and in_hand and hole_visible and len(room.board) >= 3 and len(p.hole) >= 2:
            try:
                if room.rule == "omaha":
                    score, five = best_hand_omaha(p.hole, room.board)
                else:
                    score, five = best_hand_any(p.hole + room.board)
                hand_info = {
                    "category": score[0],
                    "categoryName": HAND_CATEGORY_NAME.get(score[0], str(score[0])),
                    "five": [c.code() for c in five],
                }
            except Exception:
                hand_info = None
        players.append(
            {
                "pid": p.pid,
                "nickname": p.nickname,
                "isHost": p.pid == room.host_pid,
                "inHand": in_hand,
                "pending": bool(p.pending),
                "spectator": bool(p.spectator),
                "hole": [c.code() for c in p.hole] if hole_visible else [],
                "chipsByRound": dict(p.chips_by_round),
                "chipMarksByRound": dict(p.chip_marks_by_round),
                "hand": hand_info,
                "joinIndex": p.join_index,
            }
        )
    players.sort(key=lambda x: x["joinIndex"])

    # 只下发“系统/房间日志(handNo=0)”与“当前牌局日志(handNo=room.hand_no)”
    # 兼容旧内存数据：缺失 handNo 的日志按 0 处理
    filtered_logs = [e for e in room.logs if e.get("handNo", 0) in (0, room.hand_no)]

    return {
        "rid": room.rid,
        "started": room.started,
        "stage": room.stage,
        "stageName": STAGE_NAME_MAP.get(room.stage, room.stage),
        "round": room.round_idx,
        "settings": {
            "holeCount": room.hole_count,
            "rule": room.rule,
            "handNo": room.hand_no,
            "fastPick": room.fast_pick,
        },
        "board": [c.code() for c in room.board],
        "publicChips": sorted(list(room.public_chips)),
        "currentTurnPid": room.current_turn_pid,
        "players": players,
        "logs": filtered_logs[-200:],
    }


def _is_drafting_stage(stage: str) -> bool:
    # 兼容旧版本 stage="drafting"
    return stage in {"drafting", "preflop", "flop", "turn", "river"}


def _room_log(room: Room, text: str, round_idx: Optional[int] = None) -> None:
    room.log_seq += 1
    room.logs.append(
        {
            "seq": room.log_seq,
            "handNo": room.hand_no,
            "round": round_idx if round_idx is not None else room.round_idx,
            "stage": room.stage,
            "stageName": STAGE_NAME_MAP.get(room.stage, room.stage),
            "text": text,
        }
    )


def _room_event(room: Room, text: str, round_idx: Optional[int] = None, **extra: Any) -> None:
    """
    结构化事件日志：用于前端按“某轮/某筹码”回放。
    extra 示例：
      event="chip_take"/"chip_steal", chip=3, actorPid="..", from="public"/"player", fromPid=".."
    """
    room.log_seq += 1
    e: Dict[str, Any] = {
        "seq": room.log_seq,
        "handNo": room.hand_no,
        "round": round_idx if round_idx is not None else room.round_idx,
        "stage": room.stage,
        "stageName": STAGE_NAME_MAP.get(room.stage, room.stage),
        "text": text,
    }
    e.update(extra)
    room.logs.append(e)


async def room_broadcast(room: Room, msg: Dict[str, Any]) -> None:
    """
    向房间所有连接广播同一条消息（不区分视角）。
    主要用于 game_over 这类所有人都相同的消息。
    """
    payload = json.dumps(msg, ensure_ascii=False)
    for p in room.players.values():
        for ws in list(p.wss):
            try:
                await ws.send_text(payload)
            except Exception:
                p.wss.discard(ws)


async def room_broadcast_state(room: Room) -> None:
    """
    向房间内每个玩家的所有连接下发“该玩家视角”的 room_state。

    重要：前端会直接用最新一条 room_state 覆盖本地 state，
    如果服务端广播过 viewer_pid=None 的“公共视角状态”，会把自己的底牌清空覆盖掉，
    导致“断线重连后手牌不显示”的问题（私有状态先来、公共状态后到的竞态）。
    """
    for pid, p in room.players.items():
        msg = {"type": "room_state", "state": room_public_state(room, pid), "forPid": pid}
        payload = json.dumps(msg, ensure_ascii=False)
        for ws in list(p.wss):
            try:
                await ws.send_text(payload)
            except Exception:
                p.wss.discard(ws)


def _ensure_room(rid: str) -> Room:
    if rid not in ROOMS:
        raise KeyError("room_not_found")
    return ROOMS[rid]


def _n_players(room: Room) -> int:
    return len(room.players)


def _hand_pids(room: Room) -> List[str]:
    # 保证稳定顺序（按 joinIndex）
    ps = [room.players[pid] for pid in room.hand_pids if pid in room.players]
    ps.sort(key=lambda p: p.join_index)
    return [p.pid for p in ps]


def _n_hand_players(room: Room) -> int:
    return len([pid for pid in room.hand_pids if pid in room.players])


def _all_have_chip(room: Room, round_idx: int) -> bool:
    return all(round_idx in room.players[pid].chips_by_round for pid in _hand_pids(room))


def _unassigned_players(room: Room, round_idx: int) -> List[Player]:
    return [room.players[pid] for pid in _hand_pids(room) if round_idx not in room.players[pid].chips_by_round]


def _first_round_rank(room: Room) -> Dict[str, int]:
    # order 中越靠前 rank 越小
    return {pid: i for i, pid in enumerate(room.first_round_order)}


def _next_turn(room: Room) -> None:
    if not _is_drafting_stage(room.stage):
        room.current_turn_pid = None
        return
    unassigned = {p.pid for p in _unassigned_players(room, room.round_idx)}
    if not unassigned:
        room.current_turn_pid = None
        return

    # 行动顺序规则（更贴近描述）：
    # - 第1轮：按随机顺序，从“本轮还没拿筹码的人”里挑最靠前者行动
    # - 其他轮：从“上一轮筹码最小且本轮未拿筹码的人”开始行动；每次都重新在未拿筹码者中取最小
    if room.round_idx == 1:
        rank = _first_round_rank(room)
        room.current_turn_pid = min(unassigned, key=lambda pid: (rank.get(pid, 10**9), room.players[pid].join_index))
        return
    prev = room.round_idx - 1
    room.current_turn_pid = min(
        unassigned,
        key=lambda pid: (room.players[pid].chips_by_round.get(prev, 10**9), room.players[pid].join_index),
    )
    return


def _deal_for_round(room: Room, round_idx: int) -> None:
    # round1: 发两张底牌 + 不翻公共牌
    # round2: 翻牌 3 张
    # round3: 转牌 1 张
    # round4: 河牌 1 张
    if round_idx == 1:
        room.board = []
        hc = min(6, max(2, int(room.hole_count)))
        for pid in _hand_pids(room):
            p = room.players[pid]
            p.hole = [room.deck.pop() for _ in range(hc)]
        _room_log(room, f"发底牌：每人{hc}张（仅自己可见）", round_idx=1)
    elif round_idx == 2:
        room.board.extend([room.deck.pop(), room.deck.pop(), room.deck.pop()])
        _room_log(room, f"翻牌：{' '.join([c.code() for c in room.board[-3:]])}", round_idx=2)
    elif round_idx == 3:
        room.board.append(room.deck.pop())
        _room_log(room, f"转牌：{room.board[-1].code()}", round_idx=3)
    elif round_idx == 4:
        room.board.append(room.deck.pop())
        _room_log(room, f"河牌：{room.board[-1].code()}", round_idx=4)


def _start_round(room: Room, round_idx: int) -> None:
    room.round_idx = round_idx
    room.stage = {1: "preflop", 2: "flop", 3: "turn", 4: "river"}.get(round_idx, room.stage)
    n = _n_hand_players(room)
    room.public_chips = set(range(1, n + 1))
    _room_log(room, f"第{round_idx}轮开始：生成公共筹码 1..{n}", round_idx=round_idx)
    if room.fast_pick:
        room.current_turn_pid = None
    else:
        _next_turn(room)


def _reset_for_new_game(room: Room) -> None:
    # 保留玩家与日志，重置牌局相关状态
    room.started = True
    room.hand_no += 1
    # 下一局：把 pending 玩家纳入本局参与
    room.hand_pids = {pid for pid, p in room.players.items() if not p.spectator}
    for p in room.players.values():
        if not p.spectator:
            p.pending = False
    room.stage = "preflop"
    room.round_idx = 0
    room.board = []
    room.public_chips = set()
    room.current_turn_pid = None
    room.first_round_order = []

    for pid in list(room.players.keys()):
        p = room.players[pid]
        p.hole = []
        p.chips_by_round = {}
        p.chip_marks_by_round = {}

    room.deck = new_deck()
    secrets.SystemRandom().shuffle(room.deck)
    pids = _hand_pids(room)
    secrets.SystemRandom().shuffle(pids)
    room.first_round_order = pids

    _room_log(room, "—— 新一局开始 ——", round_idx=0)
    _room_log(room, f"玩家数={_n_hand_players(room)}", round_idx=0)
    _deal_for_round(room, 1)
    _start_round(room, 1)


def _check_showdown_win(room: Room) -> Dict[str, Any]:
    """
    比较牌型大小（支持 Hold'em / Omaha），得到从大到小的名次。
    要求：名次第k的玩家在第4轮持有的筹码数字 == (n-k+1) 才算排列正确。
    若出现同牌型（完全相等）则允许筹码不严格匹配（任意分配都算对）。
    """
    n = _n_hand_players(room)
    hand_pids = _hand_pids(room)
    hc = min(6, max(2, int(room.hole_count)))
    if len(room.board) != 5 or n < 2:
        return {"ok": False, "reason": "cards_not_complete"}

    scored: List[Tuple[Tuple[int, List[int]], str, Dict[str, Any]]] = []
    for pid in hand_pids:
        p = room.players[pid]
        if len(p.hole) != hc:
            return {"ok": False, "reason": "cards_not_complete"}
        if room.rule == "omaha":
            score, five = best_hand_omaha(p.hole, room.board)
        else:
            score, five = best_hand_any(p.hole + room.board)
        detail = {
            "pid": pid,
            "nickname": p.nickname,
            "score": score,
            "category": score[0],
            "categoryName": HAND_CATEGORY_NAME.get(score[0], str(score[0])),
            "five": [c.code() for c in five],
            "chipR4": p.chips_by_round.get(4),
        }
        scored.append((score, pid, detail))
    scored.sort(key=lambda x: x[0], reverse=True)  # best first

    # 分组处理完全相等的手牌
    groups: List[List[str]] = []
    i = 0
    while i < len(scored):
        j = i + 1
        while j < len(scored) and scored[j][0] == scored[i][0]:
            j += 1
        groups.append([pid for _, pid, _ in scored[i:j]])
        i = j

    # 需要匹配的目标筹码（第4轮）
    # 位置 1..n 对应筹码 n..1
    pos = 1
    for g in groups:
        if len(g) == 1:
            pid = g[0]
            need_chip = n - pos + 1
            have_chip = room.players[pid].chips_by_round.get(4)
            if have_chip != need_chip:
                return {
                    "ok": False,
                    "reason": "chip_order_mismatch",
                    "detail": {"pid": pid, "need": need_chip, "have": have_chip},
                    "showdown": {"ranking": [d for _, _, d in scored], "rule": room.rule, "holeCount": hc},
                }
        # 同牌型组：不要求严格匹配
        pos += len(g)

    return {"ok": True, "showdown": {"ranking": [d for _, _, d in scored], "rule": room.rule, "holeCount": hc}}


# ----------------------------
# 静态页面
# ----------------------------


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
async def index():
    # Render 上若工作目录/打包异常导致 web/ 丢失，这里返回可读的诊断页，避免只看到 Not Found
    path = WEB_DIR / "index.html"
    if path.exists():
        return FileResponse(str(path))
    try:
        entries = [p.name for p in WEB_DIR.iterdir()]
    except Exception as e:
        entries = [f"<无法读取 WEB_DIR：{e}>"]
    return HTMLResponse(
        f"""
<!doctype html>
<meta charset="utf-8" />
<title>纸牌帮</title>
<h2>部署诊断：未找到 web/index.html</h2>
<p>WEB_DIR: <code>{WEB_DIR}</code></p>
<p>目录内容: <code>{entries}</code></p>
<p>请打开 <code>/docs</code> 或 <code>/debug</code> 查看更多信息。</p>
""",
        status_code=500,
    )


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/debug")
async def debug():
    path = WEB_DIR / "index.html"
    return {
        "baseDir": str(BASE_DIR),
        "webDir": str(WEB_DIR),
        "indexExists": path.exists(),
        "webEntries": [p.name for p in WEB_DIR.iterdir()] if WEB_DIR.exists() else [],
    }


@app.get("/api/history")
async def api_history(limit: int = 50, rid: Optional[str] = None):
    """
    返回最近的对局历史（尽可能多信息）。
    注意：此接口会包含亮牌信息；适合自用/局域网部署。
    """
    limit = max(1, min(200, int(limit)))
    async with HISTORY_LOCK:
        items = HISTORY
        if rid:
            items = [x for x in items if x.get("rid") == rid]
        return {"items": items[-limit:]}


# ----------------------------
# WebSocket 协议
# ----------------------------
# client -> server:
# - {"type":"create_room","nickname":"xx"}
# - {"type":"join_room","rid":"...","nickname":"xx"}
# - {"type":"set_options","holeCount":2~6,"rule":"holdem"|"omaha"}  (仅房主；建议在未开始/亮牌后设置)
# - {"type":"start_game"}
# - {"type":"new_game"}
# - {"type":"pick_chip","chip":number,"from":"public"|"player","fromPid":optional}
#
# server -> client:
# - {"type":"hello","pid":...}
# - {"type":"room_state", "state":..., "forPid":...}
# - {"type":"error","message":"..."}
# - {"type":"game_over","result":...}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    # ----------------------------
    # 断线重连 / 刷新恢复：
    # - 客户端可在 ws url 上带 query 参数：?rid=xxx&pid=yyy
    # - 若该 pid 已存在于房间中，则直接把此 ws 重新绑定到该玩家
    # ----------------------------
    qp = ws.query_params
    resume_rid = (qp.get("rid") or "").strip()
    resume_pid = (qp.get("pid") or "").strip()

    current_room: Optional[Room] = None
    pid: str

    if resume_rid and resume_pid:
        room = ROOMS.get(resume_rid)
        if room and resume_pid in room.players:
            async with room.lock:
                p = room.players[resume_pid]
                p.wss.add(ws)
                current_room = room
                pid = resume_pid
                _room_log(room, f"玩家重连：{p.nickname}", round_idx=0)
                await ws.send_text(json.dumps({"type": "hello", "pid": pid, "resumed": True}, ensure_ascii=False))
                # 对所有玩家下发各自视角的状态，避免公共状态覆盖私有手牌
                await room_broadcast_state(room)
        else:
            pid = _id()
            await ws.send_text(json.dumps({"type": "hello", "pid": pid}, ensure_ascii=False))
    else:
        pid = _id()
        await ws.send_text(json.dumps({"type": "hello", "pid": pid}, ensure_ascii=False))
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await ws.send_text(json.dumps({"type": "error", "message": "消息不是合法 JSON"}, ensure_ascii=False))
                continue

            if not isinstance(msg, dict):
                await ws.send_text(json.dumps({"type": "error", "message": "消息格式应为 JSON 对象"}, ensure_ascii=False))
                continue

            # 若该连接曾经加入过房间，但玩家已不在房间（例如同 pid 的其它页面点了退出），
            # 则清空 current_room，避免后续处理时访问 room.players[pid] 抛异常。
            if current_room is not None and pid not in current_room.players:
                try:
                    await ws.send_text(
                        json.dumps(
                            {"type": "left_room", "rid": current_room.rid, "message": "你已不在该房间，请重新加入"},
                            ensure_ascii=False,
                        )
                    )
                except Exception:
                    pass
                current_room = None

            raw_mtype = msg.get("type")
            mtype: Any = raw_mtype
            # 兼容：不同客户端可能的写法差异（大小写/空格/连字符/别名）
            if isinstance(mtype, str):
                norm = mtype.strip().lower().replace("-", "_")
                norm = "_".join(norm.split())  # 把多余空白折叠为下划线
                alias_to_new_game = {
                    "newgame",
                    "new_game",
                    "nextgame",
                    "next_game",
                    "next_round",
                    "nextround",
                    "restart",
                    "replay",
                }
                if norm in alias_to_new_game:
                    mtype = "new_game"
                else:
                    mtype = norm
            else:
                # 有些客户端可能用 action/cmd/op 来表示消息类型
                for k in ("action", "cmd", "op"):
                    v = msg.get(k)
                    if isinstance(v, str) and v.strip():
                        mtype = v.strip().lower().replace("-", "_")
                        break

            if mtype == "create_room":
                nickname = (msg.get("nickname") or "").strip()[:20]
                if not nickname:
                    await ws.send_text(json.dumps({"type": "error", "message": "昵称不能为空"}, ensure_ascii=False))
                    continue
                rid = _id(5)
                room = Room(rid=rid, host_pid=pid)
                room.join_counter += 1
                player = Player(pid=pid, nickname=nickname, join_index=room.join_counter)
                player.wss.add(ws)
                room.players[pid] = player
                room.hand_pids.add(pid)
                _room_log(room, f"创建房间：{nickname}（房主）进入房间", round_idx=0)
                ROOMS[rid] = room
                current_room = room
                await room_broadcast_state(room)

            elif mtype == "join_room":
                rid = (msg.get("rid") or "").strip()
                nickname = (msg.get("nickname") or "").strip()[:20]
                if not rid or not nickname:
                    await ws.send_text(json.dumps({"type": "error", "message": "房间号和昵称不能为空"}, ensure_ascii=False))
                    continue
                try:
                    room = _ensure_room(rid)
                except KeyError:
                    await ws.send_text(json.dumps({"type": "error", "message": "房间不存在"}, ensure_ascii=False))
                    continue
                async with room.lock:
                    room.join_counter += 1
                    player = Player(pid=pid, nickname=nickname, join_index=room.join_counter)
                    player.wss.add(ws)
                    spectate = bool(msg.get("spectate") or msg.get("spectator") or (str(msg.get("mode") or "").strip().lower() == "spectate"))
                    if spectate:
                        player.spectator = True
                    # 若游戏已开始：允许加入房间，但本局作为观战/待加入，下一局开始时进入游戏
                    if room.started and not player.spectator:
                        player.pending = True
                    room.players[pid] = player
                    if not room.started and not player.spectator:
                        room.hand_pids.add(pid)
                    _room_log(room, f"玩家加入：{nickname} 进入房间", round_idx=0)
                    current_room = room
                    await room_broadcast_state(room)

            elif mtype == "start_game":
                if current_room is None:
                    await ws.send_text(json.dumps({"type": "error", "message": "未加入房间"}, ensure_ascii=False))
                    continue
                room = current_room
                async with room.lock:
                    if pid != room.host_pid:
                        await ws.send_text(json.dumps({"type": "error", "message": "只有房主可以开始"}, ensure_ascii=False))
                        continue
                    if room.started:
                        continue
                    if _n_players(room) < 2:
                        await ws.send_text(json.dumps({"type": "error", "message": "至少需要2名玩家"}, ensure_ascii=False))
                        continue
                    room.started = True
                    # 第一次开局：从 1 开始计数
                    room.hand_no = max(room.hand_no, 0) + 1
                    room.hand_pids = {pid for pid, p in room.players.items() if not p.spectator}
                    for p in room.players.values():
                        if not p.spectator:
                            p.pending = False
                    room.stage = "preflop"
                    room.deck = new_deck()
                    secrets.SystemRandom().shuffle(room.deck)
                    pids = _hand_pids(room)
                    secrets.SystemRandom().shuffle(pids)
                    room.first_round_order = pids
                    _room_log(room, f"房主开始游戏：玩家数={_n_hand_players(room)}", round_idx=0)

                    # 第1轮发底牌并生成筹码
                    _deal_for_round(room, 1)
                    _start_round(room, 1)
                    await room_broadcast_state(room)

            elif mtype == "set_options":
                if current_room is None:
                    await ws.send_text(json.dumps({"type": "error", "message": "未加入房间"}, ensure_ascii=False))
                    continue
                room = current_room
                async with room.lock:
                    if pid != room.host_pid:
                        await ws.send_text(json.dumps({"type": "error", "message": "只有房主可以修改设置"}, ensure_ascii=False))
                        continue
                    if room.started and _is_drafting_stage(room.stage):
                        await ws.send_text(
                            json.dumps({"type": "error", "message": "游戏进行中无法修改设置，请在等待中或亮牌后修改"}, ensure_ascii=False)
                        )
                        continue

                    try:
                        hole_count = int(msg.get("holeCount") or msg.get("hole_count") or msg.get("hole") or 2)
                    except Exception:
                        hole_count = 2
                    hole_count = max(2, min(6, hole_count))

                    rule_raw = (msg.get("rule") or "holdem")
                    rule = str(rule_raw).strip().lower().replace("-", "_")
                    if rule not in {"holdem", "omaha"}:
                        await ws.send_text(json.dumps({"type": "error", "message": "rule 不合法（holdem/omaha）"}, ensure_ascii=False))
                        continue

                    fast_pick = bool(msg.get("fastPick") or msg.get("fast_pick") or msg.get("fastpick"))

                    room.hole_count = hole_count
                    room.rule = rule
                    room.fast_pick = fast_pick
                    _room_log(room, f"房主设置：手牌{hole_count}张，规则={rule}，实时筹码={'开' if fast_pick else '关'}", round_idx=0)
                    await room_broadcast_state(room)

            elif mtype == "new_game":
                if current_room is None:
                    await ws.send_text(json.dumps({"type": "error", "message": "未加入房间"}, ensure_ascii=False))
                    continue
                room = current_room
                async with room.lock:
                    if pid != room.host_pid:
                        await ws.send_text(json.dumps({"type": "error", "message": "只有房主可以开始下一局"}, ensure_ascii=False))
                        continue
                    if _n_players(room) < 2:
                        await ws.send_text(json.dumps({"type": "error", "message": "至少需要2名玩家"}, ensure_ascii=False))
                        continue
                    # 允许在结束后或中途直接重开（方便调试）
                    _reset_for_new_game(room)
                    await room_broadcast_state(room)

            elif mtype == "leave_room":
                if current_room is None:
                    await ws.send_text(json.dumps({"type": "error", "message": "未加入房间"}, ensure_ascii=False))
                    continue
                room = current_room
                async with room.lock:
                    if pid not in room.players:
                        current_room = None
                        continue
                    leaver = room.players[pid]
                    leaver_wss = list(leaver.wss)
                    # 从房间移除
                    room.players.pop(pid, None)
                    room.hand_pids.discard(pid)
                    _room_log(room, f"玩家退出：{leaver.nickname}", round_idx=0)
                    # 房主转移
                    if room.host_pid == pid and room.players:
                        new_host = min(room.players.values(), key=lambda p: p.join_index)
                        room.host_pid = new_host.pid
                        _room_log(room, f"房主转移给：{new_host.nickname}", round_idx=0)

                    # 房间空了就删除
                    if not room.players:
                        ROOMS.pop(room.rid, None)
                    else:
                        # 若游戏中，可能影响筹码与回合：直接重算当前行动者
                        _next_turn(room)
                        await room_broadcast_state(room)

                # 同 pid 可能有多个页面：都通知“已退出房间”，让前端同步清空 UI
                for w in leaver_wss:
                    try:
                        await w.send_text(json.dumps({"type": "left_room", "rid": room.rid}, ensure_ascii=False))
                    except Exception:
                        pass

                current_room = None

            elif mtype == "pick_chip":
                if current_room is None:
                    await ws.send_text(json.dumps({"type": "error", "message": "未加入房间"}, ensure_ascii=False))
                    continue
                room = current_room
                chip = int(msg.get("chip") or 0)
                from_where = msg.get("from")
                from_pid = msg.get("fromPid")
                async with room.lock:
                    if not _is_drafting_stage(room.stage):
                        await ws.send_text(json.dumps({"type": "error", "message": "当前不在选筹码阶段"}, ensure_ascii=False))
                        continue
                    if pid not in room.hand_pids:
                        await ws.send_text(json.dumps({"type": "error", "message": "你本局为观战/待加入，下一局开始后再参与"}, ensure_ascii=False))
                        continue
                    if (not room.fast_pick) and room.current_turn_pid != pid:
                        await ws.send_text(json.dumps({"type": "error", "message": "还没轮到你行动"}, ensure_ascii=False))
                        continue
                    if chip < 1 or chip > _n_hand_players(room):
                        await ws.send_text(json.dumps({"type": "error", "message": "筹码数字不合法"}, ensure_ascii=False))
                        continue
                    me = room.players[pid]
                    if room.round_idx in me.chips_by_round:
                        await ws.send_text(json.dumps({"type": "error", "message": "你本轮已经拿过筹码"}, ensure_ascii=False))
                        continue

                    if from_where == "public":
                        if chip not in room.public_chips:
                            await ws.send_text(json.dumps({"type": "error", "message": "公共区没有这个筹码"}, ensure_ascii=False))
                            continue
                        room.public_chips.remove(chip)
                        me.chips_by_round[room.round_idx] = chip
                        _room_event(
                            room,
                            f"第{room.round_idx}轮：{me.nickname} 从公共区拿取筹码 #{chip}",
                            round_idx=room.round_idx,
                            event="chip_take",
                            chip=chip,
                            actorPid=me.pid,
                            actor=me.nickname,
                            frm="public",
                        )
                    elif from_where == "player":
                        if not from_pid or from_pid not in room.players:
                            await ws.send_text(json.dumps({"type": "error", "message": "目标玩家不存在"}, ensure_ascii=False))
                            continue
                        if from_pid == pid:
                            await ws.send_text(json.dumps({"type": "error", "message": "不能从自己那里拿"}, ensure_ascii=False))
                            continue
                        if from_pid not in room.hand_pids:
                            await ws.send_text(json.dumps({"type": "error", "message": "目标玩家本局不参与"}, ensure_ascii=False))
                            continue
                        other = room.players[from_pid]
                        if room.round_idx not in other.chips_by_round:
                            await ws.send_text(json.dumps({"type": "error", "message": "对方本轮还没有筹码，不能拿取"}, ensure_ascii=False))
                            continue
                        if other.chips_by_round[room.round_idx] != chip:
                            await ws.send_text(json.dumps({"type": "error", "message": "对方本轮筹码不是这个数字"}, ensure_ascii=False))
                            continue
                        # 交换：我拿走对方筹码，对方本轮变为“未分配”，筹码回到公共区？
                        # 根据描述“可以拿取其他玩家的筹码”：实现为“直接夺取”，对方本轮失去筹码，需之后再选；
                        # 夺取后公共区不变（筹码从对方面前转移到我面前）。
                        other.chips_by_round.pop(room.round_idx, None)
                        other.chip_marks_by_round.pop(room.round_idx, None)
                        me.chips_by_round[room.round_idx] = chip
                        _room_event(
                            room,
                            f"第{room.round_idx}轮：{me.nickname} 夺取 {other.nickname} 的筹码 #{chip}（{other.nickname} 本轮需重新拿取）",
                            round_idx=room.round_idx,
                            event="chip_steal",
                            chip=chip,
                            actorPid=me.pid,
                            actor=me.nickname,
                            frm="player",
                            fromPid=other.pid,
                            fromPlayer=other.nickname,
                        )
                    else:
                        await ws.send_text(json.dumps({"type": "error", "message": "from 参数不合法"}, ensure_ascii=False))
                        continue

                    # 记录筹码“超越标记”（第2轮及之后才有意义）
                    if room.round_idx >= 2:
                        prev = room.round_idx - 1
                        my_prev = me.chips_by_round.get(prev)
                        mark = False
                        if my_prev is not None:
                            for opid in _hand_pids(room):
                                if opid == pid:
                                    continue
                                o_prev = room.players[opid].chips_by_round.get(prev)
                                if o_prev is None:
                                    continue
                                # 上一轮曾“比我大”的玩家：o_prev > my_prev
                                if o_prev > my_prev and chip > o_prev:
                                    mark = True
                                    break
                        me.chip_marks_by_round[room.round_idx] = mark

                    # 推进回合
                    if _all_have_chip(room, room.round_idx):
                        if room.round_idx == 4:
                            # 进入摊牌
                            room.stage = "showdown"
                            room.current_turn_pid = None
                            result = _check_showdown_win(room)
                            if result.get("ok"):
                                _room_log(room, "游戏结束：排列正确，胜利！", round_idx=4)
                            else:
                                _room_log(room, f"游戏结束：失败（{result.get('reason')}）", round_idx=4)

                            # 落库历史记录（jsonl），尽可能多的信息
                            try:
                                ts = datetime.now(timezone.utc)
                                hand_logs = [e for e in room.logs if e.get("handNo") == room.hand_no]
                                ranking = (result.get("showdown") or {}).get("ranking") or []
                                best_by_pid = {d.get("pid"): d for d in ranking if isinstance(d, dict)}
                                players_record = []
                                for hpid in _hand_pids(room):
                                    pp = room.players[hpid]
                                    players_record.append(
                                        {
                                            "pid": pp.pid,
                                            "nickname": pp.nickname,
                                            "joinIndex": pp.join_index,
                                            "hole": [c.code() for c in pp.hole],
                                            "chipsByRound": dict(pp.chips_by_round),
                                            "chipMarksByRound": dict(pp.chip_marks_by_round),
                                            "best": best_by_pid.get(pp.pid),
                                        }
                                    )
                                spectators = [
                                    {"pid": sp.pid, "nickname": sp.nickname, "joinIndex": sp.join_index}
                                    for sp in room.players.values()
                                    if sp.pid not in room.hand_pids
                                ]
                                record = {
                                    "id": _id(8),
                                    "ts": ts.timestamp(),
                                    "iso": ts.isoformat(),
                                    "rid": room.rid,
                                    "handNo": room.hand_no,
                                    "settings": {"holeCount": room.hole_count, "rule": room.rule},
                                    "board": [c.code() for c in room.board],
                                    "players": players_record,
                                    "spectators": spectators,
                                    "logs": hand_logs,
                                    "result": result,
                                }
                                await append_history(record)
                            except Exception:
                                pass
                            await room_broadcast_state(room)
                            await room_broadcast(room, {"type": "game_over", "result": result})
                        else:
                            # 下一轮：先翻牌/转牌/河牌，再生成筹码
                            _deal_for_round(room, room.round_idx + 1)
                            _start_round(room, room.round_idx + 1)
                            await room_broadcast_state(room)
                    else:
                        if not room.fast_pick:
                            _next_turn(room)
                        await room_broadcast_state(room)

            else:
                # 把未知消息写到房间日志，便于排查“下一局触发未知消息类型”
                if current_room is not None:
                    try:
                        _room_log(
                            current_room,
                            f"收到未知消息：type={repr(raw_mtype)[:120]} norm={repr(mtype)[:120]} raw={raw[:260]}",
                            round_idx=0,
                        )
                    except Exception:
                        pass
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": f"未知消息类型：{repr(raw_mtype)[:120]}（归一化后：{repr(mtype)[:120]}）",
                        },
                        ensure_ascii=False,
                    )
                )

    except WebSocketDisconnect:
        # 断线：保留玩家在房间中，但移除该 ws（同 pid 可能还有其它连接仍在线）
        if current_room is not None and pid in current_room.players:
            current_room.players[pid].wss.discard(ws)
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": f"服务器异常：{e}"}, ensure_ascii=False))
        except Exception:
            pass

