from __future__ import annotations

import asyncio
import json
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles


app = FastAPI()
BASE_DIR = Path(__file__).resolve().parents[1]
WEB_DIR = BASE_DIR / "web"


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


# ----------------------------
# 游戏状态
# ----------------------------


@dataclass
class Player:
    pid: str
    nickname: str
    join_index: int
    ws: Optional[WebSocket] = None
    hole: List[Card] = field(default_factory=list)
    chips_by_round: Dict[int, int] = field(default_factory=dict)  # round -> chip number


@dataclass
class Room:
    rid: str
    host_pid: str
    players: Dict[str, Player] = field(default_factory=dict)
    started: bool = False
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
    players = []
    for p in room.players.values():
        players.append(
            {
                "pid": p.pid,
                "nickname": p.nickname,
                "isHost": p.pid == room.host_pid,
                "hole": [c.code() for c in p.hole] if (reveal_all or viewer_pid == p.pid) else [],
                "chipsByRound": dict(p.chips_by_round),
                "joinIndex": p.join_index,
            }
        )
    players.sort(key=lambda x: x["joinIndex"])

    return {
        "rid": room.rid,
        "started": room.started,
        "stage": room.stage,
        "stageName": STAGE_NAME_MAP.get(room.stage, room.stage),
        "round": room.round_idx,
        "board": [c.code() for c in room.board],
        "publicChips": sorted(list(room.public_chips)),
        "currentTurnPid": room.current_turn_pid,
        "players": players,
        "logs": room.logs[-200:],
    }


def _is_drafting_stage(stage: str) -> bool:
    # 兼容旧版本 stage="drafting"
    return stage in {"drafting", "preflop", "flop", "turn", "river"}


def _room_log(room: Room, text: str, round_idx: Optional[int] = None) -> None:
    room.log_seq += 1
    room.logs.append(
        {
            "seq": room.log_seq,
            "round": round_idx if round_idx is not None else room.round_idx,
            "stage": room.stage,
            "stageName": STAGE_NAME_MAP.get(room.stage, room.stage),
            "text": text,
        }
    )


async def room_broadcast(room: Room, msg: Dict[str, Any]) -> None:
    dead: List[str] = []
    for pid, p in room.players.items():
        if p.ws is None:
            continue
        try:
            await p.ws.send_text(json.dumps(msg, ensure_ascii=False))
        except Exception:
            dead.append(pid)
    for pid in dead:
        room.players[pid].ws = None


def _ensure_room(rid: str) -> Room:
    if rid not in ROOMS:
        raise KeyError("room_not_found")
    return ROOMS[rid]


def _n_players(room: Room) -> int:
    return len(room.players)


def _all_have_chip(room: Room, round_idx: int) -> bool:
    return all(round_idx in p.chips_by_round for p in room.players.values())


def _unassigned_players(room: Room, round_idx: int) -> List[Player]:
    return [p for p in room.players.values() if round_idx not in p.chips_by_round]


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
        for p in room.players.values():
            p.hole = [room.deck.pop(), room.deck.pop()]
        _room_log(room, "发底牌：每人两张（仅自己可见）", round_idx=1)
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
    n = _n_players(room)
    room.public_chips = set(range(1, n + 1))
    _room_log(room, f"第{round_idx}轮开始：生成公共筹码 1..{n}", round_idx=round_idx)
    _next_turn(room)


def _reset_for_new_game(room: Room) -> None:
    # 保留玩家与日志，重置牌局相关状态
    room.started = True
    room.stage = "preflop"
    room.round_idx = 0
    room.board = []
    room.public_chips = set()
    room.current_turn_pid = None
    room.first_round_order = []

    for p in room.players.values():
        p.hole = []
        p.chips_by_round = {}

    room.deck = new_deck()
    secrets.SystemRandom().shuffle(room.deck)
    pids = list(room.players.keys())
    secrets.SystemRandom().shuffle(pids)
    room.first_round_order = pids

    _room_log(room, "—— 新一局开始 ——", round_idx=0)
    _room_log(room, f"玩家数={_n_players(room)}", round_idx=0)
    _deal_for_round(room, 1)
    _start_round(room, 1)


def _check_showdown_win(room: Room) -> Dict[str, Any]:
    """
    按德州规则比较牌型大小，得到从大到小的名次。
    要求：名次第k的玩家在第4轮持有的筹码数字 == (n-k+1) 才算排列正确。
    若出现同牌型（完全相等）则允许筹码不严格匹配（任意分配都算对）。
    """
    n = _n_players(room)
    # 计算每人最佳 7 张
    scored: List[Tuple[Tuple[int, List[int]], str]] = []
    for pid, p in room.players.items():
        if len(p.hole) != 2 or len(room.board) != 5:
            return {"ok": False, "reason": "cards_not_complete"}
        scored.append((eval_7(p.hole + room.board), pid))
    scored.sort(reverse=True)  # best first

    # 分组处理完全相等的手牌
    groups: List[List[str]] = []
    i = 0
    while i < len(scored):
        j = i + 1
        while j < len(scored) and scored[j][0] == scored[i][0]:
            j += 1
        groups.append([pid for _, pid in scored[i:j]])
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
                }
        # 同牌型组：不要求严格匹配
        pos += len(g)

    return {"ok": True}


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


# ----------------------------
# WebSocket 协议
# ----------------------------
# client -> server:
# - {"type":"create_room","nickname":"xx"}
# - {"type":"join_room","rid":"...","nickname":"xx"}
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
                p.ws = ws
                current_room = room
                pid = resume_pid
                _room_log(room, f"玩家重连：{p.nickname}", round_idx=0)
                await ws.send_text(json.dumps({"type": "hello", "pid": pid, "resumed": True}, ensure_ascii=False))
                # 该玩家视角状态（含自己的底牌）
                await ws.send_text(
                    json.dumps({"type": "room_state", "state": room_public_state(room, pid), "forPid": pid}, ensure_ascii=False)
                )
            # 其他人也刷新一下状态（可选）
            try:
                await room_broadcast(room, {"type": "room_state", "state": room_public_state(room, None)})
            except Exception:
                pass
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
                room.players[pid] = Player(pid=pid, nickname=nickname, join_index=room.join_counter, ws=ws)
                _room_log(room, f"创建房间：{nickname}（房主）进入房间", round_idx=0)
                ROOMS[rid] = room
                current_room = room
                await room_broadcast(room, {"type": "room_state", "state": room_public_state(room, pid), "forPid": pid})

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
                    if room.started:
                        await ws.send_text(json.dumps({"type": "error", "message": "游戏已开始，暂不支持中途加入"}, ensure_ascii=False))
                        continue
                    room.join_counter += 1
                    room.players[pid] = Player(pid=pid, nickname=nickname, join_index=room.join_counter, ws=ws)
                    _room_log(room, f"玩家加入：{nickname} 进入房间", round_idx=0)
                    current_room = room
                    await room_broadcast(room, {"type": "room_state", "state": room_public_state(room, pid), "forPid": pid})

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
                    room.stage = "preflop"
                    room.deck = new_deck()
                    secrets.SystemRandom().shuffle(room.deck)
                    pids = list(room.players.keys())
                    secrets.SystemRandom().shuffle(pids)
                    room.first_round_order = pids
                    _room_log(room, f"房主开始游戏：玩家数={_n_players(room)}", round_idx=0)

                    # 第1轮发底牌并生成筹码
                    _deal_for_round(room, 1)
                    _start_round(room, 1)
                    await room_broadcast(room, {"type": "room_state", "state": room_public_state(room, None)})
                    # 分别给每个玩家发“私有视角状态”（含自己底牌）
                    for p in room.players.values():
                        if p.ws:
                            await p.ws.send_text(
                                json.dumps({"type": "room_state", "state": room_public_state(room, p.pid), "forPid": p.pid}, ensure_ascii=False)
                            )

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
                    await room_broadcast(room, {"type": "room_state", "state": room_public_state(room, None)})
                    for p in room.players.values():
                        if p.ws:
                            await p.ws.send_text(
                                json.dumps({"type": "room_state", "state": room_public_state(room, p.pid), "forPid": p.pid}, ensure_ascii=False)
                            )

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
                    # 从房间移除
                    room.players.pop(pid, None)
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
                        await room_broadcast(room, {"type": "room_state", "state": room_public_state(room, None)})
                        for p in room.players.values():
                            if p.ws:
                                await p.ws.send_text(
                                    json.dumps({"type": "room_state", "state": room_public_state(room, p.pid), "forPid": p.pid}, ensure_ascii=False)
                                )

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
                    if room.current_turn_pid != pid:
                        await ws.send_text(json.dumps({"type": "error", "message": "还没轮到你行动"}, ensure_ascii=False))
                        continue
                    if chip < 1 or chip > _n_players(room):
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
                        _room_log(room, f"第{room.round_idx}轮：{me.nickname} 从公共区拿取筹码 #{chip}", round_idx=room.round_idx)
                    elif from_where == "player":
                        if not from_pid or from_pid not in room.players:
                            await ws.send_text(json.dumps({"type": "error", "message": "目标玩家不存在"}, ensure_ascii=False))
                            continue
                        if from_pid == pid:
                            await ws.send_text(json.dumps({"type": "error", "message": "不能从自己那里拿"}, ensure_ascii=False))
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
                        me.chips_by_round[room.round_idx] = chip
                        _room_log(
                            room,
                            f"第{room.round_idx}轮：{me.nickname} 夺取 {other.nickname} 的筹码 #{chip}（{other.nickname} 本轮需重新拿取）",
                            round_idx=room.round_idx,
                        )
                    else:
                        await ws.send_text(json.dumps({"type": "error", "message": "from 参数不合法"}, ensure_ascii=False))
                        continue

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
                            await room_broadcast(room, {"type": "room_state", "state": room_public_state(room, None)})
                            await room_broadcast(room, {"type": "game_over", "result": result})
                        else:
                            # 下一轮：先翻牌/转牌/河牌，再生成筹码
                            _deal_for_round(room, room.round_idx + 1)
                            _start_round(room, room.round_idx + 1)
                            await room_broadcast(room, {"type": "room_state", "state": room_public_state(room, None)})
                            for p in room.players.values():
                                if p.ws:
                                    await p.ws.send_text(
                                        json.dumps(
                                            {"type": "room_state", "state": room_public_state(room, p.pid), "forPid": p.pid},
                                            ensure_ascii=False,
                                        )
                                    )
                    else:
                        _next_turn(room)
                        await room_broadcast(room, {"type": "room_state", "state": room_public_state(room, None)})
                        for p in room.players.values():
                            if p.ws:
                                await p.ws.send_text(
                                    json.dumps(
                                        {"type": "room_state", "state": room_public_state(room, p.pid), "forPid": p.pid},
                                        ensure_ascii=False,
                                    )
                                )

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
        # 断线：保留玩家在房间中，但 ws 置空（便于重连扩展）
        if current_room is not None and pid in current_room.players:
            current_room.players[pid].ws = None
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": f"服务器异常：{e}"}, ensure_ascii=False))
        except Exception:
            pass

