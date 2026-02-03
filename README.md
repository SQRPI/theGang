# 纸牌帮（局域网德州筹码顺序游戏）

## 启动（Windows / PowerShell）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

然后在局域网内其它设备访问：

- `http://<你的电脑局域网IP>:8000/`

## 说明

- 支持：创建房间、昵称加入、房主开始、4 轮发牌+筹码选择、最终胜负判定。
- 实时通信：WebSocket（前端自动连接）。

