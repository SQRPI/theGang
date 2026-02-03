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

## 部署到 Render.com

### 准备

- 确保仓库里有：`requirements.txt`、`runtime.txt`、`render.yaml`
- 把整个项目推到 GitHub（Render 会从 GitHub 拉取）

### 部署步骤（Blueprint 推荐）

1. 登录 Render
2. 选择 **New +** → **Blueprint**
3. 选择你的 GitHub 仓库
4. Render 会识别 `render.yaml` 并创建一个 Web Service
5. 等待 Build + Deploy 完成后，打开 Render 提供的公网域名即可访问

### 部署步骤（手动创建 Web Service）

1. **New +** → **Web Service** → 选择 GitHub 仓库
2. Environment 选 **Python**
3. Build Command：

```bash
pip install -r requirements.txt
```

4. Start Command：

```bash
uvicorn server.main:app --host 0.0.0.0 --port $PORT
```

5. 点击 Deploy

### 说明

- Render 会自动提供 `PORT` 环境变量，所以启动命令必须用 `$PORT`
- WebSocket 在 Render 上可用（前端用相对域名连接 `/ws`）

