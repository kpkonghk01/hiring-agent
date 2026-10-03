# custom 分支使用說明

這份說明只涵蓋 `custom` 分支新增的功能：`batch.py`、Google Drive、SQLite 快取、Ranking、OpenRouter/llama.cpp。
upstream 原本的 `score.py`（單一 PDF）用法照舊，見 [README](../README.md)。
用語定義見 [CONTEXT.md](../CONTEXT.md)，設計決定見 [docs/adr/](adr/)。

## 1. 安裝

```bash
uv sync          # 依 pyproject.toml / uv.lock 建立 .venv（Python 3.11.13）
uv run pytest    # 確認環境正常
```

所有指令都用 `uv run python ...` 執行。第一次 OCR 會載入 Marker 的模型（約數 GB）。如果 cv-agent 已經下載過，會共用 `~/Library/Caches/datalab` 裡的那份。

## 2. `.env` 設定

```bash
cp .env.custom.example .env
```

[.env.custom.example](../.env.custom.example) 包含所有變數。標【必填】的項目要填，其餘可以留空，留空等同沒設定。
upstream 的 `.env.example` 沒有改動，它只有 `DEFAULT_MODEL` 和 `GEMINI_API_KEY` 兩個變數，這份範本已經涵蓋。

`.env`、`credentials/`、`*.sqlite`、`reports/` 都已經 gitignore。
`.env` 裡的相對路徑（`STORE_PATH`、`GOOGLE_OAUTH_*`）一律相對於 **repo 根目錄**，跟你在哪個目錄執行指令無關。

## 3. Google Drive 授權

- `credentials/client_secret.json` 已經從 cv-agent 複製過來。
- 第一次用 `--source gdrive` 或 `--upload-sheet` 時會開瀏覽器，請你用**公司帳號**同意授權，token 存到 `credentials/token.json`。之後 token 會自動 refresh，refresh 失敗時會自動重新開瀏覽器授權。
- 授權的 scope：
  - `drive.readonly`：讀取 Resume。
  - `drive.file`：只能建立和修改**這個 app 自己建立**的檔案，用來上傳 Ranking Sheet。
- hiring-agent 和 cv-agent 各用各的 token，不會互相影響。

### Drive 資料夾結構

```text
GDRIVE_FOLDER_ID/
├── 20261001/      ← Batch：名稱必須是 YYYYMMDD
│   ├── 張三.pdf
│   └── 李四.pdf
├── 20261002/
└── archive/       ← 名稱不是真實日期（含 20261340 這類），會被略過（log 會出現警告）
```

母資料夾裡直接放的 PDF 不會被讀取。

### Ranking Sheet 放在哪

`drive.file` 這個權限只能寫入 app 自己建立的資料夾，所以第一次 `--upload-sheet` 時，app 會在你的「我的雲端硬碟」建立一個 **`hiring-agent rankings`** 資料夾，之後的 Sheet 都放進去。這個資料夾可以移動或分享，app 仍然找得到它。

## 4. 申請 `GITHUB_TOKEN`

不設也能跑，但 GitHub 未登入的上限是每小時 60 次，批次跑很快就會用完。

1. GitHub 右上角頭像 → **Settings** → **Developer settings** → **Personal access tokens** → **Fine-grained tokens** → **Generate new token**
2. 填寫表單：
   - **Token name**：例如 `hiring-agent`。
   - **Expiration**：建議 90 天。
   - **Repository access**：選 **Public repositories (read-only)**。
   - **Permissions**：全部不勾。
3. 產生之後立刻複製，因為只會顯示一次。貼到 `.env` 的 `GITHUB_TOKEN=`。上限會提高到每小時 5000 次。

## 5. LLM provider

provider 定義在 `providers.json`（upstream 的檔案，不要改）和 `providers.custom.json`（這個分支新增的）。

| provider | 用途 | 需要的 env |
|---|---|---|
| `openrouter` | OpenRouter（`"models": "*"`，不列出 model） | `OPENROUTER_API_KEY` |
| `llamacpp` | 本機或區網的 `llama-server`（`"models": "*"`） | `LLAMACPP_BASE_URL`（選填） |
| `ollama` / `gemini` / `anthropic` | upstream 原有 | 見 README |

`batch.py` 怎麼決定 provider：
- 有 `--provider`（評分）、`EXTRACTION_PROVIDER`（抽取）或 `LLM_PROVIDER` 時，就用指定的那個，**任何 model 名稱都接受**。列在 provider 底下的 model 用列出的 temperature/top_p；沒列出的用 0.1/0.9。
- 都沒有時，就和 upstream 一樣，找「列出這個 model」的 provider。`"*"` 的 provider 不會被這樣推斷出來。

- **`openrouter`** 預設帶 `data_collection: deny`，排除會拿資料去訓練的上游 provider。還有 `require_parameters: true`，只路由到支援 JSON schema 輸出的上游 provider。如果某個 model 因此找不到可用的上游 provider，請換一個 model。
- **`llamacpp`**：
  - `llama-server` 不在意 model 名稱，但 `--model` 仍然要填一個能代表它的名字，例如 `qwen3-30b-a3b-q4`，因為這個名字會成為 Evaluation 的一部分（judge 是 `llamacpp:qwen3-30b-a3b-q4`）。
  - 如果你的 llama-server 有設 `--api-key`，就在 `providers.custom.json` 把 `"api_key_env"` 改成 `"LLAMACPP_API_KEY"`，再到 `.env` 設定這個值。
- 要新增其他 OpenAI-compatible 服務，在 `providers.custom.json` 加一段，照抄 `llamacpp` 的格式就可以。
- upstream 的 `score.py` **不看** `LLM_PROVIDER`，只認列在 provider 底下的 model。如果要用 `score.py` 打 OpenRouter，就在 `providers.custom.json` 把 `openrouter` 的 `"models"` 改成列出 model 的 object，例如 `{"qwen/qwen3-235b-a22b-2507": {"temperature": 0.1, "top_p": 0.9}}`，並讓 `DEFAULT_MODEL` 指向它。

### 抽取 model 和評分 model 分開設定

| 步驟 | 用哪個 model |
|---|---|
| 抽取（OCR 之後 → Extracted Resume） | `EXTRACTION_PROVIDER`/`EXTRACTION_MODEL`，沒設就用 `LLM_PROVIDER`/`DEFAULT_MODEL`。**不受 `--provider`/`--model` 影響** |
| GitHub 挑選 project、評分 | `--provider`/`--model`，沒給就用 `LLM_PROVIDER`/`DEFAULT_MODEL` |

如果要比較不同 model 的評分，建議把抽取 model 固定在 `.env`，每次只用 `--model` 換 judge。這樣所有 judge 看到的都是同一份 Extracted Resume。

## 6. `batch.py` 常用指令

```bash
# 跑 10/01～10/03 三天的 Batch（含頭含尾）
uv run python batch.py --role ai_application_engineer --source gdrive --since 20261001 --to 20261003

# 只指定起日（到最新）或只指定迄日（從最早）
uv run python batch.py --role ai_application_engineer --source gdrive --since 20261001

# 本地：單一 PDF，或一個資料夾（只讀第一層的 *.pdf）
uv run python batch.py --role ai_application_engineer --source local ~/Downloads/resumes/

# 同一批 Resume 換一個 judge 比較：抽取結果全部沿用，只重新評分
uv run python batch.py --role ai_application_engineer --source gdrive --since 20261001 --to 20261003 \
  --provider llamacpp --model qwen3-30b-a3b-q4

# 先試跑 3 份
uv run python batch.py --role ai_application_engineer --source gdrive --since 20261001 --limit 3

# 跑完上傳成 Google Sheet
uv run python batch.py --role ai_application_engineer --source gdrive --since 20261001 --upload-sheet

# 看 upstream 的完整報告和 log
uv run python batch.py ... --verbose
```

| flag | 作用 |
|---|---|
| `--re-extract` | 重新 OCR 和抽取，並**刪除這些 Resume 所有的 Evaluation**（各 Role、各 judge 都算） |
| `--re-evaluate` | 用目前的 Role 和 judge 重新評分，覆蓋舊的那一份 |
| `--limit N` | 最多處理 N 份 Resume（依 Batch、檔名排序） |

結束代碼：
- `0`：全部成功。
- `1`：有 Resume 失敗（會列在 Ranking 最後），或 Sheet 上傳失敗（CSV 已經寫好）。
- `2`：參數或設定錯誤，或在處理過程中遇到 Google Drive、OAuth、網路、SQLite 錯誤。

## 7. 輸出

- **進度**：每份 Resume 一行，例如 `[3/40] 20261002 張三.pdf … extract:cached evaluate:new → 72.5`。
- **Ranking 表**：依 overall score 排序。overall score 的算法是各 category 分數（以 Role 的上限封頂）加 bonus 減 deductions，和 `score.py` 印出的 OVERALL SCORE 相同。失敗的 Resume 列在最後，附上原因。
- **CSV**：寫到 `reports/<role>__<provider_model>__<範圍>.csv`，編碼是 UTF-8 BOM，Excel 可以直接開。
  - `<範圍>`：gdrive 是 `<since>-<to>`，沒指定時分別寫成 `first`、`last`；local 是檔名或資料夾名稱。
  - 加了 `--limit N` 時，檔名後面會再加 `__limitN`，避免不完整的 Ranking 蓋掉完整的那份。
  - 欄位沿用 upstream CSV，另外加上 `rank`、`overall_score`、`batch`、`judge`、`resume_hash`、`failed_step`、`failure`。
  - 同樣的條件再跑一次會覆蓋舊檔，因為內容一定是完整的。
- **Google Sheet**（`--upload-sheet`）：每次執行都建一份新的，標題後面附上建立時間。

## 8. 快取規則：什麼時候會重做？

| 情況 | 抽取 | 評分 |
|---|---|---|
| 同一份 PDF 再跑一次（即使在不同的 Batch） | 沿用 | 沿用 |
| 換 judge（`--provider`/`--model`） | 沿用 | 新增一份，舊的保留 |
| 修改 Role 的 `role.json`、`criteria.jinja`、`system_message.jinja` | 沿用 | 新增一份，舊的保留 |
| 換抽取 model | **沿用**（要重抽請加 `--re-extract`） | 沿用 |
| `--re-extract` | 重做 | 該 Resume 全部作廢，重新評分 |
| PDF 內容有任何不同（例如重新匯出） | 視為新 Resume | 視為新 Resume |
| 上次失敗（抽取失敗、GitHub 暫時抓不到、LLM 錯誤） | 下次自動重試 | 下次自動重試 |

GitHub Enrichment 的規則：
- 抓取期間只要遇到 rate limit（403/429）、5xx 或網路錯誤，該 Evaluation 就算失敗，下次再試。
- Resume 上的 GitHub 帳號如果不存在（404）或網址解析不出帳號，就在沒有 GitHub Enrichment 的情況下照常評分。

## 9. Role

- `ai_application_engineer`：依據 cv-agent 的「AI 应用工程师」JD 建立。evidence 等文字欄位輸出繁體中文。
- 新增 Role：`uv run python score.py --init-role <name>`，再編輯 `roles/<name>/` 底下的檔案。

## 10. 同步 upstream

```bash
git fetch upstream
git switch main && git merge --ff-only upstream/main && git push origin main
git switch custom && git merge main
# 如果 upstream 的 requirements.txt 有變動，就在 pyproject.toml 跟進，然後執行 uv lock
```

## 已知限制

- OCR 很慢，而且全程循序執行，沒有平行處理。不過每份 Resume 只會 OCR 一次。
- 只要 PDF 的 bytes 不同，就算是不同的 Resume。
- 只支援 GitHub，不支援 Gitee。
