# MiniMax-H3 插件安装手册（运维版）

本分支（`plugin-standalone`）只包含插件项目本体：MiniMax-H3 兼容插件源码（`packages/`）、构建工具（`tools/`）、启动脚本（`scripts/run_h3.sh`）与本手册。**不含任何 sglang 源码**——sglang 0.5.22 由你自行获取安装，插件安装其上。

全程免编译：所有二进制（Rust 内核 `.so`、sageattention 内核）都在 wheel 里，无需 Rust/maturin 工具链。

## 前提

- NVIDIA GPU（≥30GB 显存建议；32GB 单卡已验证）+ 足够新的驱动
- x86_64 Linux、glibc ≥ 2.34（manylinux_2_34 wheel 要求）
- 网络：GitHub（拉 sglang 源码）、内网 Gitea（拉插件 wheel）、PyPI（拉依赖，约 15GB）
- 模型权重 `minimax-h3-cobijada-fp8` 自备
- [uv](https://docs.astral.sh/uv/)（`curl -LsSf https://astral.sh/uv/install.sh | sh`）

## 步骤 1：获取并安装 sglang 0.5.22 源码

```bash
git clone --branch v0.5.22 https://github.com/sgl-project/sglang.git
cd sglang
uv venv venv --python 3.12
uv pip install --python venv/bin/python -e "./python[diffusion]"
```

⚠️ **sage_attn 后端需要 nvcc**（JIT 编译内核）。无系统 CUDA 13 时必须补装 pip 版（版本与生产一致）：

```bash
uv pip install --python venv/bin/python cuda-toolkit==13.0.3.0 nvidia-cuda-nvcc==13.4.92
```

> 复现生产精确版本可选：改为 `uv pip install --python venv/bin/python -e "./python[diffusion]" -r <本仓>/requirements-deps.txt`（pin 死全部依赖；sglang 本体仍来自源码 editable 安装）。

## 步骤 2：拉取本分支并安装插件（装进**同一个 venv**）

```bash
# 内网用户替换为实际 Gitea 地址：
GITEA_COMPAT_URL=<internal gitea sglang_compat clone url>
GITEA_COMPAT_RELEASE=<internal gitea sglang_compat release download base>
git clone -b plugin-standalone "$GITEA_COMPAT_URL" h3_plugins
cd h3_plugins
mkdir -p dist
curl -fL -O "$GITEA_COMPAT_RELEASE/sglang-h3-compat-0.3.1-wheels.zip"
unzip sglang-h3-compat-0.3.1-wheels.zip -d .
curl -fL -o dist/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl \
  "$GITEA_COMPAT_RELEASE/sageattention-2.2.0-cp312-cp312-linux_x86_64.whl"

# 公共替代渠道：GitHub Release（v0.3.1 含 9 插件 zip + sageattention；media-api-v0.1.0 含 media 插件）
# https://github.com/JustinWRZ/sglang_tag/releases

VENV=<步骤1中sglang目录>/venv    # 绝对路径
uv pip install --python $VENV/bin/python --find-links dist sglang-h3-compat
uv pip install --python $VENV/bin/python --find-links dist sageattention==2.2.0
```

插件 wheel 唯一分发渠道是 Gitea Release `v0.3.1`；`packages/dist/` 是本地构建产物，不在 git 里。

## 步骤 3：启动

```bash
cd h3_plugins
export MODEL_PATH=/path/to/minimax-h3-cobijada-fp8
export H3_VENV=<sglang目录>/venv        # 或把 venv 放在 ../venv、./venv 自动探测
bash scripts/run_h3.sh
```

脚本自动处理：venv 探测、CUDA_HOME（venv 内置 nvcc 优先，回退系统 CUDA）、`SGLANG_PLUGINS` 六插件、`MINIMAX_H3_VAE_ENCODE_DTYPE=bf16`、输出目录创建。

常用环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MODEL_PATH` | （必填） | 模型权重路径 |
| `H3_VENV` | 自动探测 | 装好 sglang+插件的 venv |
| `NUM_GPUS` / `BASE_GPU_ID` | `1` / `0` | 卡数 / 起始卡号 |
| `OFFLOAD_MODE` | `full` | `full`=全量 offload（低显存）；`minimal`=仅 dit layerwise；`off`=引擎按显存自动 |
| `ATTENTION_BACKEND` | `sage_attn` | 缺 sageattention 内核时降级 FA/SDPA |
| `PORT` / `MASTER_PORT` / `SCHEDULER_PORT` | `30314`/`31274`/`5578` | 端口三件套 |
| `DRY_RUN` | `0` | `=1` 只打印启动命令不执行 |

## 步骤 4：验收

```bash
curl http://127.0.0.1:30314/health        # 期望 200 {"status":"ok"}

# 一次带定版工作流的生成（10 步 + er_sde + ManualSigmas，与生产同款，约 4 分钟）
curl -sS -X POST http://127.0.0.1:30314/v1/videos -H 'content-type: application/json' -d '{
  "task": "ref2va",
  "prompt": "一只猫缓缓转头",
  "num_inference_steps": 10,
  "minimax_h3_sampler": "er_sde",
  "minimax_h3_sigmas": {"video": [1.0, 0.9882, 0.973, 0.9524, 0.9231, 0.878, 0.8, 0.6316, 0.4737, 0.1579, 0.0]},
  "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 4},
  "conditions": [{"type": "image", "role": "reference", "uri": "<服务端可下载的图片URL>"}]}'

# 用返回的 id 轮询至 completed，然后取成品 mp4：
curl -sS http://127.0.0.1:30314/v1/videos/<id>
curl -sS http://127.0.0.1:30314/v1/videos/<id>/content -o out.mp4
```

## （可选）media v1 协议端点

安装 `sglang-h3-media-api` wheel（内部 Release `media-api-v0.1.0`，或 `python tools/build_packages.py` 同源构建）并加入 `SGLANG_PLUGINS` 列表（插件名 `h3_media_api`），即获得统一 media 协议：

- `POST /generate`（严格校验 422/幂等 409/202 受理）、`GET /{job_id}/status`（支持 `?wait=0..60` 长轮询）、`GET /{job_id}`（B.5 结果信封，进行中 202/失败 410）、`DELETE /{job_id}`、`GET /jobs`、`GET /health`（media 信封，覆盖引擎原生）
- 支持范围与全部拒绝信息以 `packages/media_api/sglang_h3_media_api/capabilities.py` 为唯一事实源（文档见该包 README）
- 环境变量：`MEDIA_SERVICE_NAME` / `MEDIA_MODELS` / `MEDIA_DEFAULT_BACKGROUND` / `MEDIA_SYNC_WAIT_S` / `MEDIA_SUBMIT_TIMEOUT_S` / `MEDIA_ARTIFACT_TTL_S`

## 注意

- sglang 版本必须精确 **0.5.22**；插件必须与 sglang 装**同一个 venv**（sglang 通过安装元数据 entry-point 发现插件）
- 本分支**没有** business 直连端点（`POST /generate` 等）——那属于引擎补丁（见 `h3-v0522-deploy` 分支），本分支服务走原生 `/v1/videos`
- 修改插件源码（`packages/`）才需要重新构建：`python tools/build_packages.py`（需要 Rust/maturin）；纯使用无需任何编译
- 请求默认不带 `num_inference_steps` 时引擎跑 50 步（约 12 分钟）；要生产同款 4 分钟出片，请求带上步骤 4 示例中的工作流三参数
