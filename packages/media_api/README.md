# sglang-h3-media-api

media v1 协议插件：在 SGLang Diffusion 0.5.22 上为 MiniMax-H3 暴露统一的 media 协议（7 服务共用契约）。零引擎改动——通过 `create_app` AROUND 钩子挂载路由。

安装后由 `SGLANG_PLUGINS`（或自动发现）加载；挂载时会**移除并接管**既有 `POST /generate`、`GET /{job_id}`、`GET /{job_id}/status` 路由，并**覆盖** `GET /health` 为 media 信封。

## 能力矩阵（H3 支持范围与拒绝信息）

所有 422 拒绝信息以 `sglang_h3_media_api/capabilities.py` 为**唯一事实源**，本表为镜像。

### 请求根字段（additionalProperties: false）

| 字段 | 支持 | 说明 / 拒绝理由 |
|---|---|---|
| `prompt` | ✅ 必填 | 1..32000 字符 |
| `negative_prompt` | ✅ | 直传引擎 |
| `assets` | ✅ 必填≥1 | 仅 `type=image` + `role=reference`（video/audio 资产、mask 角色 → 422） |
| `seed` | ✅ | 0..2147483647 |
| `batch` / `n` | ⚠️ 仅 1 | `>1` → "minimax_h3 produces exactly one output per job"；两者同传 → 422 |
| `model` | ✅ | 白名单 = `MEDIA_MODELS` 环境变量（默认 minimax-h3-cobijada-fp8） |
| `background` | ✅ | 缺省 `MEDIA_DEFAULT_BACKGROUND`（默认 true）；false 同步等待至多 `MEDIA_SYNC_WAIT_S`(300s) 后降级 202 |
| `job_id` | ✅ | hex32 预分配；重复 → 409 |

### output 字段（白名单外一律 422）

| 支持 | 不支持（422 理由摘自 capabilities.py） |
|---|---|
| `duration`(4..15)、`short_edge`(仅 768)、`ratio`(16:9/9:16/1:1/4:3/3:4)、`num_inference_steps`(1..50)、`guidance_scale`(≥0) | `size`、`quality`（固定档）、`output_format`、`output_compression`、`background`、`moderation`、`input_fidelity`、`style`、`resolution`、`frames`、`camera_fixed`、`generate_audio`（ref2va 恒带音频）、`return_last_frame`、`watermark`、`prompt_extend` |

### 工作流缺省

请求不带 `num_inference_steps` 时注入定版工作流：10 步 + `er_sde` + 11 点 ManualSigmas（与 business 旧执行器逐字一致）。

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `MEDIA_SERVICE_NAME` | `minimax-h3` | health 信封 service 字段 |
| `MEDIA_MODELS` | `minimax-h3-cobijada-fp8` | 逗号分隔模型白名单 |
| `MEDIA_DEFAULT_BACKGROUND` | `true` | background 缺省 |
| `MEDIA_SYNC_WAIT_S` | `300` | 同步模式等待上限 |
| `MEDIA_SUBMIT_TIMEOUT_S` | `30` | 提交期（含参考图下载）超时 → 422 |
| `MEDIA_ARTIFACT_TTL_S` | `86400` | 自托管 artifact URL 过期时间（过期 → 410） |

## artifacts[].url

H3 无对象存储，成品 URL 为本服务自托管：`http://<host>:<port>/media/{job_id}/artifact.mp4`，信封内附 `expires_at`；过期后 410。job 注册表为内存态，服务重启后未完成/已完成 job 均变为 404（调用方走恢复分支）。
