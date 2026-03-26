# gpu-keepalive — CLAUDE.md

## 项目概览

自动维持 GPU SM 利用率 ≥ 40% 的守护进程，防止集群因利用率过低强杀任务。
支持混卡环境（V100 / A100 / H100 / B200 等），通过 PID 控制器动态调节占位强度。

## 目录结构

```
gpu-keepalive/
├── gpu_keepalive/
│   ├── __init__.py   # 版本号 (__version__)，公开导出
│   ├── cli.py        # argparse 入口，子命令 go / status / list
│   └── core.py       # 核心逻辑：KeepaliveKernel、MultiGpuController、GpuState
├── pyproject.toml    # 包元数据与构建配置（setuptools）
├── .github/
│   └── workflows/
│       └── publish.yml  # 推送 v*.*.* tag 自动发布到 PyPI
└── CLAUDE.md
```

## 核心架构

- **`KeepaliveKernel`**：每张 GPU 一个后台线程，循环执行 `torch.mm()` 矩阵乘法占位。`start()` / `stop()` 控制开关，`set_sleep()` 调节间隔。
- **`GpuState`**：每张卡独立的 PID 状态（KP=0.03, KI=0.005, KD=0.02），含滑动平均（窗口=4）消除 NVML 采样噪声，死区 ±3% 防抖。
- **`MultiGpuController`**：主循环，每 `interval` 秒轮询所有卡：利用率 < min_util 则启动占位并用 PID 调节趋近 target_util；≥ min_util 则停止让路。

## 发布流程

版本号需同步修改两个文件：
- `pyproject.toml` → `version`
- `gpu_keepalive/__init__.py` → `__version__`

发布命令：
```bash
git tag v0.x.x && git push origin v0.x.x
```

推送 tag 后 GitHub Actions 自动构建并上传到 PyPI（约 20s）。

PyPI 页面：https://pypi.org/project/gpu-keepalive/

## 常用命令

```bash
# 安装
pip install gpu-keepalive

# 守护所有 GPU（默认阈值 40%，目标 50%）
gpu_keepalive go

# 守护指定 GPU
gpu_keepalive go --gpus 0,1 --min 40 --target 50

# 查看当前各卡 SM 利用率
gpu_keepalive status

# 列出所有 GPU 信息
gpu_keepalive list
```

## GPU 配置档位

`core.py` 中 `GPU_PROFILES` 按型号预设矩阵维度和 sleep 基准值：

| 型号 | 矩阵维度 | sleep_base (ms) |
|------|----------|-----------------|
| B200/B100 | 16384 | 0.15 |
| H100/H800 | 8192  | 0.20 |
| A100       | 8192  | 0.30 |
| A10/A30/A40/V100/4090 | 4096 | 0.40–0.50 |
| DEFAULT    | 2048  | 0.80 |

## GitHub Secrets

| Secret | 用途 |
|--------|------|
| `PYPI_TOKEN` | PyPI 发包认证 token |
