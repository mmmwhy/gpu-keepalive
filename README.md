# gpu-keepalive

自动维持 GPU SM 利用率 ≥ 40%，防止任务被集群强杀。  
任务真正运行时自动让路，支持混卡环境（V100 / A100 / H100 / B200 …）。

## 安装

```bash
pip install gpu-keepalive
```

或直接从 Git 仓库安装：

```bash
pip install "git+https://github.com/mmmwhy/gpu-keepalive.git"
```

本地开发：

```bash
git clone https://github.com/mmmwhy/gpu-keepalive.git
cd gpu_keepalive
pip install -e .
```

## 快速使用

```bash
# 守护所有 GPU，默认阈值 40%
gpu_keepalive go

# 只守护 GPU 0 和 1
gpu_keepalive go --gpus 0,1

# 守护 GPU 0~3，自定义阈值
gpu_keepalive go --gpus 0-3 --min 40 --target 50

# 查看当前各卡利用率
gpu_keepalive status

# 列出系统所有 GPU 信息
gpu_keepalive list
```

## 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `--gpus` | `all` | GPU 编号，支持 `0,1,3`、`0-3`、`all` |
| `--min` | `40` | SM 利用率低于此值时启动占位 kernel |
| `--target` | `50` | 占位 kernel 的目标利用率 |
| `--interval` | `2.0` | 采样间隔（秒） |

## 工作原理

1. 每 `interval` 秒查询一次各卡 SM 利用率
2. 利用率 < `--min` 时，启动占位 kernel（fp16 矩阵乘法），通过 PI 控制器将利用率稳定在 `--target`
3. 检测到真实负载（利用率 ≥ `--min`）时，立刻停止占位 kernel，彻底让路
4. 自动识别 GPU 型号，按 V100 / H100 / B200 等选择合适的矩阵规模

## 依赖

- Python ≥ 3.10
- PyTorch ≥ 2.0（需要 CUDA 版本）
- pynvml ≥ 11.0

## 推荐使用方式

```bash
# 放入 tmux / screen，挂在后台
tmux new -s keepalive
gpu_keepalive go --gpus all
# Ctrl-B D 分离，自由 debug
```
