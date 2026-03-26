"""
GPU Keepalive — 自动维持所有/指定 GPU 的 SM 利用率 ≥ 40%
任务运行时自动让路，支持混卡环境 (V100 / A100 / H100 / B200 …)

用法:
    python gpu_keepalive.py                       # 守护所有 GPU
    python gpu_keepalive.py --gpus 0,1,3          # 守护指定 GPU
    python gpu_keepalive.py --gpus 0-3            # 守护 GPU 0~3
    python gpu_keepalive.py --min 40 --target 50  # 调整阈值
    python gpu_keepalive.py --interval 2          # 采样间隔秒
"""

import argparse
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import torch
import pynvml

# ──────────────────────────────────────────────────────────────
# GPU 型号配置
# ──────────────────────────────────────────────────────────────

GPU_PROFILES = {
    # name_keyword : (fp16_matrix_dim, sleep_base_ms)
    "B200":    (16384, 0.15),
    "B100":    (16384, 0.15),
    "H100":    (8192,  0.20),
    "H800":    (8192,  0.20),
    "A100":    (8192,  0.30),
    "A10G":    (4096,  0.50),
    "A10":     (4096,  0.50),
    "A30":     (4096,  0.50),
    "A40":     (4096,  0.50),
    "V100":    (4096,  0.50),
    "4090":    (4096,  0.40),
    "3090":    (4096,  0.50),
    "DEFAULT": (2048,  0.80),
}


def _detect_profile(gpu_index: int) -> tuple[str, int, float]:
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode()
    for key, (dim, slp) in GPU_PROFILES.items():
        if key != "DEFAULT" and key in name:
            return name, dim, slp
    dim, slp = GPU_PROFILES["DEFAULT"]
    return name, dim, slp


def _get_util(gpu_index: int) -> int:
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu


# ──────────────────────────────────────────────────────────────
# 单卡占位 Kernel
# ──────────────────────────────────────────────────────────────

class KeepaliveKernel:
    def __init__(self, gpu_index: int, matrix_dim: int, sleep_ms: float):
        self.gpu_index = gpu_index
        self._sleep_ms = sleep_ms
        self._enabled  = False
        self._lock     = threading.Lock()

        device   = torch.device(f"cuda:{gpu_index}")
        self._a  = torch.randn(matrix_dim, matrix_dim, dtype=torch.float16, device=device)
        self._b  = torch.randn(matrix_dim, matrix_dim, dtype=torch.float16, device=device)

        t = threading.Thread(target=self._loop, daemon=True,
                             name=f"keepalive-gpu{gpu_index}")
        t.start()

    def start(self):
        with self._lock:
            self._enabled = True

    def stop(self):
        with self._lock:
            self._enabled = False

    def set_sleep(self, ms: float):
        with self._lock:
            self._sleep_ms = max(0.01, ms)

    def _loop(self):
        torch.cuda.set_device(self.gpu_index)
        while True:
            with self._lock:
                enabled = self._enabled
                slp     = self._sleep_ms
            if enabled:
                _ = torch.mm(self._a, self._b)
                torch.cuda.synchronize()
                time.sleep(slp / 1000.0)
            else:
                time.sleep(0.05)


# ──────────────────────────────────────────────────────────────
# 单卡状态（PI 控制器参数独立）
# ──────────────────────────────────────────────────────────────

_DEAD_BAND = 3   # 误差在 ±3% 以内时不调整，避免持续抖动


@dataclass
class GpuState:
    gpu_index:    int
    name:         str
    kernel:       KeepaliveKernel
    sleep_ms:     float
    active:       bool  = False
    integral:     float = 0.0
    prev_error:   float = 0.0
    tick:         int   = 0
    KP:           float = 0.03   # 降低：减少过冲
    KI:           float = 0.005  # 降低：慢速积分
    KD:           float = 0.02   # 新增：微分项，抑制震荡
    util_history: deque = field(default_factory=lambda: deque(maxlen=4))


def _adjust(s: GpuState, util: int, target: int):
    # 滑动平均消除 NVML 采样噪声
    s.util_history.append(util)
    smooth = sum(s.util_history) / len(s.util_history)

    error = smooth - target
    # 死区：误差很小时不调整，防止持续抖手
    if abs(error) < _DEAD_BAND:
        s.prev_error = error
        return

    derivative = error - s.prev_error
    s.integral  = max(-20, min(20, s.integral + error))  # 收紧限幅
    s.sleep_ms  = max(0.01, s.sleep_ms + s.KP * error + s.KI * s.integral + s.KD * derivative)
    s.prev_error = error
    s.kernel.set_sleep(s.sleep_ms)


# ──────────────────────────────────────────────────────────────
# 多卡控制器
# ──────────────────────────────────────────────────────────────

class MultiGpuController:

    def __init__(self, gpu_indices: list[int], min_util: int,
                 target_util: int, interval: float):
        self.min_util    = min_util
        self.target_util = target_util
        self.interval    = interval
        self.running     = True
        self.states: list[GpuState] = []

        pynvml.nvmlInit()

        print(f"[keepalive] 检测到 {len(gpu_indices)} 张待守护 GPU，逐一初始化…\n")
        for idx in gpu_indices:
            name, dim, slp = _detect_profile(idx)
            kernel  = KeepaliveKernel(idx, dim, slp)
            mem_mb  = 2 * dim * dim * 2 / 1024 ** 2
            tag     = next((k for k in GPU_PROFILES if k != "DEFAULT" and k in name), "DEFAULT")
            print(f"  GPU {idx}: {name}")
            print(f"           配置={tag}  矩阵={dim}x{dim}  "
                  f"sleep_base={slp}ms  显存占用≈{mem_mb:.0f} MB×2")
            self.states.append(
                GpuState(gpu_index=idx, name=name, kernel=kernel, sleep_ms=slp)
            )
        print()

    # ── 单步轮询 ─────────────────────────────

    def _step(self):
        for s in self.states:
            try:
                util = _get_util(s.gpu_index)
            except pynvml.NVMLError as e:
                print(f"[keepalive] GPU {s.gpu_index} 查询失败: {e}")
                continue

            if util < self.min_util:
                if not s.active:
                    print(f"[keepalive] GPU {s.gpu_index} | SM={util:3d}% < {self.min_util}%"
                          f"  → 启动占位")
                    s.kernel.start()
                    s.active   = True
                    s.integral = 0.0
                else:
                    _adjust(s, util, self.target_util)
            else:
                if s.active:
                    print(f"[keepalive] GPU {s.gpu_index} | SM={util:3d}% ≥ {self.min_util}%"
                          f"  → 停止占位，让路")
                    s.kernel.stop()
                    s.active = False
            s.tick += 1

        # 每 10 拍打印汇总表
        if self.states and self.states[0].tick % 10 == 0:
            self._print_table()

    def _print_table(self):
        rows = ["  GPU  型号            SM%   状态    sleep(ms)",
                "  " + "─" * 46]
        for s in self.states:
            try:    util = _get_util(s.gpu_index)
            except: util = -1
            short = s.name.replace("NVIDIA ", "").replace("Tesla ", "")[:14]
            status = "▶ 占位中" if s.active else "  让路中"
            rows.append(f"  {s.gpu_index:3d}  {short:<14}  {util:3d}%  {status}  {s.sleep_ms:7.2f}")
        print("\n".join(rows) + "\n")

    # ── 主循环 ───────────────────────────────

    def run(self):
        idx_str = ", ".join(str(s.gpu_index) for s in self.states)
        print(f"[keepalive] 守护 GPU [{idx_str}]  "
              f"阈值={self.min_util}%  目标={self.target_util}%  "
              f"采样={self.interval}s\n"
              "[keepalive] Ctrl-C 退出\n")

        def _stop(sig, frame):
            print("\n[keepalive] 退出，停止所有占位 kernel…")
            for s in self.states:
                s.kernel.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT,  _stop)
        signal.signal(signal.SIGTERM, _stop)

        while self.running:
            try:
                self._step()
            except Exception as e:
                print(f"[keepalive] 异常: {e}")
            time.sleep(self.interval)


# ──────────────────────────────────────────────────────────────
# 解析 --gpus 参数
# ──────────────────────────────────────────────────────────────

def parse_gpu_list(raw: str, total: int) -> list[int]:
    """'all' | '0,1,3' | '0-3' | 混合均支持"""
    raw = raw.strip().lower()
    if raw == "all":
        return list(range(total))
    indices = []
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.extend(range(int(lo), int(hi) + 1))
        else:
            indices.append(int(part))
    indices = sorted(set(indices))
    for i in indices:
        if i < 0 or i >= total:
            raise ValueError(f"GPU {i} 不存在（系统共 {total} 张）")
    return indices


# ──────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GPU Keepalive — 多卡 SM 利用率守护进程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python gpu_keepalive.py                     # 守护所有 GPU
  python gpu_keepalive.py --gpus 0,1          # 只守护 GPU 0 和 1
  python gpu_keepalive.py --gpus 0-3          # 守护 GPU 0~3
  python gpu_keepalive.py --gpus 0,2-4        # 混合写法
  python gpu_keepalive.py --min 40 --target 50
""")
    parser.add_argument("--gpus",     default="all",
                        help="GPU 编号，逗号/范围，如 0,1,3 或 0-3 或 all（默认 all）")
    parser.add_argument("--min",      type=int,   default=40,
                        help="最低 SM 利用率阈值 %% (默认 40)")
    parser.add_argument("--target",   type=int,   default=50,
                        help="占位目标利用率 %% (默认 50)")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="采样间隔秒 (默认 2)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[keepalive] 未检测到 CUDA，退出")
        sys.exit(1)

    total = torch.cuda.device_count()
    print(f"[keepalive] 系统共 {total} 张 GPU")

    try:
        gpu_indices = parse_gpu_list(args.gpus, total)
    except ValueError as e:
        print(f"[keepalive] 参数错误: {e}")
        sys.exit(1)

    MultiGpuController(
        gpu_indices=gpu_indices,
        min_util=args.min,
        target_util=args.target,
        interval=args.interval,
    ).run()


if __name__ == "__main__":
    main()
