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
    # name_keyword : fp16_matrix_dim
    # sleep_ms 不再写死，启动时自动标定
    "B200":    16384,
    "B100":    16384,
    "H100":    8192,
    "H800":    8192,
    "A100":    8192,
    "A10G":    4096,
    "A10":     4096,
    "A30":     4096,
    "A40":     4096,
    "V100":    4096,
    "4090":    4096,
    "3090":    4096,
    "DEFAULT": 2048,
}


def _detect_profile(gpu_index: int) -> tuple[str, int]:
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode()
    for key, dim in GPU_PROFILES.items():
        if key != "DEFAULT" and key in name:
            return name, dim
    return name, GPU_PROFILES["DEFAULT"]


def _get_util(gpu_index: int) -> int:
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu


# ──────────────────────────────────────────────────────────────
# 单卡占位 Kernel
# ──────────────────────────────────────────────────────────────

class KeepaliveKernel:
    def __init__(self, gpu_index: int, matrix_dim: int, target_util: int = 50):
        self.gpu_index = gpu_index
        self._enabled  = False
        self._lock     = threading.Lock()

        device   = torch.device(f"cuda:{gpu_index}")
        self._a  = torch.randn(matrix_dim, matrix_dim, dtype=torch.float16, device=device)
        self._b  = torch.randn(matrix_dim, matrix_dim, dtype=torch.float16, device=device)

        # 标定：实测 mm() 耗时，反推达到目标利用率所需的 sleep
        self._sleep_ms_init = self._calibrate(device, target_util)
        self._sleep_ms = self._sleep_ms_init

        t = threading.Thread(target=self._loop, daemon=True,
                             name=f"keepalive-gpu{gpu_index}")
        t.start()

    def _calibrate(self, device: torch.device, target_util: int,
                   n: int = 20) -> float:
        """实测一次 mm() 耗时，计算 duty cycle = target_util% 所需的 sleep_ms。"""
        for _ in range(5):  # warmup
            torch.mm(self._a, self._b)
        torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        for _ in range(n):
            torch.mm(self._a, self._b)
        torch.cuda.synchronize(device)
        compute_ms = (time.perf_counter() - t0) / n * 1000

        # duty = compute / (compute + sleep) = target/100
        # => sleep = compute * (100 - target) / target
        sleep_ms = compute_ms * (100 - target_util) / target_util
        return max(0.05, sleep_ms)

    def start(self):
        with self._lock:
            self._enabled = True

    def stop(self):
        with self._lock:
            self._enabled = False
            self._sleep_ms = self._sleep_ms_init  # 重置，下次启动从标定值开始

    def set_sleep(self, ms: float):
        with self._lock:
            self._sleep_ms = max(0.05, ms)

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

_DEAD_BAND  = 3   # 误差在 ±3% 以内时不调整，避免持续抖动
_STOP_BAND  = 20  # keepalive 运行时，SM > target+STOP_BAND 才认为是真实负载，停止让路


_WARMUP_TICKS = 3  # 启动后跳过头几个采样周期，等 NVML 数据稳定再开始 PID 调节


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
    warmup_ticks: int   = 0      # 剩余预热周期，>0 时不做 PID 调节
    KP:           float = 0.05   # 比例项：适当加快收敛
    KI:           float = 0.005  # 积分项：慢速消除稳态误差
    KD:           float = 0.02   # 微分项：抑制震荡
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
    s.sleep_ms  = max(0.05, s.sleep_ms + s.KP * error + s.KI * s.integral + s.KD * derivative)
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

        print(f"[keepalive] 检测到 {len(gpu_indices)} 张待守护 GPU，逐一初始化并标定…\n")
        for idx in gpu_indices:
            name, dim = _detect_profile(idx)
            print(f"  GPU {idx}: {name}  标定中…", flush=True)
            kernel  = KeepaliveKernel(idx, dim, target_util)
            slp     = kernel._sleep_ms
            mem_mb  = 2 * dim * dim * 2 / 1024 ** 2
            tag     = next((k for k in GPU_PROFILES if k != "DEFAULT" and k in name), "DEFAULT")
            print(f"  GPU {idx}: {name}")
            print(f"           配置={tag}  矩阵={dim}x{dim}  "
                  f"sleep_calibrated={slp:.2f}ms  显存占用≈{mem_mb:.0f} MB×2")
            self.states.append(
                GpuState(gpu_index=idx, name=name, kernel=kernel, sleep_ms=slp)
            )
        print()

    # ── 单步轮询 ─────────────────────────────

    def _step(self):
        stop_threshold = self.target_util + _STOP_BAND

        for s in self.states:
            try:
                util = _get_util(s.gpu_index)
            except pynvml.NVMLError as e:
                print(f"[keepalive] GPU {s.gpu_index} 查询失败: {e}")
                continue

            if s.active:
                if util > stop_threshold:
                    # SM 远超目标，说明真实负载已到，停止占位让路
                    print(f"[keepalive] GPU {s.gpu_index} | SM={util:3d}% > {stop_threshold}%"
                          f"  → 检测到真实负载，停止占位")
                    s.kernel.stop()
                    s.active = False
                    s.integral = 0.0
                    s.sleep_ms = s.kernel._sleep_ms_init
                    s.util_history.clear()
                elif s.warmup_ticks > 0:
                    # 预热期：等 NVML 数据稳定，不做 PID 调节
                    s.warmup_ticks -= 1
                else:
                    # 持续用 PID 调节趋近 target（无论高于还是低于 target）
                    _adjust(s, util, self.target_util)
            else:
                if util < self.min_util:
                    print(f"[keepalive] GPU {s.gpu_index} | SM={util:3d}% < {self.min_util}%"
                          f"  → 启动占位")
                    s.kernel.start()
                    s.active       = True
                    s.integral     = 0.0
                    s.warmup_ticks = _WARMUP_TICKS
                    s.util_history.clear()
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
