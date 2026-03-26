"""
gpu_keepalive CLI
-----------------
gpu_keepalive go                          # 守护所有 GPU，默认参数
gpu_keepalive go --gpus 0,1 --min 40     # 指定 GPU
gpu_keepalive status                      # 打印当前各卡 SM 利用率后退出
gpu_keepalive list                        # 列出系统所有 GPU 信息
"""

import argparse
import sys


def cmd_go(args):
    """启动守护进程"""
    import torch
    import pynvml
    from .core import MultiGpuController, parse_gpu_list

    if not torch.cuda.is_available():
        print("[keepalive] 未检测到 CUDA，退出")
        sys.exit(1)

    pynvml.nvmlInit()
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


def cmd_status(_args):
    """打印每张卡当前 SM 利用率"""
    try:
        import pynvml
        import torch
    except ImportError as e:
        print(f"缺少依赖: {e}")
        sys.exit(1)

    pynvml.nvmlInit()
    total = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if total == 0:
        print("未检测到 CUDA GPU")
        return

    print(f"  {'GPU':>3}  {'型号':<22}  {'SM%':>4}  {'显存使用':>14}")
    print("  " + "─" * 50)
    for i in range(total):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name   = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        util   = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_str = f"{mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB"
        short  = name.replace("NVIDIA ", "").replace("Tesla ", "")[:22]
        print(f"  {i:3d}  {short:<22}  {util:4d}%  {mem_str:>14}")


def cmd_list(_args):
    """列出所有 GPU 详细信息"""
    try:
        import pynvml
        import torch
    except ImportError as e:
        print(f"缺少依赖: {e}")
        sys.exit(1)

    pynvml.nvmlInit()
    total = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if total == 0:
        print("未检测到 CUDA GPU")
        return

    for i in range(total):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name   = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
        print(f"GPU {i}: {name}")
        print(f"        显存: {mem.total/1024**3:.1f} GB")
        print(f"        驱动: {driver}")


def main():
    parser = argparse.ArgumentParser(
        prog="gpu_keepalive",
        description="GPU Keepalive — 自动维持 SM 利用率，防止任务被强杀",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""子命令:
  go       启动守护进程（最常用）
  status   查看当前各卡利用率后退出
  list     列出系统所有 GPU 信息

示例:
  gpu_keepalive go
  gpu_keepalive go --gpus 0,1 --min 40 --target 50
  gpu_keepalive go --gpus 0-3
  gpu_keepalive status
  gpu_keepalive list
""")

    sub = parser.add_subparsers(dest="command")
    sub.required = True

    # ── go ──────────────────────────────────────
    p_go = sub.add_parser("go", help="启动守护进程")
    p_go.add_argument("--gpus",     default="all",
                      help="GPU 编号，逗号/范围，如 0,1,3 或 0-3 或 all（默认 all）")
    p_go.add_argument("--min",      type=int,   default=40,
                      help="最低 SM 利用率阈值 %% (默认 40)")
    p_go.add_argument("--target",   type=int,   default=50,
                      help="占位目标利用率 %% (默认 50)")
    p_go.add_argument("--interval", type=float, default=2.0,
                      help="采样间隔秒 (默认 2)")
    p_go.set_defaults(func=cmd_go)

    # ── status ──────────────────────────────────
    p_st = sub.add_parser("status", help="打印当前各卡 SM 利用率后退出")
    p_st.set_defaults(func=cmd_status)

    # ── list ────────────────────────────────────
    p_ls = sub.add_parser("list", help="列出系统所有 GPU 详细信息")
    p_ls.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
