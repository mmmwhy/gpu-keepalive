"""gpu_keepalive — 自动维持 GPU SM 利用率守护进程"""

__version__ = "0.1.0"
__author__  = "you"

from .core import MultiGpuController, KeepaliveKernel, parse_gpu_list

__all__ = ["MultiGpuController", "KeepaliveKernel", "parse_gpu_list"]
