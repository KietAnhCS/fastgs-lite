"""Kiểm tra môi trường, cài đặt phụ thuộc, theo dõi và giải phóng bộ nhớ."""

import gc
import os
import subprocess
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEPS_FLAG = "/content/.deps_ok"
PIP_PACKAGES = ["plyfile", "psutil", "pandas", "matplotlib", "tqdm", "lpips"]
SUBMODULES = [
    "submodules/diff-gaussian-rasterization_fastgs",
    "submodules/fused-ssim",
    "submodules/simple-knn",
]


def _gb(value):
    return value / (1024 ** 3)


def mem():
    import psutil
    import torch

    vm = psutil.virtual_memory()
    usage = dict(ram_used_gb=_gb(vm.used), ram_total_gb=_gb(vm.total), ram_pct=vm.percent)
    if torch.cuda.is_available():
        usage["vram_alloc_gb"] = _gb(torch.cuda.memory_allocated())
        usage["vram_reserved_gb"] = _gb(torch.cuda.memory_reserved())
        usage["vram_total_gb"] = _gb(torch.cuda.get_device_properties(0).total_memory)
    return usage


def show_mem(tag=""):
    usage = mem()
    line = (f"[MEM {tag:<22}] RAM {usage['ram_used_gb']:5.2f}/{usage['ram_total_gb']:4.1f} GB "
            f"({usage['ram_pct']:4.1f}%)")
    if "vram_alloc_gb" in usage:
        line += (f" | VRAM {usage['vram_alloc_gb']:5.2f} alloc / "
                 f"{usage['vram_reserved_gb']:5.2f} reserved / {usage['vram_total_gb']:4.1f} GB")
    print(line)
    return usage


def free_memory(*objects, tag="cleanup"):
    """Xoá tham chiếu lớn, thu gom rác, trả VRAM về cho driver."""
    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return show_mem(tag)


def check_gpu(require=False):
    """In thông tin GPU/torch/RAM. Trả về dict; `require=True` thì raise nếu không có GPU."""
    import torch

    info = dict(torch=torch.__version__, cuda=torch.version.cuda,
                gpu=None, vram_total_gb=None, available=torch.cuda.is_available())
    if info["available"]:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_total_gb"] = round(_gb(torch.cuda.get_device_properties(0).total_memory), 2)
        print(f"GPU        : {info['gpu']} ({info['vram_total_gb']} GB VRAM)")
    else:
        print("GPU        : KHÔNG THẤY -> Runtime > Change runtime type > T4 GPU")
        if require:
            raise RuntimeError("cần GPU CUDA để train 3DGS")
    print(f"torch      : {info['torch']} | CUDA {info['cuda']}")
    print(f"python     : {sys.version.split()[0]}")
    show_mem("startup")
    return info


def _run(command):
    print("$", " ".join(command))
    return subprocess.run(command, check=False).returncode


def install_dependencies(force=False, flag_path=DEPS_FLAG):
    """Cài pip package + 3 submodule CUDA. Lần đầu ~3-5 phút, sau đó bỏ qua nhờ cờ."""
    if os.path.exists(flag_path) and not force:
        print("dependencies đã cài (xoá", flag_path, "để cài lại)")
        return False
    _run([sys.executable, "-m", "pip", "-q", "install", *PIP_PACKAGES])
    for module in SUBMODULES:
        if os.path.isdir(module):
            _run([sys.executable, "-m", "pip", "-q", "install", f"./{module}"])
        else:
            print("bỏ qua submodule không tồn tại:", module)
    os.makedirs(os.path.dirname(flag_path) or ".", exist_ok=True)
    open(flag_path, "w").close()
    return True


def clone_repo(repo_url, repo_dir):
    """Clone repo nếu chưa có rồi chuyển cwd vào đó."""
    if not os.path.isdir(repo_dir):
        _run(["git", "clone", "-q", repo_url, repo_dir])
    os.chdir(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    print("cwd:", os.getcwd())
    return repo_dir
