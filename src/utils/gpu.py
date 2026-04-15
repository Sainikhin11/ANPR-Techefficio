
import torch
import GPUtil
from src.utils.logger import log

def check_gpu():
    """Checks for CUDA availability and lists devices."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        log.info(f"CUDA is available. Device count: {gpu_count}")
        log.info(f"Using GPU 0: {gpu_name}")
        return True, gpu_name
    else:
        log.error("CUDA is NOT available. System will run on CPU (NOT RECOMMENDED).")
        return False, "CPU"

def get_vram_usage():
    """Returns current VRAM usage in GB and percent."""
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            total = gpu.memoryTotal / 1024
            used = gpu.memoryUsed / 1024
            percent = (used / total) * 100
            return used, total, percent
        else:
            return 0.0, 0.0, 0.0
    except Exception as e:
        log.warning(f"Could not retrieve VRAM usage: {e}")
        return 0.0, 0.0, 0.0

def log_system_stats():
    used, total, percent = get_vram_usage()
    if total > 0:
        log.info(f"VRAM Usage: {used:.2f}GB / {total:.2f}GB ({percent:.1f}%)")
    else:
        log.info("VRAM Usage: N/A (CPU Mode or Error)")
