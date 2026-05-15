"""Monitor memory, process, and disk during MCP stress test.
Writes to monitor.log for post-mortem analysis."""
import time, os, psutil, json
from pathlib import Path

LOG = Path("tools/monitor.log")
LOG.unlink(missing_ok=True)

snapshots = []

def sample(tag=""):
    ts = time.time()
    vm = psutil.virtual_memory()
    snap = {
        "t": round(ts - start, 2),
        "tag": tag,
        "sys_used_gb": round(vm.used / 1024**3, 1),
        "sys_avail_gb": round(vm.available / 1024**3, 1),
        "py_rss_mb": 0,
        "qdrant_rss_mb": 0,
        "rust_sidecar_rss_mb": 0,
        "new_download_bytes": _cache_size_change(),
    }
    for p in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
        try:
            name = (p.info["name"] or "").lower()
            cmd = " ".join(p.info["cmdline"] or [])
            if "python" in name and "raggy_mcp" in cmd:
                snap["py_rss_mb"] += p.info["memory_info"].rss / 1024**2
            if "qdrant" in name:
                snap["qdrant_rss_mb"] += p.info["memory_info"].rss / 1024**2
            if "qwen3" in name or "qwen3-embedder" in cmd:
                snap["rust_sidecar_rss_mb"] += p.info["memory_info"].rss / 1024**2
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    snapshots.append(snap)
    LOG.write_text(json.dumps(snapshots, indent=2))

def _cache_size_change():
    """Return how much huggingface cache grew since start."""
    hf = Path.home() / ".cache" / "huggingface" / "hub"
    if not hf.exists():
        return 0
    total = sum(f.stat().st_size for f in hf.rglob("*") if f.is_file())
    return total - _cache_start

# Record initial cache size
_cache_start = 0
hf = Path.home() / ".cache" / "huggingface" / "hub"
if hf.exists():
    _cache_start = sum(f.stat().st_size for f in hf.rglob("*") if f.is_file())

start = time.time()
sample("init")

print("Monitor running — will sample every 2s. Ctrl+C to stop.")
try:
    while True:
        time.sleep(2)
        sample()
except KeyboardInterrupt:
    sample("final")
    print(f"\n{len(snapshots)} samples written to {LOG}")
    # Print summary
    if snapshots:
        peaks = {k: max(s.get(k, 0) for s in snapshots) for k in ["sys_used_gb", "py_rss_mb", "qdrant_rss_mb", "rust_sidecar_rss_mb", "new_download_bytes"]}
        print(f"Peak sys: {peaks['sys_used_gb']}GB, Python: {peaks['py_rss_mb']}MB, Qdrant: {peaks['qdrant_rss_mb']}MB")
        print(f"Peak Rust sidecar: {peaks['rust_sidecar_rss_mb']}MB, New cache: {peaks['new_download_bytes']/1024**3:.1f}GB")
