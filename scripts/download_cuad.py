"""下载 CUAD 数据集到 `data/cuad/`。

用法::

    python scripts/download_cuad.py
    python scripts/download_cuad.py --output data/cuad --force

数据来源：Hugging Face `theatticusproject/cuad`（CC BY 4.0）。
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

# --- 允许在未 poetry install 的情况下直接运行脚本 ---
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config import get_settings  # noqa: E402

HF_REPO = "theatticusproject/cuad"
HF_FILENAME = "CUAD_v1.json"
DIRECT_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{HF_FILENAME}"
MIN_EXPECTED_BYTES = 1_000_000


def download_via_hf_hub(target_dir: Path, filename: str) -> Path | None:
    """优先使用 huggingface_hub（支持断点续传与缓存）。"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None

    print(f"[1/2] 通过 huggingface_hub 下载 {HF_REPO}/{filename} …")
    cached = hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        repo_type="dataset",
    )
    return Path(cached)


def download_via_urllib(url: str, destination: Path) -> Path:
    """无 huggingface_hub 时的兜底下载。"""
    print(f"[1/2] 直接下载 {url} …")

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        percent = min(100.0, block_num * block_size * 100.0 / total_size)
        print(f"\r      进度: {percent:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, destination, _progress)  # noqa: S310
    print()
    return destination


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="下载 CUAD 数据集")
    parser.add_argument(
        "--output", type=Path, default=settings.cuad_dir, help="输出目录（默认 data/cuad）"
    )
    parser.add_argument("--filename", default=settings.cuad_file, help="目标文件名")
    parser.add_argument("--url", default=DIRECT_URL, help="备用直链地址")
    parser.add_argument("--force", action="store_true", help="已存在时仍重新下载")
    args = parser.parse_args()

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / args.filename

    if destination.exists() and not args.force:
        size = destination.stat().st_size
        if size >= MIN_EXPECTED_BYTES:
            print(f"数据已存在，跳过下载: {destination} ({size / 1e6:.1f} MB)")
            return 0
        print(f"检测到不完整文件（{size} 字节），将重新下载")

    source: Path | None = None
    try:
        source = download_via_hf_hub(output_dir, args.filename)
    except Exception as exc:  # noqa: BLE001
        print(f"      huggingface_hub 下载失败（{exc}），改用直链")

    if source is None:
        try:
            source = download_via_urllib(args.url, destination)
        except (urllib.error.URLError, OSError) as exc:
            print(f"下载失败: {exc}", file=sys.stderr)
            print(
                "\n请手动下载后放到 data/cuad/ 目录：\n"
                f"  {DIRECT_URL}\n"
                "或使用 HF 镜像：HF_ENDPOINT=https://hf-mirror.com python scripts/download_cuad.py",
                file=sys.stderr,
            )
            return 1

    if source != destination:
        print(f"[2/2] 复制到 {destination} …")
        destination.write_bytes(source.read_bytes())

    size = destination.stat().st_size
    print(f"完成：{destination} ({size / 1e6:.1f} MB)")
    if size < MIN_EXPECTED_BYTES:
        print("警告：文件体积异常偏小，可能未完整下载。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
