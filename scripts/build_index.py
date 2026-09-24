"""构建 BM25 + 向量索引。

用法::

    python scripts/build_index.py                      # 全量 CUAD，混合索引
    python scripts/build_index.py --limit 30           # 只用前 30 份合同（快速验证）
    python scripts/build_index.py --no-vector          # 只建 BM25（无需下载向量模型）
    python scripts/build_index.py --config configs/retrieval.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config import get_settings  # noqa: E402
from knowledge.bm25 import BM25Index  # noqa: E402
from knowledge.cuad_loader import load_cuad_records  # noqa: E402
from knowledge.retriever import BM25_FILENAME, HybridRetriever  # noqa: E402


def load_config(path: str | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists():
        print(f"警告：配置文件不存在 {config_path}，使用默认参数", file=sys.stderr)
        return {}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="构建 CUAD 检索索引")
    parser.add_argument("--config", default="configs/retrieval.yaml", help="检索配置 YAML")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 份合同")
    parser.add_argument("--no-vector", action="store_true", help="跳过向量索引，仅建 BM25")
    parser.add_argument("--reset", action="store_true", help="重建前清空已有向量库")
    parser.add_argument("--output", type=Path, default=None, help="索引输出目录")
    args = parser.parse_args()

    config = load_config(args.config)
    index_cfg = config.get("index", {}) or {}
    limit = args.limit if args.limit is not None else index_cfg.get("limit_contracts")
    batch_size = int(index_cfg.get("batch_size", 64))
    reset = args.reset or bool(index_cfg.get("reset", False))

    index_dir = Path(args.output) if args.output else Path(settings.chroma_dir).parent
    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"索引目录: {index_dir}")
    print(f"CUAD 数据: {settings.cuad_path}")

    started = time.perf_counter()
    print("\n[1/3] 加载 CUAD 记录 …")
    records = load_cuad_records(
        limit_contracts=limit,
        min_answer_chars=int(index_cfg.get("min_answer_chars", 20)),
        context_window=int(index_cfg.get("context_window", 400)),
    )
    if not records:
        print("未加载到任何记录，请先运行 scripts/download_cuad.py", file=sys.stderr)
        return 1
    print(f"      共 {len(records)} 条条款记录")

    print("\n[2/3] 构建 BM25 索引 …")
    bm25 = BM25Index(
        k1=float(config.get("retrievers", {}).get("bm25", {}).get("k1", 1.5)),
        b=float(config.get("retrievers", {}).get("bm25", {}).get("b", 0.75)),
    )
    bm25.add(records)
    bm25_path = bm25.save(index_dir / BM25_FILENAME)
    print(f"      已保存: {bm25_path}")

    retriever = HybridRetriever(bm25=bm25, vector_store=None)

    if args.no_vector:
        print("\n[3/3] 跳过向量索引（--no-vector）")
    else:
        print("\n[3/3] 构建向量索引（首次运行需下载向量模型）…")
        try:
            from knowledge.vectorstore import ChromaVectorStore

            store = ChromaVectorStore()
            if reset:
                store.reset()
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                store.add(batch)
                done = min(start + batch_size, len(records))
                print(f"\r      进度: {done}/{len(records)}", end="", flush=True)
            print()
            retriever.vector_store = store
            print(f"      向量库条目数: {store.count}")
        except Exception as exc:  # noqa: BLE001
            print(f"      向量索引构建失败（{exc}），已保留 BM25 索引。", file=sys.stderr)

    elapsed = time.perf_counter() - started
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_records": len(records),
        "limit_contracts": limit,
        "bm25_file": str(bm25_path),
        "vector_enabled": retriever.vector_store is not None,
        "vector_count": getattr(retriever.vector_store, "count", 0),
        "embedding_model": settings.embedding_model,
        "elapsed_seconds": round(elapsed, 2),
    }
    manifest_path = index_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成，用时 {elapsed:.1f}s；索引清单: {manifest_path}")
    print("提示：服务端需重启或调用 graph.reset_graph_cache() 以加载新索引。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
