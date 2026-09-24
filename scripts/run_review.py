"""命令行审查入口。

用法::

    python scripts/run_review.py data/samples/nda.docx
    python scripts/run_review.py contract.pdf --format markdown --output report.md
    cat contract.txt | python scripts/run_review.py - --format json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from graph import run_review  # noqa: E402
from schemas.review import ReviewReport, Severity  # noqa: E402

_SEVERITY_ZH = {
    Severity.CRITICAL: "严重",
    Severity.HIGH: "高",
    Severity.MEDIUM: "中",
    Severity.LOW: "低",
    Severity.INFO: "提示",
}


def render_markdown(report: ReviewReport) -> str:
    """把报告渲染为便于粘贴到工单系统的 Markdown。"""
    lines = [
        "# 合同审查报告",
        "",
        f"- **文档**: {report.filename or report.doc_id}",
        f"- **合同类型**: {report.contract_type}",
        f"- **总体风险**: {_SEVERITY_ZH.get(report.overall_risk, '-')}"
        f"（{report.overall_risk.value}）",
        f"- **条款数**: {len(report.clauses)} ｜ **风险项**: {len(report.issues)}",
        f"- **报告 ID**: {report.report_id}",
        "",
        "## 执行摘要",
        "",
        report.summary or "（无）",
        "",
        "## 风险明细",
        "",
    ]
    if not report.issues:
        lines.append("未发现风险项。")
    for i, issue in enumerate(report.issues, start=1):
        lines.extend(
            [
                f"### {i}. [{_SEVERITY_ZH.get(issue.severity, '-')}] {issue.title}",
                "",
                f"- 条款: `{issue.clause_id or '-'}`（{issue.clause_type.value}）",
                f"- 类别: `{issue.category.value}` ｜ 置信度: {issue.confidence:.2f}",
                f"- 说明: {issue.description}",
                f"- 建议: {issue.suggestion}",
            ]
        )
        if issue.citations:
            lines.append("- 依据:")
            lines.extend(f"  - {c.ref}（{c.corpus.value}, score={c.score:.3f}）" for c in issue.citations)
        lines.append("")

    lines.extend(["## 条款清单", "", "| # | 类型 | 标题 | 置信度 |", "| --- | --- | --- | --- |"])
    for i, clause in enumerate(report.clauses, start=1):
        lines.append(f"| {i} | {clause.type.value} | {clause.title or '-'} | {clause.confidence:.2f} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="审查一份合同并输出报告")
    parser.add_argument("source", help="合同文件路径；使用 '-' 从 stdin 读取纯文本")
    parser.add_argument("--format", choices=["json", "markdown"], default="markdown")
    parser.add_argument("--output", type=Path, default=None, help="输出文件，默认打印到终端")
    parser.add_argument("--no-retrieval", action="store_true", help="关闭外部证据检索")
    parser.add_argument("--top-k", type=int, default=None, help="检索条数")
    parser.add_argument("--max-clauses", type=int, default=None, help="最多审查的条款数")
    parser.add_argument(
        "--block-on",
        choices=[s.value for s in Severity],
        default="high",
        help="风险达到该等级时以退出码 2 结束（默认 high，便于 CI 门禁）",
    )
    args = parser.parse_args()

    if args.source == "-":
        report = run_review(
            raw_text=sys.stdin.read(),
            filename="stdin.txt",
            use_retrieval=not args.no_retrieval,
            top_k=args.top_k,
            max_clauses=args.max_clauses,
        )
    else:
        path = Path(args.source)
        if not path.exists():
            print(f"文件不存在: {path}", file=sys.stderr)
            return 1
        report = run_review(
            source_path=str(path),
            filename=path.name,
            use_retrieval=not args.no_retrieval,
            top_k=args.top_k,
            max_clauses=args.max_clauses,
        )

    content = (
        report.model_dump_json(indent=2) if args.format == "json" else render_markdown(report)
    )
    if args.output:
        args.output.write_text(content, encoding="utf-8")
        print(f"报告已写入 {args.output}")
    else:
        print(content)

    # 依据阈值决定退出码（可用于 CI 门禁）
    threshold = Severity(args.block_on)
    if report.overall_risk.weight >= threshold.weight:
        print(
            f"\n[退出码 2] 总体风险 {report.overall_risk.value} 已达到阻断阈值 {threshold.value}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
