# CUAD 数据集目录

本目录用于存放 **CUAD (Contract Understanding Attainment Dataset)** 原始数据与派生索引。

## 为什么用它

CUAD 由 The Atticus Project 发布，包含 **510 份真实商业合同**、**13,000+ 条**
由执业律师标注的关键条款（覆盖 41 个类目），是合同审查类任务最常用的公开
评测语料。在本项目中它有两个用途：

1. **检索先例库**：为风险审查 Agent 提供"市场上这类条款通常怎么写"的对照证据；
2. **抽取评测集**：用律师标注作为金标准，评估条款抽取的召回率/准确率。

## 获取数据

```bash
make download-cuad
# 或
python scripts/download_cuad.py
```

脚本会从 Hugging Face 下载 `CUAD_v1.json` 并解压到本目录。若网络受限，
也可手动下载后放到本目录，文件名保持 `CUAD_v1.json`。

## 目录约定

```
data/cuad/
├── README.md          # 本文件
├── CUAD_v1.json       # 原始数据（不入 git，见 .gitignore）
└── CUAD_v1/           # 原始压缩包解压出的完整目录（可选）
```

派生索引（不入 git）位于 `data/index/`：

```
data/index/
├── bm25_index.pkl     # BM25 稀疏索引（rank_bm25 持久化）
└── chroma/            # Chroma 向量库
```

## 数据结构

```jsonc
{
  "data": [
    {
      "title": "合同名称",
      "paragraphs": [
        {
          "context": "合同全文",
          "qas": [
            {
              "id": "uuid",
              "question": "Highlight the parts ... related to \"Governing Law\" ...",
              "answers": [{"text": "德克萨斯州法律", "answer_start": 12345}],
              "is_impossible": false
            }
          ]
        }
      ]
    }
  ]
}
```

`src/knowledge/cuad_loader.py` 负责把它展平为 `PassageRecord`，并把 41 个
CUAD 类目映射到项目内的 `ClauseType`。

## 许可与引用

CUAD 采用 **CC BY 4.0** 许可，可自由用于研究，请在使用时注明出处：

> The Atticus Project. CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review.
> https://www.atticusprojectai.org/cuad
