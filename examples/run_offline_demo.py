"""
run_offline_demo.py —— 离线管线演示（mock 模式）

演示：一份模拟规程 PDF -> 路由 -> 版面分析 -> 表格解析 -> 清洗脱敏
      -> 语义切分 -> 两级去重 -> 增量索引。
全程使用 mock / 占位实现，不连接任何真实服务、不下载模型。

运行：python examples/run_offline_demo.py
"""
from __future__ import annotations

from pipeline_offline import (
    router, layout_analyzer, table_parser, cleaner,
    semantic_chunker, dedup, incremental_index,
)


def main():
    print("=" * 60)
    print("离线管线演示（mock）：模拟规程文档入湖")
    print("=" * 60)

    # A1 路由
    doc = router.route("./data/raw/mock_规程.pdf")
    print(f"[A1 路由] {doc.path} -> type={doc.doc_type}, need_ocr={doc.need_ocr}")

    # A2 版面分析（mock）
    analyzer = layout_analyzer.MockLayoutAnalyzer()
    blocks = analyzer.analyze(doc.path)
    print(f"[A2 版面分析] 得到 {len(blocks)} 个区块："
          f"{[b.block_type for b in blocks]}")

    # A3 表格解析（mock）
    parser = table_parser.MockTableParser()
    for b in analyzer.extract_tables(blocks):
        tbl = parser.parse(b)
        print(f"[A3 表格独立成块] {tbl.chunk_id} | {tbl.table_no} | {tbl.text[:30]}...")

    # A6 清洗 + 脱敏
    clean = cleaner.clean_blocks(blocks)
    print(f"[A6 清洗脱敏] 保留 {len(clean)} 个有效区块")

    # A7 语义切分（doc_key 传文档标识，seq_offset 跨版面块累加，保证 chunk_id 全局唯一）
    chunks = []
    seq = 0
    for b in clean:
        cs = semantic_chunker.chunk_block(b, doc_key="mock_规程", seq_offset=seq)
        seq += len(cs)
        chunks.extend(cs)
    print(f"[A7 语义切分] 生成 {len(chunks)} 个 chunk：{[c.chunk_id for c in chunks]}")

    # A8 两级去重（演示：构造一个重复 chunk 触发分档）
    dup = semantic_chunker.Chunk("dup", chunks[0].text, chunks[0].page,
                                 meta=chunks[0].meta)
    chunks.append(dup)
    deduper = dedup.TwoStageDedup()
    decisions = deduper.run(chunks)
    for d in decisions:
        if d.tier.value != "keep_both":
            print(f"[A8 去重] {d.chunk_id} ~ {d.twin_id} -> {d.tier.value} "
                  f"(mh={d.minhash_sim:.2f}, cos={d.cosine})")

    # A9 增量索引（mock）
    idx = incremental_index.MockIncrementalIndexer()
    recs = [{"chunk_id": c.chunk_id, "text": c.text,
             "document": "mock_规程.pdf", "version": "v1"} for c in chunks]
    written = idx.upsert(recs)
    print(f"[A9 增量索引] 写入 {len(written)} 个 chunk，"
          f"当前指纹表 {len(idx.snapshot())} 条")


if __name__ == "__main__":
    main()
