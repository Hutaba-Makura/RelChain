# テスト方針

## 実行

```powershell
python -m pytest
python -m pytest --cov=relchain --cov-report=term-missing
```

## レイヤー別

- schema: ID、revision、source span、self-loop拒否
- extraction: 相対時刻、modality、否定、speaker/subject分離
- episode: 境界、根拠ID、再実行時のID安定性
- relation: 3 mode、同一候補topology、方向、候補上限
- state: 6操作、明示/推定、重複根拠、旧版保持
- retrieval: 並列候補、hop上限、Free探索、原文重複排除
- persistence: append-only違反、mode混入拒否、export
- evaluation: F1、Evidence Recall、Relation統計

## 統合テスト

1. JSONL相当のturnを投入する。
2. Event→Episode→Relation→Stateが保存されたことを確認する。
3. 現在の質問と過去の質問で状態版の扱いを確認する。
4. 回答の根拠Eventが必ず保存済み原文へ到達することを確認する。
5. 同じ入力を再投入し、件数と状態revisionが増えないことを確認する。

## 実験回帰

正式LLM実験では、固定fixtureに対するnode ID集合、candidate pair集合、入力prompt hash、取得上限をsnapshotする。Original相当のFixed baselineを変更したPRは、同じfixtureの検索traceが一致しない限り受け入れない。

