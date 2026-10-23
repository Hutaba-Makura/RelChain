# RelChain

RelChain は、長期対話記憶を次の3層で管理する研究用リファレンス実装です。

```text
Conversation / Event  ->  Episode  ->  User State
        ^                    |             |
        +---- evidence ------+-------------+
                             |
                    Fixed / Free / Hybrid Relation
```

上位の要約だけで回答せず、必ず Event と原文まで遡れること、状態の更新を上書きではなく履歴として残すこと、Episode 間 Relation の表現方式を交換可能にすることを重視しています。

## クイックスタート

```powershell
python -m pip install -e ".[dev]"
relchain init --db data/relchain.db
relchain ingest --db data/relchain.db --input examples/demo.jsonl --relation-mode hybrid
relchain query --db data/relchain.db --question "旅行で好むホテルの立地は？"
pytest
```

LoCoMo sample 0（1会話、全QA）をOpenAI互換のQwen3 8Bサーバーで実行する場合：

```powershell
$env:RELCHAIN_LLM_API_KEY = "必要な場合のみ設定"
relchain locomo `
  --dataset C:/Users/syaro/Workspaces/MAGMA/data/locomo10.json `
  --sample 0 `
  --db data/locomo_qwen3_8b_sample0.db `
  --output results/locomo_qwen3_8b_sample0.jsonl `
  --relation-mode hybrid `
  --base-url http://localhost:8000/v1 `
  --model Qwen/Qwen3-8B
```

最初は `--max-questions 3` でsmoke testし、その後オプションを外すとsample 0の199問を再開可能なJSONLとして保存します。

入力は1行1発話の JSON Lines です。

```json
{"conversation_id":"demo","turn_id":"t1","speaker_id":"user","observed_at":"2026-10-20T10:00:00+09:00","text":"旅行では駅近のホテルが好きです。"}
```

## 実装済みの範囲

- 原文の追記保存と Event の版管理
- 根拠範囲、主体、否定・予定、相対時刻を保持する Event 抽出
- セッション内 Episode 分割、要約、下層への根拠リンク
- 共通候補対に対する Fixed / Free / Hybrid Relation
- ADD / SUPPORT / COEXIST / SUPERSEDE / RETRACT / HOLD の状態更新
- Episode・User State の並列検索、有限ホップ探索、原文への遡及
- 根拠IDと不確実性を返す回答生成
- 検索トレース、Relation 統計、QA F1、Evidence Recall の評価
- 依存注入可能な抽出器・Relation生成器・回答器
- OpenAI互換/ローカル推論サーバー向けJSON LLMアダプタと1回再試行

標準実装はオフラインで動作する決定的なルールベース実装です。研究実験では `llm.py` の検証付きアダプタを各ポートへ注入し、固定した writer / reader LLM を接続できます。

詳細は [技術スタック](docs/technical-stack.md)、[構成](docs/architecture.md)、[実装プラン](docs/implementation-plan.md)、[各機能](docs/feature-implementation.md)、[テスト](docs/testing.md) を参照してください。
