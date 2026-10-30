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
  --common-artifact results/sample0.common.json `
  --relation-generation llm `
  --base-url http://localhost:8000/v1 `
  --model Qwen/Qwen3-8B `
  --relation-model Qwen/Qwen3-8B `
  --tokenizer C:/models/Qwen3-8B `
  --full-input-tokens 8192
```

`--common-artifact` の初回実行では、固定方式の共通構築DBと凍結成果物を作り、指定方式のDBへ複製します。Free / Hybridで自由関係名のLLM生成を検証するには `--relation-generation llm` を指定します。Fixedでは関係名生成LLMを呼びません。`--max-questions 3` で小さく試し、同じ条件でオプションを外すと成功済み質問を飛ばして再開します。モデルや予算などを変えると別runになります。

`--tokenizer` はローカルに保存された回答モデル用Transformers tokenizerです。`--full-input-tokens` を使う場合は必須です。`transformers` のインストールが必要です。サーバー側のチャット形式がローカルtokenizerと異なる場合、その差は計測対象外です。

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
- 検索トレース、Relation 統計、独自 token_f1、3段階の根拠Recallの評価
- 依存注入可能な抽出器・Relation生成器・回答器
- OpenAI互換/ローカル推論サーバー向けJSON LLMアダプタと1回再試行

標準実装のEvent抽出、Episode分割、State抽出、関係名、回答はオフラインの決定的処理です。`locomo` CLIは回答に指定したLLMを使い、関係名は既定で決定的生成です。`--relation-generation llm` を選んだFree / Hybridのみ、別途関係生成LLMを接続します。抽出・分割・State処理は変えません。研究比較の手順、指標、互換性、FakeTransportでの確認方法は [研究実験手順](docs/research-experiment.md) を参照してください。

詳細は [技術スタック](docs/technical-stack.md)、[構成](docs/architecture.md)、[実装プラン](docs/implementation-plan.md)、[各機能](docs/feature-implementation.md)、[テスト](docs/testing.md) を参照してください。

日常の改善確認には [短時間評価の手順](docs/quick-evaluation.md) を使えます。固定質問セットを作り、全履歴の記憶・KG、検索、回答、採点を段階別に保存・再開します。`smoke` は1会話で最大10問、`dev60` は2会話で最大60問を選びます。実際の件数はデータの不足に応じてmanifestに記録されます。
