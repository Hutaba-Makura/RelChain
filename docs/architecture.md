# フォルダ構成とアーキテクチャ

```text
RelChain/
├─ src/relchain/
│  ├─ models.py       # 4種ノード、Relation、履歴、検索結果
│  ├─ config.py       # 公平性を保つ共通探索予算とRelation mode
│  ├─ ports.py        # LLM・埋め込みの交換境界
│  ├─ extraction.py   # Event / User State候補抽出の標準実装
│  ├─ episodes.py     # Episode分割、要約、根拠リンク
│  ├─ relations.py    # 共通候補対とFixed / Free / Hybrid表現
│  ├─ state.py        # 6種の状態更新と版管理
│  ├─ storage.py      # SQLite追記・最新版取得・探索ログ
│  ├─ retrieval.py    # 並列検索、関係探索、原文遡及
│  ├─ answering.py    # 根拠限定の標準reader
│  ├─ pipeline.py     # 書き込み・読み出しのユースケース
│  ├─ evaluation.py   # QA、根拠、Relation統計
│  ├─ llm.py          # 検証・再試行付き実験用LLMアダプタ
│  └─ cli.py          # 実験・検査・export
├─ tests/             # 単体、統合、回帰
├─ examples/          # JSONL入力例
└─ docs/              # 設計、実装、テスト手順
```

## 書き込み経路

```text
Turn append
  -> EventExtractor
  -> EpisodeBuilder (affected conversation)
  -> common candidate pairs
  -> RelationBuilder (one isolated mode)
  -> StateExtractor
  -> StateUpdater
  -> indexes / metadata
```

原文は `(conversation_id, turn_id)` の追記専用で、同じIDに異なる本文を書こうとすると失敗する。EventとEpisodeとUser Stateは `id + revision` で保存し、通常の読み出しは最新版を使う。上位レコードには常に下位IDがあり、EpisodeやStateだけで回答を確定しない。

## 読み出し経路

```text
QuestionAnalyzer
  ├─ Episode ranking
  └─ User State ranking
        -> relation-aware bounded traversal
        -> Event / raw Turn backtracking
        -> person/time/negation/status-aware context
        -> AnswerGenerator
```

Free modeには固定categoryを保存しない。探索は自由記述と質問の語彙適合性およびconfidenceを利用する。Fixed/Hybridではcategory一致を追加点として使う。全方式で候補数、hop数、取得総数、コンテキスト予算は同じ `MemoryConfig` に従う。

