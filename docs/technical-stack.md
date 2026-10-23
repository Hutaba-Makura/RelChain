# 技術スタック

## 採用構成

| 領域 | 採用技術 | 理由 |
|---|---|---|
| 言語 | Python 3.12 | 既存のMAGMA/G-Memory系実装と研究用ツール群へ接続しやすい |
| スキーマ | Pydantic 2 | LLM出力を接続した際も同じ必須項目・型検証を利用できる |
| 永続化 | SQLite 3 | 原文、版、根拠、探索ログを単一ファイルで再現可能に保存できる |
| グラフ探索 | 隣接リスト（NetworkX互換モデル） | 有向多重エッジを保持し、有限ホップ探索を軽量に実行できる |
| 標準検索 | 384次元 hashing embedding + 語彙類似度 | ネットワークやモデル取得なしで決定的なテストができる |
| 実験用検索 | `Embedder` ポートの差し替え | MiniLM等を全条件で固定して利用できる |
| LLM接続 | httpx + OpenAI互換Chat Completions + 各種ポート | OpenAI互換クラウド/ローカル推論サーバーへ同じ検証層で接続できる |
| CLI | argparse | 実験サーバーでも追加ランタイムなしで実行できる |
| テスト | pytest | 単体、統合、回帰、アブレーションの自動化 |

NetworkXは依存に含めているが、永続データをpickleへ閉じ込めないため、現行の探索処理はSQLiteから読み出した明示的な隣接リストで動く。将来の複雑な分析では同じ `Relation` を `nx.MultiDiGraph` へ損失なく変換できる。

## LLMを標準必須にしない理由

研究上、LLMによる抽出・要約・判定を使う一方、ユニットテストがAPI、モデル更新、乱数に依存すると回帰を特定できない。このため標準アダプタは決定的なルールベースとし、実験時のみ `OpenAICompatibleTransport` と検証付きLLMアダプタを注入する。LLM出力はJSONをPydanticへ通し、不正なIDや必須項目欠損なら同じ入力で1回だけ再試行し、再失敗時はpipelineが `failures` へ記録する。

## バージョン固定

各DBには `relation_mode`、writer、reader、embedding名を保存する。Relation modeが異なるDBを開こうとすると停止するため、Fixed / Free / Hybridのキャッシュ混入を防げる。正式実験ではモデルID、checkpoint、量子化、seed、temperature、プロンプトhashもmetadataへ追加する。

