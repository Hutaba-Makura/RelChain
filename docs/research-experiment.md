# RelChain 研究比較手順（結果スキーマ v2）

## 前提と構築

`locomo` は1サンプルの会話と質問を読みます。`--common-artifact` を指定した初回実行は、同じ場所に `.common.db` を作り、ルールベースのEvent・Episode・Stateと埋め込み、Relation候補を一度だけ凍結します。候補には内部カテゴリ、根拠Event、共通confidence、安定した候補IDを含みます。Fixed / Free / Hybridはこの成果物を別々のDBへ取り込みます。成果物の内容ハッシュ、ノードハッシュ、候補ハッシュが比較セットの識別子です。既存成果物の上書きはしません。方式DBの関係は再度開いても再生成しません。関係生成条件が変わった場合は別DBを指定してください。

回答モデルのローカルtokenizerを用意し、必要なら `python -m pip install transformers` を実行します。以下の `C:/models/Qwen3-8B` は実在するローカルtokenizerパスに置き換えます。実行中の回答サーバーはOpenAI互換のchat completionsを提供する必要があります。

```powershell
$common = "results/sample0.common.json"
$dataset = "C:/data/locomo10.json"
$base = "http://localhost:8000/v1"
$tokenizer = "C:/models/Qwen3-8B"
relchain locomo --dataset $dataset --sample 0 --db results/fixed.db --output results/sample0.jsonl --common-artifact $common --relation-mode fixed --relation-generation deterministic --base-url $base --model Qwen/Qwen3-8B --tokenizer $tokenizer --context-tokens 4096 --full-input-tokens 8192 --max-questions 3
relchain locomo --dataset $dataset --sample 0 --db results/free.db --output results/sample0.jsonl --common-artifact $common --relation-mode free --relation-generation llm --base-url $base --model Qwen/Qwen3-8B --relation-model Qwen/Qwen3-8B --tokenizer $tokenizer --context-tokens 4096 --full-input-tokens 8192 --max-questions 3
relchain locomo --dataset $dataset --sample 0 --db results/hybrid.db --output results/sample0.jsonl --common-artifact $common --relation-mode hybrid --relation-generation llm --base-url $base --model Qwen/Qwen3-8B --relation-model Qwen/Qwen3-8B --tokenizer $tokenizer --context-tokens 4096 --full-input-tokens 8192 --max-questions 3
```

`--max-questions` を外して同条件で再実行すると、成功済み質問を飛ばし、失敗した質問を再試行します。結果JSONLは複数runを含められます。各行の `run_id` とmanifestでrunを識別し、集計はrun単位で行います。CLIは該当runの集計を表示します。Pythonからは `EvaluationLogger("results/sample0.jsonl").aggregate(run_id, total_questions=...)` を使います。失敗行は質問ID、段階、例外種別のみ保存し、例外本文は保存しません。試行履歴は追記し、同一質問は成功記録を1件だけ採用します。

関係探索なし診断は同じ方式DBと共通成果物を使い、同じCLIに `--max-hops 0` を付けます。State・Episodeの初期検索は残ります。RelationをReader入力から外す診断は `--no-relation-context`、カテゴリ加点を全方式で無効にする診断は `--category-boost 0` です。いずれも別runになります。`--confidence-policy common` が表現比較の既定で、探索confidenceは候補値を共通にします。`generation` は生成confidenceも探索へ使う別実験です。

## 指標と分母

質問レコードは検索済み `retrieved_*_ids`、回答入力に原文が残った `context_*_ids`、入力内で検証されたモデル引用 `cited_*_ids` を別々に記録します。Turn IDは会話をまたぐ場合に `conversation_id:turn_id` 形式の `*_turn_refs` も記録します。原文が入らないState・Episode要約だけでは、そのTurnを入力済み根拠に数えません。検索・入力・引用のRecallは、正解Turn注釈との集合一致率です。各段階の全根拠取得率は、その段階のTurn集合が正解集合を包含した質問の割合です。根拠注釈がない質問は分母から除き、有効件数0なら `null` です。ID一致は注釈との一致であり、本文が回答を意味的に支持する保証ではありません。

`average_token_f1_success_only` はRelChain独自の文字・単語トークンF1で、LoCoMo公式F1ではありません。カテゴリごとに同じ計算法を使います。正解注釈がない質問はF1を `null` として分母から除きます。正解が明示的に空文字で回答も空ならF1=1、片方だけ空なら0です。明示的な `information_not_found` は回答保留として別に記録します。失敗は成功分母のF1に入れず、全対象数・成功数・失敗数・未処理数を併記します。無効引用や引用欠落は引用Recallで0、明示的な回答保留は別状態です。LLM以外のオフライン回答器の出力はモデル引用成績として集計しません。

記憶予算 `--context-tokens` は記憶部分のみです。回答入力全体は質問、指示、許可ID、チャットテンプレートを含み、`--full-input-tokens` で制限します。正式計測にはローカルの回答モデルtokenizerを `--tokenizer` で指定します。取得に失敗すると実行は失敗し、近似へ切り替えません。tokenizer未指定のオフライン実行は文字数÷4の `approximate` と明記します。ローカルtokenizerのチャットテンプレートが実際の推論サーバーと異なる場合は、入力全体の計測にも差が残ります。原文はTurn単位で採用し、収まらない大きな候補の次も検査します。

## 比較条件

| 項目 | 共通または方式差 |
| --- | --- |
| 原文、Event、Episode、State、埋め込み、Relation候補 | 共通凍結成果物 |
| 候補の内部カテゴリ、方向、対称性 | 共通。固定カテゴリ由来の候補グラフという制約を保持 |
| 探索confidence | 既定は共通候補confidence。生成confidenceは別に記録 |
| Reader、予算、seed、質問、tokenizer | 同じ値に固定しmanifestへ保存 |
| Fixed | カテゴリ名と固定関係名を使い、自由関係名LLMは呼ばない |
| Free | カテゴリを関係名生成プロンプト・探索加点・Reader表現へ渡さない |
| Hybrid | 上位カテゴリを示して自由関係名を生成する |
| カテゴリ加点 | Fixed / Hybridの方式差。`--category-boost 0` で全方式無効化可能 |

manifestにはGit commitと追跡ファイルのdirty差分ハッシュ、データ・質問・成果物ハッシュ、役割ごとの生成器とモデル名、seed、予算、検索設定、プロンプト版、結果スキーマ版を保存します。取得できないモデルrevision・量子化情報は `unknown` とします。APIキーや認証URLを保存しません。接続先URLはハッシュのみ保存します。設定変更時は別run IDになります。DB内の関係生成条件変更は拒否します。

## 旧データと検証

旧JSONLは読めますが、旧 `retrieved_turn_ids` は実際には引用Turnです。legacyとして保持し、新しい検索・入力・引用指標は復元できないため `null` にします。旧DBは自動削除・上書きしません。凍結成果物のない既存DBを共通比較DBとして自動採用せず、新しい方式DBを使用します。

外部LLMなしの小さなfixtureは次で実行できます。

```powershell
python -m pytest tests/test_research_invariants.py
```

このテストはFakeTransportと小さな対話で、複数引用、入力外IDの拒否、予算、3方式の凍結候補、再生成防止、旧結果の扱いを確認します。実サーバー・GPU・LoCoMo全件の性能確認ではありません。

## 後続課題（P2）

原文BM25・密ベクトル・正解根拠直接入力との比較、LoCoMo公式採点器との照合、会話単位bootstrapと生成反復、人手による意味的支持評価、LongMemEval、候補グラフ自体の変更、日時後処理監査はこの実装に含みません。
