# LoCoMo短時間評価

この経路は日常の改善判断用です。質問を減らしても、`smoke` と `dev60` は選んだ会話の**全履歴**から共通記憶とKGを作ります。初回構築時間の短縮を質問数削減の効果として扱いません。

## 固定質問セット

`locomo-set` はdataset内容のSHA-256と固定seedから、カテゴリ別・会話別に質問を選び、元のsample IDと元のquestion IDを保存します。IDがない質問は元配列位置とsample IDから安定IDを作ります。選定には回答成績を使いません。カテゴリ1の複数根拠問題は、注釈から複数セッションに根拠があると分かるものを優先します。不足した層を報告してから、他の対象カテゴリの未選択問題へ決定的に再配分します。重複や架空問題は作りません。

公開 `locomo10.json` のカテゴリIDは、1=複数根拠、2=時間、3=外部知識、4=単一根拠、5=回答不能です。開発用セットは3を除外します。この対応は[LoCoMo公式評価コード](https://github.com/snap-research/locomo/blob/main/task_eval/evaluation.py)と[公式リポジトリでのカテゴリ対応の議論](https://github.com/snap-research/locomo/issues/6)に基づきます。論文本文での列挙順をJSONのIDと読み替えません。

| プロファイル | 標準会話 | 単一根拠 | 複数根拠 | 時間 | 回答不能 |
| --- | --- | ---: | ---: | ---: | ---: |
| `smoke` | sample 0 | 3 | 3 | 2 | 2 |
| `dev60` | sample 0, 1 | 15 | 20 | 15 | 10 |
| `confirm120` | sample 2, 3, 4 | 30 | 40 | 30 | 20 |
| `final` | 明示した未使用sample | 全質問 | 全質問 | 全質問 | 全質問 |

これらは初期目標であり、統計的な十分性を示しません。`final` は既定の開発・確認用sample 0〜4を拒否します。`confirm120` を改善判断に反復使用した場合、finalへ再利用しないでください。2会話・60問は一般的なモデル順位の確定に使えません。60問の二値集計で1問は約1.67ポイントに相当し、同じ会話の質問は独立ではありません。

```powershell
relchain locomo-set --dataset C:/data/locomo10.json --profile dev60 --seed 17 --output results/dev60.set.json
```

出力には選択規則、seed、dataset hash、会話・質問ID、元の配列位置、カテゴリ、根拠注釈の有無、要求数・不足数・再配分数・実件数を保存します。質問と正解注釈は検索器やReaderに渡しません。短縮した会話を構築する `prefix_smoke` は現在提供していません。

## 構築・検索・回答・採点を分ける

`--set` とデータセットを固定したまま、以下を順に実行します。実際のローカルモデル、接続先、tokenizerのパスを指定してください。ローカルtokenizerの利用には `transformers` をインストールします。検索だけの段階では回答サーバーは不要です。検索段階の入力根拠は記憶予算を適用した時点の集合です。回答段階で入力全体の上限によってさらに根拠が落ちた場合、採点結果は回答器が実際に受け取った集合を記録します。

```powershell
relchain locomo-build-common --dataset C:/data/locomo10.json --set results/dev60.set.json --artifact-dir results/dev60-common
relchain locomo-build-mode --set results/dev60.set.json --artifact-dir results/dev60-common --mode-db-dir results/dev60-kg --relation-mode fixed
relchain locomo-build-mode --set results/dev60.set.json --artifact-dir results/dev60-common --mode-db-dir results/dev60-kg --relation-mode free --relation-generation llm --relation-base-url http://localhost:8000/v1 --relation-model Qwen/Qwen3-8B
relchain locomo-build-mode --set results/dev60.set.json --artifact-dir results/dev60-common --mode-db-dir results/dev60-kg --relation-mode hybrid --relation-generation llm --relation-base-url http://localhost:8000/v1 --relation-model Qwen/Qwen3-8B
relchain locomo-retrieve --set results/dev60.set.json --artifact-dir results/dev60-common --mode-db-dir results/dev60-kg --relation-mode hybrid --output results/dev60-hybrid-retrieval.jsonl --tokenizer C:/models/Qwen3-8B --context-tokens 4096
relchain locomo-answer --set results/dev60.set.json --retrieval results/dev60-hybrid-retrieval.jsonl --output results/dev60-hybrid-reader-a.jsonl --base-url http://localhost:8000/v1 --model Qwen/Qwen3-8B --tokenizer C:/models/Qwen3-8B --full-input-tokens 8192
relchain locomo-score --set results/dev60.set.json --answers results/dev60-hybrid-reader-a.jsonl --output results/dev60-hybrid-reader-a-score.jsonl
```

共通構築・方式別構築・検索・回答・採点はそれぞれ独立して再実行できます。同じ入力・設定では完了済み成果物と成功済み質問を再利用します。Free / HybridのLLM関係生成が途中で止まった場合、同じ条件で再実行すると保存済み候補を飛ばして残りを生成します。完成マーカーのない方式別KGは検索に使用できません。共通DBが構築途中で止まった場合は新しい `--artifact-dir` で再構築します。既存成果物を上書きしません。

KGを作らずに検索設定だけを比較する場合は、同じ `--artifact-dir` と `--mode-db-dir` で `locomo-retrieve` を再実行し、`--max-hops` などを変更して新しい `--output` を指定します。Readerだけを変更する場合は同じ検索JSONLを `locomo-answer` に渡し、新しい回答 `--output` と `--model` を指定します。Readerのtokenizerを変更した場合は、新しいtokenizerで `locomo-retrieve` から再実行して記憶予算を確認します。KGは再構築されません。回答プロンプトやモデルを変えた結果は別の回答キャッシュになります。採点器だけを変える場合は保存済み回答から採点します。

```powershell
relchain locomo-retrieve --set results/dev60.set.json --artifact-dir results/dev60-common --mode-db-dir results/dev60-kg --relation-mode hybrid --max-hops 0 --output results/dev60-hybrid-zero-hop.jsonl --tokenizer C:/models/Qwen3-8B
relchain locomo-answer --set results/dev60.set.json --retrieval results/dev60-hybrid-retrieval.jsonl --output results/dev60-hybrid-reader-b.jsonl --base-url http://localhost:8000/v1 --model AnotherReader --tokenizer C:/models/Qwen3-8B --full-input-tokens 8192
```

```powershell
relchain locomo-compare --left results/dev60-hybrid-reader-a-score.jsonl --right results/dev60-hybrid-reader-b-score.jsonl
```

比較は同じ質問セットの対応する質問だけを用い、全体・カテゴリ別・会話別の改善／悪化／同点件数を出します。独自 `token_f1` の比較であり、LoCoMo公式採点やLLM Judgeではありません。カテゴリ5は正解文章がない場合F1から除き、明示的な回答保留率を別の独自指標として記録します。差が小さい場合や会話・カテゴリで逆転する場合は優劣を確定しません。採点器を変更する場合は実装と採点キーを分け、同じ回答を再利用します。

## キャッシュと計測

共通記憶のキーは全履歴Turn、構築設定、抽出・分割・埋め込み・候補生成コードから作ります。方式別関係は共通成果物、モード、関係生成設定・モデル・プロンプトをキーにします。検索はKGの内容hash、質問セット、検索設定、tokenizer内容から、回答は保存済み検索とReader設定から、採点は保存済み回答と評価器からキーを作ります。ファイル名とmtimeだけでは再利用を判定しません。Readerだけの変更でKGを無効化しません。関係生成条件が変わった場合は新しい方式DBのディレクトリを使います。現行の意味候補生成はEpisodeの全組合せを比較します。今回の短縮はそのアルゴリズムの計算量を変えず、完成済み成果物の再利用で重複計算を省きます。

コマンドはデータ読み込み、Event抽出、Episode構築、State更新、埋め込み、候補生成、関係生成、DB書き込み、検索、入力構成、回答、採点のwall timeと件数を、取得できる範囲で別表示・保存します。構築レポートは `build-common-runs.jsonl` と `build-mode-runs.jsonl`、質問単位の時間は各段階のJSONLへ追記します。LLM呼び出し・再試行回数と、サーバーが返した場合の入力・出力token数も記録します。初回構築とキャッシュ再利用を別に扱います。短時間化の倍率や所要時間は実機・実データで測定していません。

固定fixtureで外部モデルなしに一連の動作を確認するには `python -m pytest tests/test_quick_eval.py` を実行します。
