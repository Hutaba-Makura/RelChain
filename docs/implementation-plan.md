# 実装プラン

## Phase 1: 再現可能な中核（完了）

- Conversation/Event、Episode、User State、有向多重Relationの型を確定
- SQLiteスキーマ、append-only原文、revision履歴を実装
- オフラインで決定的に動く抽出・検索・回答baselineを実装
- modeごとのDB分離とmetadataを実装

## Phase 2: 3層モデル（完了）

- Eventの主体、根拠範囲、否定、modality、相対時刻
- セッション内Episode分割、事実中心の要約、原文へのリンク
- 主張単位のUser StateとADD/SUPPORT/COEXIST/SUPERSEDE/RETRACT/HOLD
- Episode/User State並列検索、有限ホップ探索、重複原文除去

## Phase 3: Relation比較（完了）

- 時間近傍・Entity共有・意味類似度による共通候補化
- Fixed: categoryと固定relation名
- Free: categoryなしの具体的自由relation
- Hybrid: 固定categoryと具体的自由sub-relation
- 取得edge、取得node、回答へ渡した表現のログ

## Phase 4: 実験用LLMアダプタ（接続基盤完了）

1. `ports.py` の5ポートとOpenAI互換transportを実装済み。
2. 全JSON出力のPydantic検証、根拠ID検査、最大1回の再試行を実装済み。
3. 使用するwriter / reader / embeddingの正式IDを確定する。
4. prompt hash、model ID、量子化、temperature、seedをrun metadataへ追加する。
5. 同一node集合と候補pairをfreezeし、3 modeへ配布する実験コマンドを追加する。

## Phase 5: 評価（基盤完了、データ接続が次）

1. 3問 smoke test、10問、40問の順で実行する。
2. LoCoMo loaderと既存採点器の定義を照合して接続する。
3. QA F1だけでなくEvidence Recall、原文支持率、更新操作Macro-F1を保存する。
4. 会話単位bootstrapと会話別差を集計する。
5. 既存300問は開発済みデータとして扱い、最終評価を未使用会話で行う。

「モデルID、300問を生成した改変版、既存Accuracy/LLM Judgeの定義」が未確認のため、そこを推測した実装にはしていない。接続点を分け、確認後に追加しても中核データを変更しない構造にしている。

