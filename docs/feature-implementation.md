# 各機能の実装

## Conversation / Event

`RuleBasedEventExtractor` は文単位の `source_span` を持つEventを作る。昨日・明日・先月・来月は発話日時から解決し、時刻がない場合はnullのままにする。「したい」はdesired、「予定」はplannedで、実施済みに変換しない。明示された他者が主語の場合はspeakerとsubjectを分離する。

本番のLLM抽出器も同じ `Event` 型を返すため、存在しないturn ID、範囲外span、主体欠損を保存前に拒否できる。

## Episode

時間間隔、隣接Eventの語彙類似度、共有Entityを境界信号にする。Episode IDはEvent列から決定的に作り、再実行で同じnodeを再利用する。summary、event_ids、source_turn_ids、観測期間と出来事期間を別々に保持する。

## Relation

全組合せに生成LLMを呼ばない。時系列の近傍最大N件と、意味類似度上位N件だけを共通候補にする。同一pairにはTemporalとSemanticなど複数edgeが存在できる。Causalは対象原文に理由表現がある場合だけ候補化し、単なる前後関係を因果へ変換しない。

`RelationTextGenerator` を差し替えるとFree/Hybridの具体名だけをLLM生成できる。Fixed/Free/Hybridで候補pair生成は共通である。

## User State

明示嗜好・目標は1根拠でactiveにできる。宿泊実績等からの推定は最初の1 Episodeではuncertainで、独立Episodeが設定値に達するとSUPPORTでactiveになる。同じ原文を複数回処理しても独立証拠として重複計数しない。

- ADD: 対応状態なし
- SUPPORT: 同じ値の新根拠
- COEXIST: 複数値が両立
- SUPERSEDE: 明示的変化
- RETRACT: 過去主張の訂正
- HOLD: 解消不能な単一値矛盾

各操作は旧版を残し、transitionのreasonとevidence_event_idsを保存する。

## Retrieval / Answer

質問から現在/過去、属性、Temporal/Causal/Semantic優先度を抽出する。EpisodeとUser Stateを別々に検索し、一方が空でも検索を続ける。Relation探索は最大hop、nodeあたりneighbor数、Episode総数で必ず停止する。

取得後はEventを経由して原文turnへ遡り、同じ原文を1回だけcontextへ入れる。標準readerは取得根拠外の主張を生成しない。LLM readerへ差し替える場合も `RetrievalResult.context` と根拠IDの返却を必須にする。

## Evaluation

`evaluation.py` はtoken F1、Evidence Recall、全必要根拠取得率、information-not-found率、Relation語彙統計を提供する。探索traceにはseed Episode、状態、通過edge、最終Episode、Event、turn、LLMへ渡すRelation表現が分離して残る。

## LLM adapters

`llm.py` はEvent抽出、Episode境界・要約、State抽出、Relation自由表現、回答生成を実装する。すべての出力はPydanticで検証し、Eventのsource range、Stateのevidence ID、Relationのsnake_caseと汎用語禁止、回答の取得外ID引用を拒否する。失敗時は修正指示を付けて1回だけ再試行する。`OpenAICompatibleTransport` はAPI keyを引数で受けるが、保存やログ出力は行わない。

```python
import os

from relchain import MemoryConfig, MemorySystem
from relchain.llm import (
    LLMAnswerGenerator,
    LLMEpisodeAnnotator,
    LLMEventExtractor,
    LLMRelationTextGenerator,
    LLMStateExtractor,
    OpenAICompatibleTransport,
    ValidatedJSONCaller,
)

transport = OpenAICompatibleTransport(
    base_url="http://localhost:8000/v1",
    model="<正式なwriter/reader ID>",
    api_key=os.getenv("RELCHAIN_LLM_API_KEY"),
)
caller = ValidatedJSONCaller(transport)
system = MemorySystem(
    "data/hybrid.db",
    MemoryConfig(relation_mode="hybrid", writer_name=transport.model),
    event_extractor=LLMEventExtractor(caller),
    episode_annotator=LLMEpisodeAnnotator(caller),
    state_extractor=LLMStateExtractor(caller),
    relation_text_generator=LLMRelationTextGenerator(caller),
    answer_generator=LLMAnswerGenerator(caller),
)
```

正式実験ではwriterとreaderを別transport/callerにし、両者の正式IDをmetadataへ保存する。

