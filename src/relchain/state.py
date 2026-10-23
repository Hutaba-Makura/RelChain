from __future__ import annotations

from collections.abc import Sequence

from .config import MemoryConfig
from .models import (
    AssertionType,
    StateCandidate,
    StateOperation,
    StateStatus,
    StateTransition,
    UserState,
)
from .storage import SQLiteMemoryStore
from .text import stable_id


class StateUpdater:
    def __init__(self, store: SQLiteMemoryStore, config: MemoryConfig) -> None:
        self.store = store
        self.config = config

    def apply(
        self, candidate: StateCandidate
    ) -> tuple[list[UserState], list[StateTransition]]:
        related = self.store.list_states(
            subject_id=candidate.subject_id,
            attribute=candidate.attribute,
            statuses=(StateStatus.ACTIVE, StateStatus.UNCERTAIN),
        )
        related = [item for item in related if item.scope == candidate.scope]
        exact = next((item for item in related if item.value == candidate.value), None)

        if candidate.retraction_cue:
            targets = [
                item for item in related if item.value == candidate.value
            ] or related
            return self._retract(candidate, targets)
        if exact is not None:
            if set(candidate.evidence_event_ids).issubset(exact.evidence_event_ids):
                return [], []
            return self._support(candidate, exact)
        if candidate.change_cue and related:
            return self._supersede(candidate, related)
        if not related:
            return self._add(candidate, StateOperation.ADD, "no corresponding state")
        if candidate.multi_valued:
            return self._add(
                candidate, StateOperation.COEXIST, "compatible additional value"
            )
        return self._hold(candidate, related)

    def _initial_status(self, candidate: StateCandidate) -> StateStatus:
        if candidate.assertion_type == AssertionType.EXPLICIT:
            return StateStatus.ACTIVE
        independent = len(set(candidate.evidence_episode_ids))
        return (
            StateStatus.ACTIVE
            if independent >= self.config.inferred_state_min_episodes
            else StateStatus.UNCERTAIN
        )

    def _new_state(self, candidate: StateCandidate, status: StateStatus) -> UserState:
        state_id = stable_id(
            "state",
            candidate.subject_id,
            candidate.attribute,
            candidate.scope,
            candidate.value,
        )
        return UserState(
            state_id=state_id,
            subject_id=candidate.subject_id,
            attribute=candidate.attribute,
            value=candidate.value,
            scope=candidate.scope,
            valid_from=candidate.valid_from,
            valid_to=candidate.valid_to,
            recorded_at=candidate.recorded_at,
            status=status,
            assertion_type=candidate.assertion_type,
            evidence_episode_ids=candidate.evidence_episode_ids,
            evidence_event_ids=candidate.evidence_event_ids,
            counter_evidence_ids=candidate.counter_evidence_ids,
            confidence=candidate.confidence,
        )

    def _transition(
        self,
        operation: StateOperation,
        state: UserState,
        previous_revision: int | None,
        candidate: StateCandidate,
        reason: str,
    ) -> StateTransition:
        return StateTransition(
            transition_id=stable_id(
                "transition",
                state.state_id,
                state.revision,
                operation.value,
                candidate.recorded_at,
            ),
            operation=operation,
            state_id=state.state_id,
            previous_revision=previous_revision,
            new_revision=state.revision,
            recorded_at=candidate.recorded_at,
            evidence_event_ids=candidate.evidence_event_ids,
            reason=reason,
        )

    def _persist(
        self, states: Sequence[UserState], transitions: Sequence[StateTransition]
    ) -> tuple[list[UserState], list[StateTransition]]:
        for state in states:
            self.store.save_state(state)
        for transition in transitions:
            self.store.save_transition(transition)
        return list(states), list(transitions)

    def _add(
        self, candidate: StateCandidate, operation: StateOperation, reason: str
    ) -> tuple[list[UserState], list[StateTransition]]:
        state = self._new_state(candidate, self._initial_status(candidate))
        transition = self._transition(operation, state, None, candidate, reason)
        return self._persist([state], [transition])

    def _support(
        self, candidate: StateCandidate, previous: UserState
    ) -> tuple[list[UserState], list[StateTransition]]:
        episodes = tuple(
            dict.fromkeys(
                previous.evidence_episode_ids + candidate.evidence_episode_ids
            )
        )
        events = tuple(
            dict.fromkeys(previous.evidence_event_ids + candidate.evidence_event_ids)
        )
        status = previous.status
        if (
            candidate.assertion_type == AssertionType.EXPLICIT
            or len(episodes) >= self.config.inferred_state_min_episodes
        ):
            status = StateStatus.ACTIVE
        state = previous.model_copy(
            update={
                "revision": previous.revision + 1,
                "recorded_at": candidate.recorded_at,
                "status": status,
                "evidence_episode_ids": episodes,
                "evidence_event_ids": events,
                "confidence": max(previous.confidence, candidate.confidence),
            }
        )
        transition = self._transition(
            StateOperation.SUPPORT,
            state,
            previous.revision,
            candidate,
            "new supporting evidence",
        )
        return self._persist([state], [transition])

    def _supersede(
        self, candidate: StateCandidate, previous_states: Sequence[UserState]
    ) -> tuple[list[UserState], list[StateTransition]]:
        states: list[UserState] = []
        transitions: list[StateTransition] = []
        for previous in previous_states:
            old = previous.model_copy(
                update={
                    "revision": previous.revision + 1,
                    "status": StateStatus.SUPERSEDED,
                    "valid_to": candidate.valid_from or candidate.recorded_at,
                    "recorded_at": candidate.recorded_at,
                }
            )
            states.append(old)
            transitions.append(
                self._transition(
                    StateOperation.SUPERSEDE,
                    old,
                    previous.revision,
                    candidate,
                    "explicit change cue replaced prior state",
                )
            )
        new = self._new_state(candidate, StateStatus.ACTIVE)
        states.append(new)
        transitions.append(
            self._transition(
                StateOperation.SUPERSEDE,
                new,
                None,
                candidate,
                "new active value after explicit change",
            )
        )
        return self._persist(states, transitions)

    def _retract(
        self, candidate: StateCandidate, targets: Sequence[UserState]
    ) -> tuple[list[UserState], list[StateTransition]]:
        if not targets:
            return self._add(
                candidate, StateOperation.HOLD, "retraction target was not found"
            )
        states: list[UserState] = []
        transitions: list[StateTransition] = []
        for previous in targets:
            state = previous.model_copy(
                update={
                    "revision": previous.revision + 1,
                    "status": StateStatus.RETRACTED,
                    "recorded_at": candidate.recorded_at,
                    "counter_evidence_ids": tuple(
                        dict.fromkeys(
                            previous.counter_evidence_ids + candidate.evidence_event_ids
                        )
                    ),
                }
            )
            states.append(state)
            transitions.append(
                self._transition(
                    StateOperation.RETRACT,
                    state,
                    previous.revision,
                    candidate,
                    "explicit correction of a prior assertion",
                )
            )
        return self._persist(states, transitions)

    def _hold(
        self, candidate: StateCandidate, previous_states: Sequence[UserState]
    ) -> tuple[list[UserState], list[StateTransition]]:
        states: list[UserState] = []
        transitions: list[StateTransition] = []
        for previous in previous_states:
            held = previous.model_copy(
                update={
                    "revision": previous.revision + 1,
                    "status": StateStatus.UNCERTAIN,
                    "recorded_at": candidate.recorded_at,
                    "counter_evidence_ids": tuple(
                        dict.fromkeys(
                            previous.counter_evidence_ids + candidate.evidence_event_ids
                        )
                    ),
                }
            )
            states.append(held)
            transitions.append(
                self._transition(
                    StateOperation.HOLD,
                    held,
                    previous.revision,
                    candidate,
                    "conflict without an explicit change or correction",
                )
            )
        new = self._new_state(candidate, StateStatus.UNCERTAIN)
        states.append(new)
        transitions.append(
            self._transition(
                StateOperation.HOLD,
                new,
                None,
                candidate,
                "unresolved conflicting value",
            )
        )
        return self._persist(states, transitions)
