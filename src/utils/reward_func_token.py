"""
Token-level GRPO rewards for local consistency.

The reward functions in this module operate on 5-token units. They are designed
for edit tasks where the target sequence is expected to preserve most source
units while removing or rewriting only a small local region.
"""

import math
from typing import Any, Dict, List, Sequence, Tuple

UNIT_SIZE = 5
MODEL_EOS_TOKEN_ID = 3
SEGMENT_SEARCH_MARGIN = 30

Unit = Tuple[int, ...]


def _strip_terminal_eos(token_ids: Sequence[int]) -> List[int]:
    if token_ids and token_ids[-1] == MODEL_EOS_TOKEN_ID:
        return list(token_ids[:-1])
    return list(token_ids)


def _split_into_units(tokens: Sequence[int]) -> List[Unit]:
    usable_length = (len(tokens) // UNIT_SIZE) * UNIT_SIZE
    return [
        tuple(tokens[index:index + UNIT_SIZE])
        for index in range(0, usable_length, UNIT_SIZE)
    ]


def _get_preserved_segments(source_units: List[Unit], target_units: List[Unit]) -> List[List[Unit]]:
    segments: List[List[Unit]] = []
    current_segment: List[Unit] = []
    source_index = 0
    target_index = 0

    while source_index < len(source_units) and target_index < len(target_units):
        if source_units[source_index] == target_units[target_index]:
            current_segment.append(target_units[target_index])
            source_index += 1
            target_index += 1
            continue

        if current_segment:
            segments.append(current_segment)
            current_segment = []
        source_index += 1

    if current_segment:
        segments.append(current_segment)

    return segments


def _unit_edit_distance(lhs: List[Unit], rhs: List[Unit]) -> int:
    lhs_length = len(lhs)
    rhs_length = len(rhs)
    if lhs_length == 0:
        return rhs_length
    if rhs_length == 0:
        return lhs_length

    previous_row = list(range(rhs_length + 1))
    for lhs_index in range(1, lhs_length + 1):
        current_row = [lhs_index] + [0] * rhs_length
        for rhs_index in range(1, rhs_length + 1):
            substitution_cost = 0 if lhs[lhs_index - 1] == rhs[rhs_index - 1] else 1
            current_row[rhs_index] = min(
                previous_row[rhs_index] + 1,
                current_row[rhs_index - 1] + 1,
                previous_row[rhs_index - 1] + substitution_cost,
            )
        previous_row = current_row
    return previous_row[rhs_length]


def _segment_edit_reward(
    output_units: List[Unit],
    target_units: List[Unit],
    source_units: List[Unit],
) -> float:
    preserved_segments = _get_preserved_segments(source_units, target_units)
    if not preserved_segments:
        return 0.0

    segment_score_total = 0.0
    current_search_pointer = 0

    for segment in preserved_segments:
        segment_length = len(segment)
        search_end = min(
            current_search_pointer + segment_length + SEGMENT_SEARCH_MARGIN,
            len(output_units),
        )
        output_slice = output_units[current_search_pointer:search_end]
        if not output_slice:
            continue

        edit_distance = _unit_edit_distance(output_slice, segment)
        normalizer = max(len(output_slice), segment_length, 1)
        segment_score_total += max(0.0, 1.0 - (edit_distance / normalizer))
        current_search_pointer += max(1, segment_length - UNIT_SIZE)

    return segment_score_total / len(preserved_segments)


def _consistency_reward(output_units: List[Unit], source_units: List[Unit]) -> float:
    if not output_units:
        return 0.0
    source_unit_set = set(source_units)
    preserved_unit_count = sum(1 for unit in output_units if unit in source_unit_set)
    return preserved_unit_count / len(output_units)


def _normalize_sources_and_targets(
    reward_kwargs: Dict[str, Any],
    batch_size: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    sources = reward_kwargs.get(
        "source_vq02vq06",
        reward_kwargs.get("source_token", [[]] * batch_size),
    )
    targets = reward_kwargs.get(
        "target_vq02vq06",
        reward_kwargs.get("target_token", [[]] * batch_size),
    )

    if sources and not isinstance(sources[0], (list, tuple)):
        sources = [sources] * batch_size
    if targets and not isinstance(targets[0], (list, tuple)):
        targets = [targets] * batch_size

    while len(sources) < batch_size:
        sources.append([])
    while len(targets) < batch_size:
        targets.append([])

    return sources, targets


def token_level_edit_reward_func(
    prompts: List[Any],
    completions: List[Any],
    completion_ids: List[List[int]],
    **reward_kwargs: Any,
) -> List[float]:
    batch_size = len(completion_ids)
    sources, targets = _normalize_sources_and_targets(reward_kwargs, batch_size)

    reward_values = []
    for batch_index in range(batch_size):
        output_units = _split_into_units(_strip_terminal_eos(completion_ids[batch_index]))
        target_units = _split_into_units(targets[batch_index])
        source_units = _split_into_units(sources[batch_index])
        reward_values.append(
            _segment_edit_reward(output_units, target_units, source_units)
        )
    return reward_values


def token_level_consistency_reward_func(
    prompts: List[Any],
    completions: List[Any],
    completion_ids: List[List[int]],
    **reward_kwargs: Any,
) -> List[float]:
    batch_size = len(completion_ids)
    sources, _ = _normalize_sources_and_targets(reward_kwargs, batch_size)

    return [
        _consistency_reward(
            _split_into_units(_strip_terminal_eos(completion_ids[batch_index])),
            _split_into_units(sources[batch_index]),
        )
        for batch_index in range(batch_size)
    ]


def token_level_follow_reward_func(
    prompts: List[Any],
    completions: List[Any],
    completion_ids: List[List[int]],
    **reward_kwargs: Any,
) -> List[float]:
    batch_size = len(completion_ids)
    sources, targets = _normalize_sources_and_targets(reward_kwargs, batch_size)

    reward_values = []
    for batch_index in range(batch_size):
        output_units = _split_into_units(_strip_terminal_eos(completion_ids[batch_index]))
        source_units = _split_into_units(sources[batch_index])
        target_units = _split_into_units(targets[batch_index])
        preserved_segments = _get_preserved_segments(source_units, target_units)

        hits = 0
        current_index = 0
        for segment in preserved_segments:
            anchor_unit = segment[0]
            for output_index in range(current_index, len(output_units)):
                if output_units[output_index] == anchor_unit:
                    hits += 1
                    current_index = output_index + 1
                    break

        reward_values.append(hits / len(preserved_segments) if preserved_segments else 1.0)

    return reward_values


def token_level_length_reward_func(
    prompts: List[Any],
    completions: List[Any],
    completion_ids: List[List[int]],
    **reward_kwargs: Any,
) -> List[float]:
    batch_size = len(completion_ids)
    _, targets = _normalize_sources_and_targets(reward_kwargs, batch_size)

    reward_values = []
    for batch_index in range(batch_size):
        output_tokens = _strip_terminal_eos(completion_ids[batch_index])
        target_tokens = targets[batch_index]
        if not target_tokens:
            reward_values.append(1.0 if not output_tokens else 0.0)
            continue

        token_length_difference = abs(len(output_tokens) - len(target_tokens))
        denominator = max(len(target_tokens) / UNIT_SIZE, 1) * 0.1 + 1
        reward_values.append(math.exp(-(token_length_difference / UNIT_SIZE) / denominator))

    return reward_values
