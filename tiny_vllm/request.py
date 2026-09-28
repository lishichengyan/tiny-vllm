"""Request state (Milestone 5, checkpoint 5.1).

A request carries only the state the engine actually needs:

    request_id        who am I
    prompt_tokens     the input
    generated_tokens  what has been produced so far
    status            WAITING -> RUNNING -> FINISHED
    max_tokens        stop after this many generated tokens
    block_table       my logical block i -> physical block id   (filled in by the engine)

Everything else (the model, the KV pool, the block pool) is shared between requests.
"""

import enum


class RequestStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


class Request:
    def __init__(self, request_id: int, prompt_tokens: list[int], max_tokens: int):
        if len(prompt_tokens) == 0:
            raise ValueError("prompt_tokens must not be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self.request_id = request_id
        self.prompt_tokens: list[int] = list(prompt_tokens)
        self.generated_tokens: list[int] = []
        self.status = RequestStatus.WAITING
        self.max_tokens = max_tokens
        self.block_table: list[int] = []

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_tokens)

    @property
    def num_tokens(self) -> int:
        """Current sequence length: prompt tokens + generated tokens."""
        # TODO(student): Milestone 5
        raise NotImplementedError("Milestone 5 (5.1): implement Request.num_tokens")

    @property
    def all_tokens(self) -> list[int]:
        """Prompt followed by generated tokens."""
        # TODO(student): Milestone 5
        raise NotImplementedError("Milestone 5 (5.1): implement Request.all_tokens")

    @property
    def last_token(self) -> int:
        """The most recent token of the sequence (last generated, or last prompt token if none)."""
        # TODO(student): Milestone 5
        raise NotImplementedError("Milestone 5 (5.1): implement Request.last_token")

    def append_token(self, token: int) -> None:
        """Record a newly sampled token."""
        # TODO(student): Milestone 5
        raise NotImplementedError("Milestone 5 (5.1): implement Request.append_token()")

    def should_stop(self, eos_token_id: int | None) -> bool:
        """True once generation of this request must stop.

        Stop when the most recently generated token is ``eos_token_id`` (if not None) or
        when ``max_tokens`` tokens have been generated. A request that has not generated
        anything yet never stops.
        """
        # TODO(student): Milestone 5
        raise NotImplementedError("Milestone 5 (5.1): implement Request.should_stop()")

    def __repr__(self) -> str:
        return (
            f"Request(id={self.request_id}, status={self.status.name}, "
            f"prompt={len(self.prompt_tokens)} tok, generated={len(self.generated_tokens)} tok, "
            f"block_table={self.block_table})"
        )
