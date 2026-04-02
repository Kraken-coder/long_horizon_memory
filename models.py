# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Long Horizon Memory Environment.

The long_horizon_memory environment simulates selective long-horizon memory management.
"""

from typing import List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class LongHorizonMemoryAction(Action):
    """Action to manage memory with add/remove/noop operations."""

    operation: Literal["add", "remove", "noop"] = Field(
        default="noop",
        description="Memory operation to apply at this step.",
    )
    remove_index: Optional[int] = Field(
        default=None,
        description="0-based memory index to remove when operation='remove'.",
    )


class LongHorizonMemoryObservation(Observation):
    """Observation for long_horizon_memory episodes."""

    domain: str = Field(
        default="long_horizon_memory",
        description="Conversation domain for the current episode.",
    )
    task_name: str = Field(
        default="easy",
        description="Task difficulty bucket for grading: easy, medium, or hard.",
    )
    new_message: str = Field(
        default="",
        description="The current message shown to the agent.",
    )
    memory: List[str] = Field(
        default_factory=list,
        description="Current long-term memory entries retained by the agent.",
    )
    memory_count: int = Field(
        default=0,
        description="Number of messages in memory.",
    )
    reward: float = Field(
        default=0.0,
        description="Step reward after applying the latest action.",
    )
    done: bool = Field(
        default=False,
        description="Whether the current episode is finished.",
    )