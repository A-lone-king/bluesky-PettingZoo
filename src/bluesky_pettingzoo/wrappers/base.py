"""Base wrapper mixin for PettingZoo ParallelEnv wrappers."""

from __future__ import annotations

from typing import Any, cast


class EnvWrapperMixin:
    """Mixin class providing common delegation for ParallelEnv wrappers.

    Subclasses must set ``self.env`` in ``__init__`` before calling
    ``super().__init__()``.
    """

    def __init__(self, env: Any, **kwargs: Any) -> None:
        """Initialize the wrapper.

        Args:
            env: The PettingZoo ParallelEnv to wrap.
            **kwargs: Additional arguments for subclasses.
        """
        self.env = env
        super().__init__(**kwargs)

    @property
    def agents(self) -> list[str]:
        """List of active agent IDs."""
        return cast(list[str], self.env.agents)

    @property
    def possible_agents(self) -> list[str]:
        """List of all possible agent IDs."""
        return cast(list[str], self.env.possible_agents)

    def observation_space(self, agent: str) -> Any:
        """Get observation space for an agent.

        Args:
            agent: Agent identifier.

        Returns:
            Observation space for the agent.
        """
        return self.env.observation_space(agent)

    def action_space(self, agent: str) -> Any:
        """Get action space for an agent.

        Args:
            agent: Agent identifier.

        Returns:
            Action space for the agent.
        """
        return self.env.action_space(agent)

    def close(self) -> None:
        """Close the wrapped environment."""
        self.env.close()
