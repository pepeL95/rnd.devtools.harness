from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cli.run import QuasipilotApp


class SlashCommand(ABC):
    """Base class for chat slash commands."""

    name: str

    @abstractmethod
    def run(self, app: "QuasipilotApp", args: str) -> bool:
        """Execute the command.

        Returns True when the app should exit.
        """
