"""
Prompt Builder layer for SAM agent.

Exports the SYSTEM_PROMPT behavioral contract and build_prompt() assembler.
"""

from .prompt_builder import SYSTEM_PROMPT, build_prompt

__all__ = ["SYSTEM_PROMPT", "build_prompt"]
