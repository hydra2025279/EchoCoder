#!/usr/bin/env python3
"""Compatibility entrypoint for iterative EchoCoder on ClarifyCodeBench."""

from echocoder.integrations.clarifycodebench.main import build_parser, main
from echocoder.integrations.clarifycodebench.workflow import process_task as _process_task

__all__ = ["_process_task", "build_parser", "main"]


if __name__ == "__main__":
    main()
