# AGENTS.md

This document serves as the blueprint for the `quasipilot-harness` codebase. It outlines the architecture, tech stack, and the engineering principles that guide our development.

## 1. Architecture Blueprint

The codebase is organized into functional modules to ensure separation of concerns and maintainability.

```text
.
├── agents/             # Agent definitions and behavioral logic
│   ├── driver/         # Primary agent implementation
│   │   ├── agent.py    # Main orchestration logic
│   │   └── config.py   # Agent-specific configuration
│   └── tools/          # Custom tool definitions
├── cli/                # CLI entry points and UI components
│   ├── components/     # Textual UI widgets (spinner, bubbles, etc.)
│   ├── slash_commands/ # Command registry and implementations
│   └── run.py          # CLI entry point
├── core/               # Core infrastructure
│   ├── compaction/     # Memory management and synthesis logic
│   │   ├── coordinator.py # Orchestrates compaction flow
│   │   ├── prompts.py     # Prompt templates for synthesis
│   │   ├── quality.py     # Validation and sanitization logic
│   │   └── policy.py      # Compaction thresholds and settings
│   ├── middleware/     # Interceptors for agent turns
│   │   ├── reasoning.py   # Reasoning injection
│   │   └── compaction.py  # Compaction trigger middleware
│   ├── session/        # Session management and I/O
│   ├── telemetry/      # Event tracking and storage
│   └── trajectory_compaction/ # Advanced trajectory synthesis
├── evals/              # Evaluation suites
└── tests/              # Unit and integration tests
```

### Key Module Responsibilities

- **`core/compaction/`**: The engine for memory synthesis. It handles token counting, prompt engineering, and quality validation to ensure memory documents remain high-signal.
- **`core/middleware/`**: The "plumbing" layer. It intercepts agent turns to inject reasoning, telemetry, and compaction triggers.
- **`agents/driver/`**: The brain. It orchestrates the agent's behavior, utilizing middleware to maintain state and context.
- **`cli/`**: The interface. Built with `textual`, it provides a responsive, interactive shell for the agent.

## 2. Tech Stack

- **Language**: Python 3.12+
- **Agent Framework**: LangChain, LangGraph, DeepAgents
- **Storage**: ChromaDB (Vector storage)
- **CLI UI**: Textual
- **Validation**: Pydantic
- **Testing**: Pytest

## 3. Golden Commands

- **Run Agent**: `quasipilot` (or `python -m cli.run`)
- **Run Tests**: `pytest`
- **Run Evals**: `pytest evals/`

## 4. Engineering Excellence

We prioritize **simplicity, maintainability, and engineering taste**. Every contribution should adhere to the following principles:

### Software Design Principles

- **SOLID**: Adhere to Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion.
- **DRY (Don't Repeat Yourself)**: Abstract common logic into reusable components.
- **KISS (Keep It Simple, Stupid)**: Complexity is a liability. If a solution can be simple, it must be simple.
- **YAGNI (You Ain't Gonna Need It)**: Do not implement features until they are actually needed.
- **Composition over Inheritance**: Prefer composing objects to build functionality rather than deep inheritance hierarchies.

### Design Patterns

- **Middleware Pattern**: Used extensively in `core/middleware/` to intercept and modify agent turns without polluting core logic.
- **Strategy Pattern**: Used for interchangeable algorithms (e.g., different compaction strategies).
- **Factory Pattern**: Used for object creation where the exact type is determined at runtime.
- **Observer Pattern**: Used for event-driven updates (e.g., telemetry, UI updates).

### Engineering Taste

- **Senior Engineer Mindset**: Think about the long-term impact of your code. Is it easy to maintain? Is it extensible?
- **Quality First**: We value high-quality, "senior engineer" output. This applies to both code and generated memory documents.
- **Refinement**: Don't settle for the first solution. Iterate and refine until the code is clean and efficient.
- **Readability**: Code is read more often than it is written. Prioritize clarity, meaningful naming, and consistent formatting.

### Memory Management

- **Compaction**: We treat memory documents as first-class citizens. They must be structured, dense, and high-signal. Avoid "meta-process" noise in memory documents.
