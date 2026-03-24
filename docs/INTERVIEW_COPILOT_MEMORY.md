# Interview Copilot Memory Boundary

## Rule

Do not run two competing long-term memory systems.

## Current Split

- `nanobot memory`
  default long-term memory layer for the runtime host
- `copilot state`
  structured, session-local control such as active project and project phase
- `copilot trace`
  replayable artifact for prep and interview workflows

## Why

Long-term memory is good for:

- durable facts
- historical context
- cross-session continuity

It is not a good place to hide:

- project routing logic
- interview policy state
- evaluation artifacts

Those belong in explicit state and trace objects.

## Practical Consequence

When adding new features:

- use `nanobot` memory for durable background knowledge
- use `copilot/interview/state.py` for runtime control
- use `copilot/interview/trace.py` for replay and evaluation artifacts

This keeps the harness inspectable and testable.
