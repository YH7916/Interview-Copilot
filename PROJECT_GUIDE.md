# Project Guide

## Working Positioning

This repository should now be treated as:

- `nanobot/`: reusable runtime host
- `copilot/`: interview preparation harness

When making future changes, keep the split clear:

- `nanobot` owns generic runtime capabilities
- `copilot` owns interview-domain logic

## Product Spine

The main workflow is:

1. collect and normalize fresh interview materials
2. build local question-bank and answer-card assets
3. generate a prep pack from resume + target role
4. run a project-centric mock interview
5. review answers
6. generate the next drill list

Default runtime mode is a single online interviewer agent. Retrieval and state guide the session, but the LLM owns phrasing, clarification, and follow-up flow.

## Keep

- explicit runtime state
- a single-agent live interview loop
- retrieval-backed grounding
- traceable artifacts
- small, inspectable modules
- thin nanobot-to-copilot adapters

## Avoid

- rebuilding a second generic memory system in `copilot`
- letting question-bank retrieval turn into a rigid scripted interviewer
- hiding runtime control in prompts only
- large multi-agent abstractions without a clear need
- adding product/UI complexity before strengthening harness quality

## Stable Entry Points

- [copilot/app.py](copilot/app.py)
- [copilot/prep.py](copilot/prep.py)
- [copilot/interview/orchestrator.py](copilot/interview/orchestrator.py)
- [copilot/interview/trace.py](copilot/interview/trace.py)
- [nanobot/agent/loop.py](nanobot/agent/loop.py)

## Main Commands

```text
/ingest 7
/recent 7
/digest 1
/prep agent --resume "D:\resume\CV.typ" --company ByteDance --position "AI Agent Intern"
/interview agent --resume "D:\resume\CV.typ"
/review
```

## Packaging Docs

- [README.md](README.md)
- [INTERVIEW_COPILOT_ARCHITECTURE.md](docs/INTERVIEW_COPILOT_ARCHITECTURE.md)
- [INTERVIEW_COPILOT_MEMORY.md](docs/INTERVIEW_COPILOT_MEMORY.md)
- [INTERVIEW_COPILOT_RESUME.md](docs/INTERVIEW_COPILOT_RESUME.md)
- [INTERVIEW_COPILOT_ROADMAP.md](docs/INTERVIEW_COPILOT_ROADMAP.md)
