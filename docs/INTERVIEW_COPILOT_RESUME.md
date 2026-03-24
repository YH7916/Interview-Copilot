# Interview Copilot Resume Notes

## Recommended Project Title

`Interview Copilot | Retrieval-Augmented Interview Agent Harness`

Alternative wording:

`Interview Copilot | AI interview preparation agent harness`

## One-Line Pitch

Built a single-agent, retrieval-augmented interview harness that turns resumes, project context, and fresh interview materials into prep packs, project-centric mock interviews, and structured review-to-drill loops.

## Resume Bullets

### Version A

- Designed and implemented a stateful interview agent harness for AI and agent-role preparation, connecting resume parsing, project cards, retrieval-grounded question selection, and multi-turn project-centric mock interviews.
- Built a `prep -> mock -> review -> next drill` workflow: the system generates target-role prep packs, runs live interviews with project and phase tracking, and converts review results into structured follow-up drills.
- Split the system into a lightweight `nanobot` runtime layer and a `copilot` interview-domain layer, keeping long-term memory, tools, and agent loop infrastructure reusable while preserving interview state, traces, and evaluation as explicit harness logic.

### Version B

- Built Interview Copilot, a retrieval-augmented interview agent harness that turns resumes and fresh interview materials into grounded preparation packs and project-centric mock interviews.
- Implemented explicit runtime state for active project, project phase, question routing, and deterministic fallback, while letting LLM skills handle interviewer phrasing, follow-ups, clarification, and review language.
- Added prep traces, interview traces, structured review scoring, and automatic next-drill generation to make prompt, retrieval, and policy iterations replayable and measurable.

## Interviewer-Friendly Description

If an interviewer asks what is technically hard here, answer around these points:

- This is not just prompt engineering; the interesting part is the harness runtime and state control.
- Retrieval is used to ground candidate questions and preparation artifacts in real materials and resume context.
- The system explicitly models interview trajectory, project focus, follow-up policy, trace persistence, and evaluation.
- The runtime and domain layers are separated so the same agent host can support other workflows later.

## If Asked "Single-Agent Or Multi-Agent?"

Say this:

> The online runtime is intentionally single-agent. I did not want fake multi-agent complexity in the critical path. Retrieval, explicit state, and traces give the interviewer agent grounded context, while the runtime keeps policy and evaluation inspectable in code.

## If Asked "Isn't This Just Prompt + RAG?"

Say this:

> Retrieval is only one layer. The harder part is the harness: project-aware question routing, phase control, review scoring, trace persistence, and converting review output into the next drill plan. The LLM handles language and follow-ups, but code still owns control flow and artifacts.

## What To Emphasize

- `runtime`
- `state`
- `retrieval`
- `trace`
- `evaluation`
- `review -> next drill`

## What Not To Emphasize

- "I just crawled interview questions and called a prompt."
- "It is mainly a frontend or UI project."
- "It is a giant multi-agent system."

## Good Short Answer

> I built it as a retrieval-augmented interview harness on top of a lightweight agent runtime. The runtime handles tools, memory, and sessions, while the interview layer owns prep generation, project-aware multi-turn questioning, review scoring, traces, and next-drill planning.
