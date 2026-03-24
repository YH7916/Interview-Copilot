# Demo Mock Interview

Command:

```text
/interview agent --resume "examples/demo/demo_resume.typ"
```

Transcript:

```text
Copilot: # Live Interview
  
  - mode: formal
  - questions: 4
  - rule: answer naturally, I may ask follow-ups based on your answer
  
  [Q1] [Opening | Opening and Background] Tell me about yourself and why you are applying for AI agent internships.
  - signal: 5 source(s)
  - latest: 2026-03-20

You: I am a sophomore in computer science at Zhejiang University graduating in 2028. My main project is Interview Copilot, a retrieval-augmented interview harness built on top of nanobot, and I also built CuteGo to keep browser-side inference responsive with WebWorker.

Copilot: [Q2] [Project Deep Dive | Agent Architecture and Skills] How did you design the memory boundary in Interview Copilot?
  - signal: 4 source(s)
  - latest: 2026-03-20

You: Nanobot keeps long-term memory, while copilot owns explicit runtime state and replayable traces. That keeps routing policy inspectable and avoids hiding control flow inside another memory layer.

Copilot: [Q3] [Project Deep Dive | Project Architecture and Runtime] How do scheduling and fallback work in the runtime?
  - signal: 3 source(s)
  - latest: 2026-03-20

You: We start with project-aware routing and then fall back to a simpler path when the signal is weak.

Copilot: 继续追问一个细节：
  Let's go one layer deeper. Give me the concrete switching rule when one project trail is exhausted or weak.

You: We keep an explicit goal state with active project, phase, and recent weak points. If confidence is low or a project has already been pushed far enough, the harness switches to another project or a different interview dimension instead of forcing more shallow follow-ups.

Copilot: [Q4] [Project Deep Dive | Evaluation and Review] How do you review answers and turn them into the next drill?
  - signal: 4 source(s)
  - latest: 2026-03-20

You: Each answer is scored on accuracy, clarity, depth, evidence, and structure. The weakest answers are converted into a next-drill list so the next session starts from the real gap instead of a random new question.

Copilot: Interview finished.
  - reviewed: 4
  - average_overall: 4.0/5
  - trace: demo_interview_trace.json
  - next_step: run /review for the full report

```

Review Snapshot:

# Interview Review

- reviewed: 4
- average_overall: 4.0/5

1. Tell me about yourself and why you are applying for AI agent internships.
   - overall: 4.0/5
   - accuracy: 4/5
   - reason: Clear background, strong project anchor, and a good role fit.
2. How did you design the memory boundary in Interview Copilot?
   - overall: 4.2/5
   - accuracy: 4/5
   - reason: Good boundary definition and rationale. One concrete failure case would make it even stronger.
3. How do scheduling and fallback work in the runtime?
   - overall: 3.6/5
   - accuracy: 4/5
   - reason: Routing policy is clear. Add one concrete threshold or failure example to make it more convincing.
4. How do you review answers and turn them into the next drill?
   - overall: 4.4/5
   - accuracy: 5/5
   - reason: Strong review loop with explicit scoring and a clear handoff into the next drill.

### Next Drill

1. How do scheduling and fallback work in the runtime?
   - focus: add concrete evidence, metrics, and implementation detail
   - reason: Routing policy is clear. Add one concrete threshold or failure example to make it more convincing.
2. Tell me about yourself and why you are applying for AI agent internships.
   - focus: add concrete evidence, metrics, and implementation detail
   - reason: Clear background, strong project anchor, and a good role fit.
3. How did you design the memory boundary in Interview Copilot?
   - focus: add concrete evidence, metrics, and implementation detail
   - reason: Good boundary definition and rationale. One concrete failure case would make it even stronger.

Artifacts:

- review: [demo_review.md](demo_review.md)
- interview trace: [demo_interview_trace.json](demo_interview_trace.json)
