# Demo Assets

This folder contains reproducible internal fixture assets for Interview Copilot.

Files:

- [demo_resume.typ](demo_resume.typ): fixed candidate snapshot used for the demo run
- [demo_prep.md](demo_prep.md): `/prep` output for the target AI agent role
- [demo_mock_interview.md](demo_mock_interview.md): deterministic live mock transcript
- [demo_review.md](demo_review.md): review summary with `Next Drill`
- [demo_prep_trace.json](demo_prep_trace.json): structured prep artifact
- [demo_interview_trace.json](demo_interview_trace.json): structured interview trace

Regenerate everything with:

```bash
python scripts/generate_demo_assets.py
```

These files are useful for tests, docs, and stable repo artifacts.

They are not the primary public demo path.

For portfolio screenshots, use the model-backed web surface launched by:

```bash
interview-demo --open-browser
```

That page talks to the real local harness and is the right surface for screenshot capture.
