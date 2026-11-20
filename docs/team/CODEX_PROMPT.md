# Prompt for a teammate's Codex agent

Copy the complete block below into Codex on your own PC. Git must be available. The agent can read and teach the project before installing Python dependencies.

```text
I am a teammate on the CropSenseAI mini-project. Help me understand the actual project and prepare to explain and run it on my PC.

Repository: https://github.com/kuntalashivasairahul/CropSense.git
Required branch: team/project-handoff

1. Clone that exact branch into a NEW folder, leaving any existing work untouched:
   git clone --single-branch --branch team/project-handoff https://github.com/kuntalashivasairahul/CropSense.git CropSense-team
   Confirm the branch and record the checked-out commit. If the folder exists, inspect it and use a different new folder rather than overwriting it. If Git LFS downloads fail, use the no-LFS clone command in docs/team/README.md in another new folder; the repository contains a checksummed model recovery ZIP.

2. Read AGENTS.md, docs/team/README.md, docs/team/PROJECT_GUIDE.md, docs/RUNTIME_CONTRACT.md, notebooks/README.md, docs/team/KAGGLE_GUIDE.md, docs/MODEL_EVALUATION.md, models/artifact_manifest.json, and docs/team/VERIFICATION.md. Then inspect app.py, core.py, scripts/, the relevant tests, all three current audit notebooks and their metadata. Use the original notebooks, older Kaggle snapshots, and linked evidence to understand history. Treat notebook output and paper/deck claims as belonging to their recorded versions; verify current behavior from code.

3. Teach me the complete flow in plain language before quizzing me. Include an architecture diagram and a table mapping each module's dataset, preprocessing, input shape/features, model, outputs, key code symbols, evidence, and limitations. Explain training versus local inference, why NDVI needs genuine aligned NIR, the pest model's prior-pressure feature, Grad-CAM's meaning, session-state invalidation, model recovery, and report generation. Walk through one example per module and a composite example where a severe component stays visible despite a high mean.

4. Explain the Kaggle history: nine included notebooks; audit sources versus original exports and snapshots; private hosted audit pages; required datasets; locked release versus fresh candidates; why rejected RF candidates, stronger pest baselines, missing original RF records, and already-inspected test sets matter. Do not claim a new experiment was executed merely because its code is present. Do not invent team roles, ownership, metrics, or field readiness. The composite is simulated and has no measured fusion accuracy.

5. After the explanation, detect my OS and Python version. Help me set up and run the demo using this branch's documented scripts and a fresh machine-local environment (64-bit Python 3.12). Initial setup is online and caches the separate ImageNet helper; full training datasets are not needed. Use the pinned dependencies. Run pip check, the unittest suite, and scripts/verify_environment.py --offline --full using that environment. Report exact commands, results, commit, and any blockers honestly. If setup cannot run here, finish the explanation and give the precise remaining local steps without claiming they passed.

6. Show how to launch the app, exercise the three modules, inspect the simulated composite, change inputs, and download a traceable report. Explain what the checks prove and which checks I must perform on this physical PC. End with a short presentation outline and likely faculty questions with answers supported by local code/evidence. Let me ask questions in my own words before starting any quiz.

This task is onboarding and local verification. Preserve the release artifacts and source. Do not retrain models, download full training datasets, edit the application, publish Kaggle notebooks, or push changes as part of this task. If later asked to implement a change, use a personal branch from this handoff branch.
```
