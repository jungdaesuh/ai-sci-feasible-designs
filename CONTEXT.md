Context

- The document that explains the challenge and its context
- The paper that ran this experiement: /Users/suhjungdae/code/software/proxima_fusion/RL-feasible-designs/2506.19583v1.md
- Code that the paper explains: /Users/suhjungdae/code/software/proxima_fusion/RL-feasible-designs/constellaration
- VMEC++ code: /Users/suhjungdae/code/software/proxima_fusion/RL-feasible-designs/vmecpp

# P1

1. Status: We found designs that are feasible at low fidelity, but none certified at strict/high fidelity (VMEC++).
2. Need: Reproducible infrastructure for directory structure, archival of previous runs, systematic fixes, and reliable resume points.
3. Measurement: We must record complete metrics for every evaluation so we can tell which changes help or hurt.
4. Problem so far: Lack of a systematic way to measure progress and checkpoint work has slowed iteration.
5. Need a feedback loop runs.
6. We have to start from scratch as previous code got contaminated.
