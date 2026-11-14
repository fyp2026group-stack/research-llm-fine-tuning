# Research protocol — initial draft, not a validated result

**Working title:** Counterfactual Fine-Tuning for Evidence-Grounded Network Fault Diagnosis with Small Language Models.

**Question:** Can a small LLM resist misleading network fault reports while remaining responsive to genuine changes in measurements?

## Scope and hypotheses

Start with static reports and verified MQTT telemetry from a controlled network testbed. The diagnostic system must distinguish verified healthy states, verified faults, and insufficient evidence.

- **H1:** Paired-report training reduces misleading-report false alarms compared with matched-data ordinary QLoRA.
- **H2:** Reduced false alarms do not cause an unacceptable decrease in true fault detection.
- **H3:** Improvements generalize to unseen network episodes, reporting styles, and, if feasible, fault types.

## Required baselines

Untuned Qwen2.5-1.5B, evidence-first prompting, ordinary QLoRA, identical-data paired QLoRA, deterministic diagnostic rules, and any proposed custom loss term.

## Integrity controls

- Never expose injection commands or ground-truth labels in model inputs.
- Group related counterfactual variants by episode and split **episodes** before creating variants.
- Independently log actual fault-injection state, processes, timestamps, and telemetry provenance.
- Distinguish incomplete observations from evidence of normal operation.
- Predeclare primary metrics, paired comparisons, exclusion rules, training budgets and seeds.
- Retain a frozen, untouched final test set.
- Report false-alarm rate, real-fault recall, root-cause accuracy, abstention, memory and runtime.
- Treat variants from one episode as correlated observations.

## Novelty limitation

Other 2025–2026 research already addresses misleading tickets, paired counterfactual evidence and report resistance. Our candidate training objective requires further novelty audit and strong baselines before being described as a distinct scientific contribution.

## Stage B exclusions

No actual faults are injected, no models are fine-tuned, and no network diagnosis benchmark is executed.
