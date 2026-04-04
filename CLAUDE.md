## System Philosophy

- LLM is an advisor, not a decision maker
- All decisions must be validated via metrics
- System must be modular and iterative
- Prefer simple, working implementations over complex abstractions

## Constraints

- All LLM outputs must be structured JSON
- Never execute raw LLM instructions directly
- Always generate multiple variants (3–5) for improvement
- Use patience-based stopping for iteration loops


## Architecture Overview

Pipeline:

Natural Language → Problem Interpretation → Data → Pipeline → Train → Evaluate → Analyze → Iterate → Deploy

Key modules:
- problem interpreter
- preprocessing engine
- trainer + evaluator
- experiment tracker
- LLM analyzer
- controller loop

## Coding Guidelines

- Keep modules small and focused
- Avoid unnecessary abstraction
- Use clear function boundaries
- Prefer readability over cleverness

## LLM Usage Rules

- LLM is used for:
  - problem interpretation
  - pipeline suggestions
  - blind spot detection

- LLM is NOT used for:
  - executing logic
  - making final decisions

  