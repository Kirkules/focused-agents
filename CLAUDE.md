# focused-agents

Experimental project exploring focused, minimal agentic AI architectures — built from scratch, no external AI providers or APIs.

## Goals
- Learn agentic infrastructure and architecture by building it
- Keep implementations small and understandable
- Run everything locally on this machine

## Language
Undecided. Prefer Python for rapid prototyping; C++ or Java when performance matters. Choose the best fit per component.

## Collaboration
- If the user does not respond to something raised, assume they missed it, forgot it, or were focused elsewhere — not that they decided against it. Re-raise it when appropriate.

## Priorities
- **Honesty first.** When in doubt, say what's true — about the code, the approach, and what I don't know. Don't rationalize or soften to avoid friction.
- **Testing is required.** Every non-trivial component needs tests.
- Minimal package management and tooling overhead — this is exploratory.
- Prefer simple, readable implementations over abstractions.
- No external AI providers. Models and agent architecture are built here.

## Code commenting
Comments should be complete and instructive — not sparse, not redundant with obvious naming.
- Every function gets at least a brief comment explaining what it does.
- If inputs or outputs are not self-evident from names and types, document them.
- Functions of 10 or more lines include at least a few inline comments breaking down logical sections of the algorithm.
