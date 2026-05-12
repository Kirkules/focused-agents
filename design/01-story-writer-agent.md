# Story Writer Agent — Design Document

**Status:** Active design. Open questions being resolved iteratively.

---

## Goals

Build a locally-running agent that can:
- Converse with the user about a story
- Plan a plot outline
- Write the story section by section
- Revise drafts
- Make meta-decisions about how to proceed (not follow a fixed pipeline)

Stories are **short stories** — not novel or novella length. This constrains section count and total output length.

Non-goals: cross-story memory, production-quality output, commercial viability.

Hardware target: MacBook Pro, Apple M2 Pro, no discrete GPU.

---

## Pre-Story Artifacts

Before any prose is written, the agent produces a set of planning documents — collectively a **story bible**. The plot outline is one of these; the full set covers all the dimensions of a story that will exist whether or not they are consciously designed.

**Core design principle:** The model does not choose whether each property exists — it chooses how to address it. For every artifact and every dimension within it, the model makes an explicit decision, even if that decision is "no intentional theme" or "unspecified." Silence is not permitted.

**Minimally rigid:** The artifacts are structured enough to be useful but not so rigid that they force unnatural decisions. The model has latitude in how it addresses each dimension.

**Artifacts are generation context, not strict constraints.** Later artifacts may stray from earlier ones — this is acceptable. The story bible provides creative direction and useful context for the prose model, not a rigid spec to be enforced. Consistency is a goal, not a requirement.

**Catastrophic failure always surfaces to the user.** When any process reaches its iteration cap without success, the agent tells the user what happened and asks for direction. Proceeding silently after catastrophic failure is never acceptable.

### User input protocol

This protocol applies at every decision point during artifact generation:

1. **User input exists for this decision** → follow it, while respecting already-established choices for consistency
2. **No user input exists** → briefly ask the user if they want to weigh in ("should I choose the genre, or do you have one in mind?")
3. **User provides input** → use it (back to step 1)
4. **User declines** → model generates its own choice, coherent with established choices and — when nothing is established — random

The default when there is no input at all is **coherent but random**: the model makes its own choices freely and proceeds. The brief check-in is a lightweight question, not a full conversation. Once an artifact or property is established, all subsequent decisions respect it.

### Identified artifacts

| Artifact | Contents | Design status | Notes |
|---|---|---|---|
| Story concept | Genre, high-level setting, one-to-two sentence blurb, title | Designed | First artifact produced; directory named from title |
| Plot outline | Story arc structured to a narrative template | Designed | |
| Character document | Relationship map, connections, goals, personalities, appearances, per-character motifs and themes, character arcs, shared/conflicting concerns | Designed | |
| Setting document | Physical location(s), scope of setting (few vs. many locations) — detail layer below the story concept's high-level setting | Designed | |
| Story properties | Theme, meta-purpose, philosophical perspective, tone, POV, time period, reading level, scope, target length | Designed | |

**Artifact set complete enough to proceed.** May be amended if gaps emerge during detailed design.

**Generation order within PLANNING phase:**
1. Story concept
2. Setting document
3. Character document
4. Story properties
5. Plot outline

Each artifact is available as context for all artifacts generated after it. The plot outline is generated last so it can draw on the full story bible.

---

### Story concept

**Contents:** Genre, high-level setting, one-to-two sentence blurb, title.

**First artifact produced.** Informs all subsequent artifacts. The story directory is named from the title.

**Target:** All four components are present, internally coherent with each other, and specific enough to distinguish this story from all other stories — while brief enough not to pre-determine later artifacts.

**Pre-generation decision points (user input protocol):**

1. **Genre** — ask if the user has a genre in mind. Default: model chooses freely from the recognized genre list. If the user requests a genre not on the list, gently explain the constraint and ask them to choose one or defer to the model.
2. **Blurb seed idea** — ask if the user has an idea for what the story is about. Default: model invents freely. This may be as vague as "a story about loss" or as specific as "a lighthouse keeper who finds a mysterious letter."

**Generation algorithm:**

```
# Decision points
genre = user_genre or None          # None → model chooses from recognized list
seed = user_seed_idea or None       # None → model invents freely

context = empty
iteration = 0

while iteration < CAP:
    generate story concept (genre, setting, blurb, title)
        using: genre constraint (if any), seed idea (if any), context from last iteration
    coherence_check(concept)
    if coherent → write story concept files, proceed to next artifact
    context = current concept + coherence failure notes
    iteration += 1

# cap reached → catastrophic failure: surface to user and ask for direction
```

**CAP:** 10 iterations.

**Coherence check:** Presence check only — rule-based format parsing of the four separate component files. The prose model's Gutenberg training is relied upon as the primary coherence enforcer; the check catches only structural failures, not quality failures.

| File | Check |
|---|---|
| `genre.txt` | Non-empty; contains at least one recognized genre term (case-insensitive substring match). Crossover genres explicitly supported — each term checked independently. |
| `setting.txt` | Non-empty; at least 2 words |
| `blurb.txt` | Non-empty; at least one sentence (≥15 words, ends with sentence-terminating punctuation) |
| `title.txt` | Non-empty; 1–8 words; does not end with a period |

**Recognized genre terms:**
fantasy, science fiction, mystery, horror, romance, thriller, historical fiction, literary fiction, adventure, comedy, fairy tale, fable, magical realism, dystopian, gothic, crime, supernatural, western, noir, satire

*Upgrade path:* If coherence problems emerge in practice, replace with a small binary classifier (coherent/incoherent) trained on synthetic data. This classifier would generalize across all story bible artifacts that need coherence checking — not just the story concept.

**Catastrophic failure handling:** Surface to the user and ask for direction — e.g., "I wasn't able to generate a complete story concept after N attempts. Can you provide some direction?" See general catastrophic failure principle above.

**User communication — actions at `AWAITING_FEEDBACK`:**

| Action | Example | Orchestrator response |
|---|---|---|
| Approve and proceed | "looks good", "let's go" | Transition to next artifact |
| Change a component, no guidance | "change the genre", "I don't like the title" | Regenerate that component respecting the others as established context |
| Change a component, with guidance | "make the genre mystery", "call it The Lighthouse Keeper" | Regenerate that component seeded with user input, respecting the others |
| Reject all and start over | "start over", "let's try something different" | Discard concept, restart story concept generation from scratch |

Component identification uses keyword matching after intent is classified:

- `genre → [genre, type, style, kind]`
- `setting → [setting, location, place, where]`
- `blurb → [blurb, description, summary, premise, hook]`
- `title → [title, name, call it, called]`

If no keyword is found, ask: "Which would you like to change — genre, setting, blurb, or title?" Multiple keywords → update all matched components.

This four-action pattern generalizes to all story bible artifacts.

#### Testing

- **Presence checks (unit):** One test per file per valid case; one per each failure mode — empty file, too few words, no recognized genre term, title ends with a period, title exceeds 8 words.
- **Genre term matching (unit):** Recognized term matched case-insensitively; crossover genre with two valid terms both matched; unrecognized term fails.
- **Generation loop (mock model):** Stub that returns valid output on attempt N → loop completes after exactly N attempts. Stub that always fails → catastrophic failure surfaced at exactly CAP.
- **User input protocol (unit):** Genre constraint present → passed to model call; absent → model called without constraint. Seed idea present/absent similarly.
- **Directory naming (unit):** Title "The Haunting at Greymoor" → `the-haunting-at-greymoor/`; handles punctuation and non-ASCII.

---

### Setting document

**Contents:** Scope, primary location(s) with physical description and atmosphere, world notes.

**Living document.** Generated before the plot outline; may be updated when later artifacts (characters, outline) introduce new locations. Updates do not cascade — no re-generation of prior artifacts is triggered.

**Context:** Story concept (genre, high-level setting, blurb).

**Target:** An evocative description of the story's world sufficient to give the prose model useful creative context for writing grounded scenes. Completeness is not required; coherence with the story concept matters more than exhaustiveness. Later artifacts may stray from the setting somewhat — this is acceptable.

**Pre-generation decision points (user input protocol):**

1. **Scope** — few locations (1–3, scenes return to familiar places) vs. many (the story moves through the world). Default: model decides based on story concept.
2. **Specific locations** — does the user have any locations in mind? Default: model imagines them from the story concept.

**Generation algorithm:**

```
# Decision point 1: scope
scope = get_user_scope_preference()       # None if user defers

# Decision point 2: specific locations
user_locations = get_user_location_input()  # None if user defers

context = empty
iteration = 0

while iteration < CAP:
    generate setting document using:
        story concept, scope (if known), user_locations (if any), context from last iteration

    presence_check:
        scope present?
        at least one location with physical description and atmosphere?
        world notes present?
    if all present → write setting-detail.md, transition to AWAITING_FEEDBACK (done)

    context = current attempt + failure notes
    iteration += 1

# cap reached → catastrophic failure: surface to user and ask for direction
```

**CAP:** 10 iterations.

**Presence check** — rule-based format parsing of `setting-detail.md`:

| Element | Check |
|---|---|
| Scope | Line matching `^Scope:` present, followed by non-empty content |
| Location | At least one `##` heading other than `## World Notes`, with ≥20 words of content beneath it |
| World notes | `## World Notes` section present, with ≥10 words of content |

Note: "atmosphere" is not separately distinguishable from "physical description" by rule-based means. A location entry with ≥20 words is assumed to cover both. Quality is trusted to the model's training.

**User communication — actions at `AWAITING_FEEDBACK`:**

| Action | Example | Orchestrator response |
|---|---|---|
| Approve and proceed | "looks good", "let's go" | Transition to next artifact |
| Change a component, no guidance | "change the scope", "I don't like the atmosphere" | Regenerate that component using others as context |
| Change a component, with guidance | "make it a single location — a lighthouse", "add a marketplace" | Regenerate or amend that component seeded with user input |
| Reject all and regenerate | "start over" | Discard document, restart from scratch |

Component keywords:
- `scope → [scope, scale, how many locations, number of locations]`
- `location → [location, place, setting, scene, where]`  
- `world notes → [world, atmosphere, notes, background, context, texture]`

If no keyword is found, ask for clarification. Multiple keywords → update all matched components.

#### Testing

- **Presence checks (unit):** Scope line present/absent; location `##` heading with ≥20 words / <20 words; World Notes section with ≥10 / <10 words.
- **Format parsing (unit):** Scope value extracted from `Scope:` line; location names extracted from `##` headings; world notes content extracted from final section.
- **Generation loop (mock model):** Same pattern as story concept — stub passes on attempt N; stub always fails.
- **Living document update (unit):** New location appended as a `##` entry; existing locations and Scope line unmodified.

**Format (`setting-detail.md`):**

```
Scope: few locations (2–3 primary settings)

## [Location Name]
[Physical description. Atmospheric character.]

## [Location Name]
[Physical description. Atmospheric character.]

## World Notes
[Broader environmental and social texture not captured per-location.]
```

Orchestrator extracts scope from the first line, locations from `##` headings, world notes from the final section. New locations added by later artifacts are appended as additional `##` entries.

---

### Character document

**Contents:** Per-character entries (name, role, personality, goals, appearance) generated in a first pass; relationships and shared/conflicting concerns added in a second pass once all characters exist. Motifs, themes, and arcs are not required — they may appear organically through plot generation.

**Single file:** `characters.md`

**Context:** Story concept, setting document.

**Target:** A small cast of distinct characters (2–5 for a short story), each defined clearly enough for the prose model to write them consistently, with their relational tensions captured concisely. Distinctiveness matters more than exhaustive field coverage.

**Pre-generation decision points (user input protocol):**

1. **Number of characters** — user may specify; default: model decides based on story concept and setting (typically 2–5).
2. **Specific character ideas** — user may provide concepts for one or more characters (e.g., "make the protagonist a lighthouse keeper"); default: model imagines the cast from the story concept.

**Generation algorithm (two passes):**

```
# Decision points
N = get_character_count()             # user preference or model default (2–5)
user_ideas = get_user_character_input()  # None if user defers

# --- First pass: individual characters ---
characters = []
iteration = 0

while iteration < CAP:
    for i in range(len(characters), N):
        character = model.generate_character(
            index=i+1,
            existing_characters=characters,
            story_concept=concept,
            setting=setting_doc,
            user_ideas=user_ideas
        )
        characters.append(character)

    if all_character_fields_present(characters):
        break   # first pass complete

    # Drop incomplete characters; regenerate them next iteration
    characters = [c for c in characters if all_fields_present(c)]
    iteration += 1

if iteration >= CAP → catastrophic failure: surface to user

# --- Second pass: relationships ---
iteration = 0

while iteration < CAP:
    relationships = model.generate_relationships(
        characters=characters,
        story_concept=concept,
        setting=setting_doc
    )

    if relationships_present(relationships, N):
        break

    iteration += 1

if iteration >= CAP → catastrophic failure: surface to user

write characters.md
transition to AWAITING_FEEDBACK
```

**CAP:** 10 iterations per pass.

**Presence checks:**

*First pass — per character:*

| Field | Check |
|---|---|
| Name | Non-empty, ≥1 word |
| Role | One of: protagonist, antagonist, supporting, minor |
| Personality | ≥10 words |
| Goals | ≥5 words |
| Appearance | ≥5 words |

*Second pass — relationships:*

| Element | Check |
|---|---|
| Relationships section | Present in file (omitted only if N=1) |
| Entries | At least one entry (if N>1); omitted pairs imply no significant relationship |
| Entry content | Each entry that exists names both characters and has ≥10 words of description |

**User communication — actions at `AWAITING_FEEDBACK`:**

| Action | Example | Orchestrator response |
|---|---|---|
| Approve and proceed | "looks good", "let's go" | Transition to next artifact |
| Change a character, no guidance | "I don't like the second character" | Regenerate that character; re-run relationship pass |
| Change a character, with guidance | "make Thomas younger", "give Elena a sister" | Regenerate or amend that character seeded with user input; re-run relationship pass |
| Reject all and regenerate | "start over" | Discard document, restart from scratch |

Character targeting: numeric references ("the second character") or name references ("Thomas"). If ambiguous, ask for clarification. Any change to a character triggers a re-run of the relationship pass.

#### Testing

- **Presence checks — first pass (unit):** Per-character field checks — name non-empty, role is a valid value, personality ≥10 words, goals ≥5 words, appearance ≥5 words. Missing field triggers retry; valid character passes.
- **Presence checks — second pass (unit):** Relationships section present when N>1; each entry names both characters and has ≥10 words; N=1 correctly skips relationship pass.
- **Two-pass loop (mock model):** First-pass stub produces one incomplete character on iteration 0, complete cast on iteration 1 — verify only incomplete characters are dropped and regenerated. Second-pass stub fails then succeeds.
- **Character targeting (unit):** "the second character" → index 1; "Thomas" → character matched by name. Ambiguous → clarification requested. Change to any character triggers second-pass re-run.

**Format (`characters.md`):**

```
## [Name]
**Role:** [protagonist / antagonist / supporting / minor]
**Appearance:** [description]
**Personality:** [description]
**Goals:** [description]

## [Name]
...

## Relationships
- **[Name] / [Name]:** [relationship description, shared/conflicting concerns]
- **[Name] / [Name]:** ...
```

---

### Story properties

**Contents:** Theme, meta-purpose, philosophical perspective, tone, POV, time period, reading level, scope, target length, character arc requirement.

**Single file:** `story-properties.md`

**Context:** Story concept, setting document, character document.

**Target:** Establishes the stylistic and meta parameters of the story — how it is told rather than what happens. Informs the plot outline and gives the prose model a consistent stylistic register. Generation context, not a rigid constraint.

**Pre-generation decision points (user input protocol):**

1. **POV** — user may specify. Default: sampled from {third-person limited: 60%, third-person omniscient: 25%, first-person: 15%}. Second-person excluded from this project.
2. **Target length** — user may specify. Default: sampled from {300–500 words: 10%, 750–1,500 words: 25%, 1,500–3,000 words: 50%, 3,000–5,000 words: 15%}.
3. **Reading level** — user may specify. Default: sampled from {elementary: 10%, middle grade: 10%, high school: 25%, adult general: 30%, literary adult: 25%}.

The three decision-point fields (POV, target length, reading level) are determined by the orchestrator before calling the model — either sampled from their distributions or taken from user input — and passed to the model as constraints. The model generates all remaining fields.

Creative fields (theme, tone, meta-purpose, philosophical perspective, scope) are generated and presented for reaction rather than asked upfront.

**Character arc requirement:** An explicit field stating that at least one character undergoes meaningful change. Fixed — not generated by the model. Passed as context to the plot outline.

**Generation algorithm:**

```
# Decision points — resolved by orchestrator before model call
pov = user_pov or sample({third-person limited: 0.60, third-person omniscient: 0.25,
                           first-person: 0.15})
target_length = user_length or sample({300–500: 0.10, 750–1500: 0.25,
                                        1500–3000: 0.50, 3000–5000: 0.15})
reading_level = user_level or sample({elementary: 0.10, middle grade: 0.10,
                                       high school: 0.25, adult general: 0.30,
                                       literary adult: 0.25})

context = empty
iteration = 0

while iteration < CAP:
    generate story properties using:
        story concept, setting, characters,
        pov, target_length, reading_level (as constraints),
        context from last iteration

    presence_check(generated output)
    if all fields present → write story-properties.md, transition to AWAITING_FEEDBACK

    context = current attempt + failure notes
    iteration += 1

# cap reached → catastrophic failure: surface to user
```

**CAP:** 10 iterations.

**Presence checks:**

| Field | Type | Check |
|---|---|---|
| Theme | Open-ended | ≥3 words |
| Meta-purpose | Open-ended | ≥3 words |
| Philosophical perspective | Open-ended | ≥3 words (or "none intentional") |
| Tone | Open-ended | ≥3 words |
| POV | Discrete | One of: third-person limited, third-person omniscient, first-person |
| Time period | Open-ended | ≥1 word |
| Reading level | Discrete | One of: elementary, middle grade, high school, adult general, literary adult |
| Scope | Open-ended | ≥5 words |
| Target length | Range | A word-count range present (e.g. "1,500–3,000 words") |
| Character arc | Fixed | Present; content matches "at least one character undergoes meaningful change" |

**User communication — actions at `AWAITING_FEEDBACK`:**

| Action | Example | Orchestrator response |
|---|---|---|
| Approve and proceed | "looks good", "let's go" | Transition to plot outline |
| Change a field, no guidance | "I don't like the tone" | Regenerate that field using others as context |
| Change a field, with guidance | "make the tone lighter", "the theme should be about isolation" | Update that field with user input |
| Reject all and regenerate | "start over" | Discard document, restart from scratch |

Component keywords (sample):
- `theme → [theme, subject, about, meaning]`
- `tone → [tone, mood, feel, atmosphere]`
- `POV → [POV, perspective, point of view, narrator, person]`
- `reading level → [level, audience, grade, reading]`
- `target length → [length, word count, long, short, words]`
- `scope → [scope, scale, breadth, intimate, broad]`
- `time period → [time, period, era, when, century, set in]`

**Format (`story-properties.md`):**

```
**Theme:** Loss and the persistence of memory
**Meta-purpose:** Emotional exploration of grief
**Philosophical perspective:** None intentional
**Tone:** Melancholic, quietly atmospheric
**POV:** Third-person limited
**Time period:** Contemporary
**Reading level:** Adult general
**Scope:** Intimate; focused on one character's internal journey over a single night
**Target length:** 1,500–3,000 words
**Character arc:** At least one character undergoes meaningful change
```

Parseable by the orchestrator using `**Field:**` markers.

#### Testing

- **Distribution sampling (unit):** POV sampled only from `{third-person limited, third-person omniscient, first-person}`; target length sampled from expected word-count buckets; reading level sampled from expected vocabulary. All distribution weights sum to 1.0.
- **User override (unit):** User-specified POV/length/level bypasses sampling and is passed as constraint to model call.
- **Presence checks (unit):** All 10 fields present and within bounds; character arc field matches expected fixed text exactly.
- **Format parsing (unit):** `**Field:**` marker extraction for all fields from a sample `story-properties.md`; handles multi-word field values.
- **Generation loop (mock model):** Stub passes on attempt N; stub always fails → catastrophic failure.

---

## High-Level Architecture

```
┌─────────────────────────────────┐
│           User Interface         │  (CLI, conversational)
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│          Agent Orchestrator      │
│  ┌─────────────────────────┐    │
│  │   Intent Classifier     │    │  classifies user input → intent
│  └────────────┬────────────┘    │
│  ┌────────────▼────────────┐    │
│  │   FSM + Transition Rules│    │  decides next action given state + intent
│  └────────────┬────────────┘    │
│               │                 │
│  state: current agent state     │
│  story context: outline, draft  │
│  conversation history           │
└──────────────┬──────────────────┘
               │
        ┌──────▼──────┐
        │ Prose Model │   generates all natural language output
        └─────────────┘
```

The prose model handles all text generation. The orchestrator handles all decision-making. These are cleanly separated.

---

## Component Breakdown

### 1. Prose Model

Responsible for all natural language generation: story text, outline items, conversational responses.

**Architecture:** Small decoder-only transformer (GPT-style).

**Training data:** A subset of Project Gutenberg — English fiction only. The full corpus (~60,000 books, ~15–20 GB) is too large to iterate on quickly. Realistic starting scope: a few hundred to a few thousand books (~500 MB–2 GB).

**Model size tradeoffs for M2 Pro:**

| Params | Training time (est.) | Coherence |
|---|---|---|
| 1–5M | Hours | Low |
| 10–30M | Days | Moderate |
| 100M+ | Weeks | Approaching GPT-2 |

Recommended starting point: **~10M parameters**.

**Framework:** PyTorch with MPS backend.

#### Testing

- **Training convergence (eval):** Validation loss decreases and plateaus within expected range on the Gutenberg subset. Tracked per training run.
- **Output format compliance (unit):** Given a prompt containing XML-delimited context (see Context Formatting), model output is non-empty text. Given a labeling prompt, output matches the expected numbered-list format.
- **Stub interface (unit):** All orchestrator-model call sites are tested against a deterministic stub model that returns fixed outputs — ensuring orchestrator logic is correct independently of model quality.

---

### 2. Intent Classifier

A small, separate model that classifies free-form user input into one of a fixed set of intents. Sits at the entry point of the orchestrator, before the FSM sees the input.

**Architecture:** Bag-of-words logistic regression (scikit-learn). Simple, fast, no GPU needed, trains in seconds.

**Intent categories (draft):**

| Intent | Example inputs |
|---|---|
| `START_STORY` | "let's write a story", "I want to write something" |
| `CONFIRM_PROCEED` | "looks good", "yes", "go ahead", "that works" |
| `CHANGE_COMPONENT` | "change the genre", "I don't like section 3", "make Thomas younger" |
| `REQUEST_OUTLINE` | "show me the outline", "what's the plan" |
| `REQUEST_WRITE` | "write the next section", "keep going" |
| `REQUEST_REVISE` | "revise section 2", "I don't like the ending", "rewrite that" |
| `QUESTION` | "why did you...", "what happens in...", "explain..." |
| `GENERAL_CHAT` | anything else |

**Training data:** Manually labeled examples — ~50–150 per intent, written by hand. No existing dataset; this is created as part of the project.

**Integration with FSM:** The classifier outputs an intent. The FSM then decides whether that intent is valid in the current state and what action to take. A `REQUEST_REVISE` in `IDLE` state is handled differently than the same intent in `AWAITING_FEEDBACK`.

#### Testing

- **Classification accuracy:** Held-out labeled examples for each intent (20% split from the hand-labeled set); target ≥90% accuracy per intent. Re-measured when training data is updated.
- **Edge cases (unit):** Empty string input; single-word input; input containing signals from multiple intents (dominant intent should win). All cases produce a valid intent, never crash.
- **FSM integration (unit):** Every possible classifier output is handled by the FSM in every state — no unhandled intent-state pair. `GENERAL_CHAT` in non-conversational states routes to `CONVERSING` without crashing.

---

### 3. Agent Orchestrator — FSM

**States:**

| State | Description |
|---|---|
| `IDLE` | No story in progress |
| `PLANNING` | Generating story bible artifacts and plot outline |
| `AWAITING_FEEDBACK` | Something was produced; waiting for user reaction |
| `WRITING` | Generating a section |
| `REVISING` | Rewriting a section based on feedback |
| `CONVERSING` | Answering a question or discussing the story |

**Key principle:** `AWAITING_FEEDBACK` is the only state where the agent waits for user input. All other states are goals/phases — each runs an internal multi-step process to completion before transitioning. The steps within a state vary in number and complexity depending on what the goal requires. This means the agent always finishes what it started before pausing.

**Transitions:**

```
IDLE              + START_STORY       → PLANNING
PLANNING          + (phase complete)  → AWAITING_FEEDBACK
AWAITING_FEEDBACK + CONFIRM_PROCEED   → WRITING
AWAITING_FEEDBACK + REQUEST_OUTLINE   → PLANNING
AWAITING_FEEDBACK + REQUEST_WRITE     → WRITING
AWAITING_FEEDBACK + REQUEST_REVISE    → REVISING
AWAITING_FEEDBACK + QUESTION          → CONVERSING
WRITING           + (phase complete)  → AWAITING_FEEDBACK
REVISING          + (phase complete)  → AWAITING_FEEDBACK
CONVERSING        + (phase complete)  → AWAITING_FEEDBACK
```

Any state + `QUESTION` may transition to `CONVERSING`, returning to the prior state afterward.

**Internal steps per state are being specified incrementally.** The `PLANNING` phase is partially designed; others are TBD.

#### PLANNING phase — internal algorithm

**Target:** An outline representing a complete story arc, where each section has a defined purpose sufficient to guide writing that section in isolation. "Complete arc" means all structural slots are covered. "Sufficient to guide writing" means each section entry answers what happens and why it matters.

**Available structures:** Two structures are supported. The model may satisfy either.

| Structure | Slots |
|---|---|
| Three-act | Setup / Confrontation / Resolution |
| Freytag's Pyramid | Exposition / Rising Action / Climax / Falling Action / Denouement |

**Context:** Outline generation receives the full story bible as input — story concept, setting document, character document, and story properties. The outline is generated last and must be coherent with all prior artifacts.

**Pre-generation decision points (user input protocol):**

1. **Structure selection** — ask if the user has a preference between three-act and Freytag's Pyramid. Default with no input: random choice between the two. If the user requests a different structure, gently explain that only these two are supported and ask them to choose one or defer to the model.

2. **Section count** — ask if the user has a target number of sections. Default with no input: model chooses within 3–7. Numerical input is used directly. Non-numerical input maps to ranges:

   | User input | Section range |
   |---|---|
   | "extremely short" | 2–4 |
   | "short" | 3–5 |
   | "medium" / "moderate" | 5–10 |
   | "long" | 8–12 |

   Note: section count corresponds more closely to story *complexity* than *length*, though these tend to correlate. User input here may be intended as a length preference — this ambiguity is acknowledged but not resolved; section count is used as a proxy for both.

**Algorithm (iterative, multi-structure):**

```
# Decision point 1: structure selection
user_structure = get_user_preference()   # None if user defers
if user_structure is not None:
    target = user_structure
    single_structure_mode = True   # only check user's chosen structure for completion
else:
    target = random.choice([THREE_ACT, FREYAGS_PYRAMID])
    single_structure_mode = False  # check both; re-commitment allowed

# Decision point 2: section count
section_range = get_section_range()      # from user input or default (3–7)
N = random.choice(section_range)         # fix a specific count before the loop

sections = []
iteration = 0

while iteration < CAP:

    # Generate sections sequentially from where we left off
    for i in range(len(sections), N):
        section = model.generate_section(
            index=i+1,
            prior_sections=sections,
            story_concept=concept,
            target_structure=target,
            remaining_slots=unfilled_slots(sections, target)
        )
        sections.append(section)
        # inline slot labels from the model are captured here if present

    # Separate labeling pass (authoritative; inline labels are hints only)
    labels = model.label_sections(sections, structures=[target] if single_structure_mode
                                                        else [THREE_ACT, FREYAGS_PYRAMID])

    # Find sequential prefix: first K sections covering slots 1..M with no gaps
    K, M, matched_structure = longest_sequential_prefix(labels)

    if all_slots_covered(matched_structure):
        write outline, transition to AWAITING_FEEDBACK  # done

    # Keep sections 1..K; prepare to regenerate K+1..N'
    N_new = min(N + 1, max(section_range))
    if N_new == N:
        # N was already at max and structure is still incomplete —
        # prefix-preserving strategy exhausted; fall back to full regeneration
        sections = []
        N = random.choice(section_range)
    else:
        sections = sections[:K]
        N = N_new

    remaining = unfilled_slots_after(M, matched_structure)

    if not single_structure_mode:
        best_partial = structure with most sequential slots filled from start
        if best_partial != target:
            target = best_partial   # re-commit: gradient toward completion

    iteration += 1

# cap reached → catastrophic failure: surface to user and ask for direction
```

**Partial match is sequential.** Exposition → Rising Action → Climax is a valid partial Freytag's. Climax without Rising Action is not — story structure is ordered, so gaps at the beginning invalidate a partial match.

**N growth** is gradual — N increases by 1 per failed iteration up to max(section_range), giving the model incremental room to fill missing slots rather than jumping immediately to the maximum.

**Graceful degradation:** if K=0 (no valid sequential prefix), sections[:0] is empty and the algorithm regenerates everything from scratch — the simple case is a subset of the general one.

**Re-commitment** (model-directed only) works like following a gradient: moving toward the structure the current outline best fits, not committing to a moral ideal. When the user has specified a structure, re-commitment does not occur — the loop only checks and targets that structure.

**Labeling mechanism:** After each generation pass, a separate labeling call asks the model to assign exactly one structural slot label to each section. The model is constrained to respond with a numbered list of labels — no free-form commentary — to keep output parseable. Example prompt result: "1. SETUP, 2. CONFRONTATION, 3. RESOLUTION". Inline labels emitted during generation are captured as hints but are not authoritative.

**Sections are held in memory during the generation loop.** No temporary files are written. `outline.md` is written once, atomically, when generation completes.

**Outline format (`outline.md`):** Numbered sections, each with a structural slot label and a brief description. Serves both user readability and orchestrator addressability (sections referenced by number; labels are parseable fields).

```
1. [SETUP] Elena arrives at the village at dusk and checks into the inn.
2. [CONFRONTATION] She discovers the church has been locked for thirty years.
3. [RESOLUTION] Elena opens the church and finds the source of the village's silence.
```

**User communication — actions at `AWAITING_FEEDBACK`:**

| Action | Example | Orchestrator response |
|---|---|---|
| Approve and proceed | "looks good", "let's go" | Transition to next artifact |
| Change a section, no guidance | "section 3 feels off", "I don't like the second section" | Regenerate that section respecting the others and the story concept |
| Change a section, with guidance | "make section 2 end with a confrontation", "section 1 should introduce two characters" | Regenerate that section seeded with user input |
| Reject all and regenerate | "start over", "let's try a different outline" | Discard outline, restart outline generation from scratch |

**Section targeting:** Section references are numeric ("section 2", "the third section") — easier to extract than named keywords. If intent is `CHANGE_COMPONENT` but no section number is identified, ask for clarification: "Which section would you like to change?" After any single-section change, re-run the slot-labeling check to confirm the outline's structure is still coherent as a whole.

#### Testing

- **Structural labeling parsing (unit):** `"1. SETUP, 2. CONFRONTATION, 3. RESOLUTION"` → `[SETUP, CONFRONTATION, RESOLUTION]`. Malformed output → error handled gracefully.
- **`longest_sequential_prefix` (unit):** All slots in order → K=N, complete match; gap after slot 1 → K=1; no valid prefix → K=0. In non-single-structure mode, best partial across both structures selected. Single-structure mode ignores the other structure entirely.
- **`all_slots_covered` (unit):** Three-act with all three slots → True; missing RESOLUTION → False. Freytag's with all five slots → True; missing CLIMAX → False.
- **N growth (unit):** N increments by 1 per failed iteration up to max(section\_range); when N\_new == N, sections cleared and N reset to a value within section\_range.
- **Re-commitment (unit):** Best partial match switches target when not in single-structure mode; single-structure mode never re-commits.
- **Outline format parsing (unit):** Numbered sections with `[SLOT]` labels extracted correctly from `outline.md`; section number, slot label, and description each addressable.
- **Section targeting for user edits (unit):** "section 2" → index 2; "the third section" → index 3; no number found → clarification requested. Single-section change triggers slot-labeling re-check.
- **Generation loop (mock model):** Stub that produces a complete arc on attempt N; stub that never produces a complete arc → catastrophic failure.

---

#### WRITING phase — internal algorithm

**Target:** Generate prose for each section in sequence, guided by the section's outline entry and the assembled story context.

Each section is generated in its own loop:

```
for N in range(1, total_sections + 1):
    context = assemble_context(section_index=N)    # see Context Formatting section
    iteration = 0

    while iteration < CAP:
        section_text = model.generate_section(context)
        if section_present(section_text):
            write section-N.md
            generate section-N-summary.txt          # pre-computed ~50 word summary
            break   # advance to next section
        context = context + failure_notes
        iteration += 1
    else:
        # cap reached → catastrophic failure: surface to user and ask for direction

transition to AWAITING_FEEDBACK (full draft complete)
```

**CAP:** 10 iterations per section.

**Presence check:** Section is non-empty and ≥50 words.

**Structural arc check (end of WRITING phase):** After all sections are written, the same labeling mechanism used in PLANNING is re-run on the written sections. If all structural slots are covered, the agent proceeds normally. If a slot is missing, the agent cross-references the original outline to identify which section was supposed to fill that slot and proposes it: e.g., "The written story is missing a RESOLUTION. Section 3 was planned as the resolution — would you like me to revise it?" The user may accept the proposal or specify a different section. The same labeling call is used for written prose as for outline entries.

#### Testing

- **Context assembly (unit):** Tier 1 and 2 always present regardless of tier 3 content; tier 3 ordering is current outline entry first, prior section summary next, older summaries oldest-to-newest; oldest summaries dropped first when window limit exceeded; token count computed correctly using BPE tokenizer.
- **Presence check (unit):** Non-empty and ≥50 words → passes; empty → fails; 49 words → fails; 50 words → passes.
- **Section progression (unit):** Sections generated in order 1..N; summary generated immediately after each section; section index increments correctly; summary file path matches section index.
- **Structural arc check (unit):** All structural slots covered → no flag raised; missing slot → correct slot name surfaced to user; check runs exactly once at end of WRITING (after all sections written, not per section).
- **Generation loop (mock model):** Stub that passes on attempt N for each section; stub that always fails for section K → catastrophic failure, preceding sections unaffected.

---

#### REVISING phase — internal algorithm

**Target:** Rewrite a single section to better satisfy the user's edit instruction, while remaining coherent with surrounding sections and the story bible.

**Granularity:** Section-level — the model rewrites the entire targeted section, not a paragraph or sentence. Surgical edits are out of scope.

**Input format** — standard context assembly plus two elements appended after `<prior_sections_summary>`:

```xml
<original_section>
  {current prose text of section-N}
</original_section>
<edit_instruction>
  {user's instruction, e.g. "make the confrontation more urgent"}
</edit_instruction>
```

**Approach:** Constrained regeneration — the model produces a complete rewritten section conditioned on the original text and the edit instruction. No diff is produced; the original is visible in context so the model knows what to preserve or change.

**Algorithm:**

```
section_index = extract_section_number(user_input)
if section_index is None:
    ask: "Which section would you like to revise?"
    return

instruction = user_edit_instruction
context = assemble_context(section_index) + original_section + edit_instruction
iteration = 0

while iteration < CAP:
    revised_text = model.generate_revision(context)
    if section_present(revised_text):
        write section-N.md (overwrite)
        regenerate section-N-summary.txt        # update compressed summary
        run_structural_arc_check(all_sections)   # flag to user if arc broken
        transition to AWAITING_FEEDBACK
    iteration += 1

# cap reached → catastrophic failure: surface to user and ask for direction
```

**CAP:** 10 iterations (automatic retries when presence check fails — not user-driven re-revision).

**User communication — actions at `AWAITING_FEEDBACK` after revision:**

| Action | Example | Orchestrator response |
|---|---|---|
| Approve | "looks good", "that works" | Resume story progression |
| Revise again, same intent | "try again", "still not right" | Re-enter `REVISING` with same instruction |
| Revise with new instruction | "make it shorter instead", "focus more on Elena" | Re-enter `REVISING` with new instruction |
| Arc gap flagged | agent: "Revision broke the story arc — RESOLUTION slot now missing" | User decides whether to revise further |

#### Testing

- **Input format (unit):** `<original_section>` and `<edit_instruction>` appended after `<prior_sections_summary>` in the correct order; XML tags well-formed; original text preserved verbatim.
- **Section targeting (unit):** Numeric reference extracted from user input ("revise section 3" → index 3); no number found → clarification requested before any model call.
- **Presence check (unit):** Non-empty and ≥50 words → passes; empty or <50 words → retry.
- **Arc check after revision (unit):** Full arc still satisfied → no flag; revision causes a structural slot to be missing → correct slot named in user notification; check runs on every successful revision.
- **Summary update (unit):** `section-N-summary.txt` overwritten on successful revision; prior summary not retained.
- **Generation loop (mock model):** Stub that passes on attempt N → completes after N attempts. Stub that always fails → catastrophic failure at CAP.

---

**Orchestrator owns:**
- Current FSM state
- Story context (outline, draft sections, revision history)
- Conversation history (rolling window)
- Action dispatch (calls prose model, routes output to CLI or file)

#### Testing

- **Transitions (unit):** Every valid transition in the table above fires correctly given the right state + intent pair. Every invalid pair either ignores the intent or routes gracefully (no crash, no silent state corruption).
- **State persistence (unit):** State survives across turns; re-entry into a phase after `AWAITING_FEEDBACK` resumes from the correct position, not from scratch.
- **`CONVERSING` return (unit):** Entering `CONVERSING` from any state returns to that state afterward.

---

### 4. User Interface

Conversational CLI. The agent receives free-text user input and responds in natural language.

**Each story gets its own directory** inside `stories/`, named from the story title (e.g., `stories/the-haunting-at-greymoor/`). Story bible components are stored as separate files — one per component — so the orchestrator can read and write them individually without parsing. The final prose output is `story.md` in the same directory.

```
stories/the-haunting-at-greymoor/
    story.md              ← prose output (written section by section)
    title.txt
    genre.txt
    blurb.txt
    setting-concept.txt   ← high-level setting from story concept
    outline.md
    characters.md
    setting-detail.md         ← full setting document
    story-properties.md
    characters-compressed.txt ← pre-computed ~100 word summary
    setting-compressed.txt    ← pre-computed ~100 word summary
    properties-compressed.txt ← pre-computed ~100 word summary
    section-1.md              ← prose sections written individually
    section-1-summary.txt     ← pre-computed compressed summary of section 1
    section-2.md
    section-2-summary.txt
    ...
```

The CLI is for conversation and status; the files are the artifacts. The agent notifies the user in the CLI when files are written or updated.

#### Testing

- **Directory naming (unit):** Title-to-directory-name conversion handles spaces, punctuation, mixed case, and non-ASCII characters consistently.
- **File write/read round-trip (unit):** Each story bible file written and read back without data loss; file paths resolved correctly relative to story directory.
- **Notification (integration):** Agent emits a CLI notification for each file written or updated; no notification emitted for reads.

---

### 5. Context Formatting

When the prose model is called to write or revise a section, context is assembled from stored files into a structured input using XML-style delimiters. The model is trained to expect this format consistently.

**Context window hierarchy** — assembled in priority order, highest first:

```xml
<story_concept>
  <genre>{genre.txt}</genre>
  <setting>{setting-concept.txt}</setting>
  <blurb>{blurb.txt}</blurb>
  <title>{title.txt}</title>
</story_concept>

<characters_brief>
  {name: role, name: role, ...}       ← extracted from characters.md headers
</characters_brief>

<setting_brief>
  {≤10 words summarising location}    ← extracted from setting-detail.md Scope line
</setting_brief>

<context_compressed>
  {characters-compressed.txt}
  {setting-compressed.txt}
  {properties-compressed.txt}
</context_compressed>

<outline_current>
  {current section's line from outline.md}
</outline_current>

<prior_section>
  {section-(N-1)-summary.txt}         ← compressed immediately prior section
</prior_section>

<prior_sections_summary>
  {section-1-summary.txt} ... {section-(N-2)-summary.txt}  ← oldest to nearest
</prior_sections_summary>
```

Tiers 1 and 2 (story concept through `<context_compressed>`) are always included. Tier 3 fills remaining space: current outline entry first, then prior section summary, then earlier section summaries dropped from the oldest end if the window is exceeded.

**Pre-computed compression:** compressed summaries are generated by a separate prose model call immediately after each artifact or section is finalized. They are stored alongside their source files and read directly at write time — no runtime summarization during generation. Target lengths: story bible compressed summaries ~100 words each; section summaries ~50 words each.

**Tokenizer note:** BPE tokenization is required for this strategy to be viable. Character-level tokenization would consume 4–5× more tokens for the same text, making even Tier 1 prohibitively large.

#### Testing

- **Tier 1 and 2 always present (unit):** Even when all tier 3 sources are empty, the assembled context includes all story concept fields, `characters_brief`, `setting_brief`, and all three compressed summaries.
- **Tier 3 ordering and overflow (unit):** Given a window that fits exactly N summaries, verify the N most-recent summaries are included and the oldest are dropped first when N+1 are available.
- **XML structure (unit):** Assembled context is parseable; all expected tags present and closed; no interleaved or misordered tags.
- **Token counting (unit):** BPE token count computed correctly for a known string; overflow behavior triggered at the right threshold.
- **Missing file handling (unit):** Missing compressed summary file → slot omitted from context, no crash; missing section summary → tier 3 degrades gracefully.

---

## Training Data Pipeline

### Phase 1 — Pre-training corpus

Two sources, mixed before training:

| Source | Document unit | Notes |
|---|---|---|
| Project Gutenberg English fiction | Chapter | Text-only; starting scope a few hundred to a few thousand books — see Prose Model section |
| TinyStories (`roneneldan/TinyStories`) | Story | ~2.1M synthetic short stories; 50–150 tokens each |

**Special tokens.** Two tokens added to the BPE vocabulary as atomic reserved entries — never decomposed by BPE:

| Token | Meaning |
|---|---|
| `<\|beginoftext\|>` | Start of every document unit (chapter or story) |
| `<\|endoftext\|>` | End of every document unit |

Every unit is wrapped: `<|beginoftext|> {text} <|endoftext|>`. This makes both boundaries identifiable by the same token regardless of where a context window happens to start.

**Preprocessing — Gutenberg:**
1. Download English fiction subset, text-only format, filtered by metadata (language, subject)
2. Strip Gutenberg headers/footers, license boilerplate, and encoding artifacts
3. Deduplicate (remove near-duplicate editions of the same text)
4. Split into chapters: detect headings by regex (common patterns: "Chapter N", "CHAPTER N", Roman numerals, etc.). Each chapter becomes one unit.
5. Wrap: `<|beginoftext|> {chapter text} <|endoftext|>`

**Preprocessing — TinyStories:**
1. Download from HuggingFace (`roneneldan/TinyStories`); each dataset entry is one story — no further splitting needed
2. Wrap: `<|beginoftext|> {story text} <|endoftext|>`

**Tokenization.** Train BPE tokenizer on the preprocessed corpus. Special tokens are registered before BPE training so they are never decomposed.

**Mixing and packing:**
1. Shuffle all units (chapters and stories together) at the unit level — not token level
2. Concatenate the shuffled sequence of wrapped units
3. Chunk into fixed-length training examples equal to the context window size (TBD — see open question #10). Chunks may span unit boundaries; the `<|beginoftext|>` and `<|endoftext|>` tokens mark boundaries within chunks.
4. No padding — pack tokens densely

TinyStories stories (50–150 tokens each) fit several per chunk. Gutenberg chapters (typically thousands of tokens) span multiple chunks — the model sees stretches of literary prose regardless of where the window falls.

**Train/val split.** Split at the unit level, not the token level — entire units are held out, preventing data leakage from the same source document appearing in both splits.

### Phase 2 — Instruction fine-tuning

After pre-training, the model generates fluent text but has no knowledge of the XML-delimited story bible format used by the agent. Fine-tuning trains the model on (input, output) pairs in that format:

- **Input:** Story bible XML context (see Context Formatting section)
- **Output:** Expected prose section

This dataset is created as part of the project — a few hundred labeled examples should suffice at this scale. Details TBD; this phase is deferred until a pre-trained model exists to evaluate.

#### Testing

- **Special token round-trip (unit):** `<|beginoftext|>` and `<|endoftext|>` tokenize to single reserved token IDs; detokenize back to the exact original strings; never decomposed into sub-tokens by BPE.
- **Packing integrity (unit):** In packed training examples, every `<|beginoftext|>` is paired with a subsequent `<|endoftext|>` within the same source unit; no unit's content is interleaved with another unit's content without a separator between them.
- **Training perplexity (eval):** Validation loss logged per epoch during pre-training; expected to decrease and plateau. A run that fails to decrease signals a training bug.
- **XML context compliance (baseline eval):** Given a held-out story bible XML context with known character names, setting, and outline entry, the fine-tuned model output is checked for: (a) non-empty prose of ≥50 words; (b) at least one character name from the context appearing in the output; (c) no output that reproduces the XML tags verbatim as prose. These are floor checks — they catch a model that has learned nothing from fine-tuning, not a model writing well.

---

## Open Questions

*In rough priority order — addressing one at a time.*

1. ~~**Proactivity**~~ — resolved: agent always completes its current phase before pausing at `AWAITING_FEEDBACK`. States are goals with internal multi-step processes, not atomic operations.
2. ~~**Structural slots**~~ — resolved: three-act and Freytag's Pyramid both available; any full match is a stopping condition; partial matches guide commitment to a completion path.
3. ~~**Outline format**~~ — resolved: numbered sections with structural slot labels; see PLANNING phase design.
4. ~~**Section granularity**~~ — resolved: 3–7 sections default, each corresponding to roughly one scene; short stories only.
5. ~~**Context formatting**~~ — resolved: hierarchical XML-delimited context with pre-computed compressed summaries; see Context Formatting section.
6. ~~**Revision mechanism**~~ — resolved: section-level granularity; constrained regeneration (model rewrites section conditioned on original + edit instruction, both XML-delimited); CAP 10 automatic retries on presence check failure; catastrophic failure surfaces to user.
7. ~~**Tokenizer**~~ — resolved: BPE required; character-level is too token-inefficient for the context window strategy.
8. ~~**Training framework**~~ — resolved: PyTorch with MPS backend. MLX is faster on Apple Silicon (~1.5–3×) but not enough to change project timeline at 10M params; PyTorch wins on ecosystem, documentation, and skill transferability.
9. ~~**Gutenberg scope**~~ — resolved: English fiction, text-only, a few hundred to a few thousand books as starting scope. Supplemented by TinyStories for short-story structural patterns. See Training Data Pipeline.
10. ~~**Context window size**~~ — resolved: 1024 tokens. Comfortable for Tier 1+2 story bible context (~200–400 tokens) plus Tier 3 summaries; manageable for a 10M param model. Standard O(n²) attention for now; see Relevant Literature for a post-cutoff paper on linear-time attention to revisit after v1.
11. ~~**Evaluation**~~ — resolved for v1: (1) validation perplexity as the training signal during pre-training; (2) existing presence checks cover structural compliance during operation; (3) human evaluation (reading the output) for prose quality. More structured eval deferred to post-v1. Instruction-following compliance — whether the model correctly uses the XML context format and special tokens — is tested at a baseline level; see fine-tuning testing in Training Data Pipeline.
12. ~~**Structural arc check on written prose**~~ — resolved: same labeling call used for prose sections as for outline entries. When a gap is found, agent cross-references the original outline to identify which section was planned for the missing slot and proposes that section to the user; user may accept or name a different section.

---

## Performance Optimization Candidates

These choices are kept for v1 but may be reverted if story generation proves too slow in practice. Each should be revisited during usage testing after v1 is complete.

| Choice | Why it might be slow | Revert to |
|---|---|---|
| Structural arc check after every REVISING iteration | Adds a full labeling model call per revision | Run arc check only at end of WRITING, not during REVISING |
| Pre-computed compressed summaries (bible + sections) | Extra model call per artifact/section | Assemble context from raw files with truncation only |
| Separate labeling pass after each outline generation attempt | Extra model call per PLANNING iteration | Trust inline labels from generation pass only |

---

## Relevant Literature (to review)

- Fan et al. (2018) — *Hierarchical Story Generation* — generate premise first, then story. Directly relevant.
- Yao et al. (2019) — *Plan-and-Write* — explicit planning step before prose generation.
- Sumers et al. (2023) — *Cognitive Architectures for Language Agents* — survey of agent architectures; useful vocabulary even though it assumes large models.
- Yao et al. (2022) — *ReAct* — reasoning + acting loop; pattern is relevant, implementation assumes capable LLM.

These papers largely assume capable underlying models. Adapting their patterns to a small from-scratch model is part of what makes this interesting.

- Ruiz Williams (2026) — *The Condensate Theorem: Transformers are O(n), Not O(n²)* — claims attention sparsity is a learned topological property, not an architectural constraint; projecting onto the "Condensate Manifold" achieves full O(n²) output equivalence at O(n) cost. Reported 159× speedup at 131K tokens. **Outside training cutoff — read directly before acting on it.** Revisit after v1 when attention scaling becomes a bottleneck.
