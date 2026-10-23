# Compile report: docs-digest

rote wrote this report from `pipeline.yaml` and the extracted code. Each node's description is the compiler's reason for its kind; each `# skill: "..."` citation is the rule a line of code implements; each `# rote: assumption:` marker is a decision the skill does not state.

Compiled with System-1: rote decided the structure with a structure pass: claude-haiku-5-5 via claude -p (subscription) read the whole skill and proposed every node in 1 round(s), and code checked each one against the skill's text, its tools' schemas and the sample input; no decision model checked it (--decider rules), and the agent wrote the implementations and judge prompts. Every decision, with its confidence and rule, is in `system1/decisions.json`. Its model calls cost $0.0030 (claude-haiku-5-5, structure pass), on top of the agent's run.

## Summary

- 4 node(s): 2 `pure_function`, 2 `external_call`, 0 `llm_judge`, 0 `agent_loop`, 0 `hitl_gate`.
- Codified (`pure_function` + `external_call`): 4 of 4 non-gate node(s), 100%.
- Implemented impl node(s): 2 of 4.
- Entry: `search_microsoft_learn`, `read_repo_outline`.
- Exit: `write_digest`.
- Mandatory: `search_microsoft_learn`, `pick_reading_list`, `read_repo_outline`, `write_digest`.
- MCP servers: `deepwiki` (read_repo_outline), `mslearn` (search_microsoft_learn).

## Nodes

| Node | Kind | Implementation | Skill section | Description |
|---|---|---|---|---|
| `search_microsoft_learn` | external_call | `extracted/search_microsoft_learn.py:search_microsoft_learn` (stub) | 1. Search Microsoft Learn | Searches Microsoft Learn with the topic as the search text and returns the tool's decoded result, an object whose results array holds the matching articles with title and contentUrl. |
| `pick_reading_list` | pure_function | `extracted/pick_reading_list.py:pick_reading_list` (implemented) | 2. Pick the reading list | Keeps the first 5 Microsoft Learn search results, each reduced to its title and link. |
| `read_repo_outline` | external_call | `extracted/read_repo_outline.py:read_repo_outline` (stub) | 3. Read the repo's wiki outline | Calls the deepwiki read_wiki_structure tool for the repository and returns the wiki outline as text, as a string. |
| `write_digest` | pure_function | `extracted/write_digest.py:write_digest` (implemented) | 4. Write the digest | Builds the markdown digest with a Reading list section of linked titles and a Repo outline section holding the wiki outline as returned. |

## Crystallization log

- `extracted/pick_reading_list.py` (1 skill citation(s))
  - line 11: "Keep the first 5 results. For each, keep its title and its link."
- `extracted/read_repo_outline.py`: 3. Read the repo's wiki outline: direct-API backend stub. (0 skill citation(s))
- `extracted/search_microsoft_learn.py`: 1. Search Microsoft Learn: direct-API backend stub. (0 skill citation(s))
- `extracted/write_digest.py` (2 skill citation(s))
  - line 14: "one bullet per kept result, the title linked to its URL."
  - line 19: "the wiki outline, as the server returned it."

## Open questions

### Contract: search_microsoft_learn

> result shape of mslearn.microsoft_docs_search is assumed, not recorded; downstream nodes read the whole result. Run `rote baseline` against the real server to record it

### Contract: read_repo_outline

> result shape of deepwiki.read_wiki_structure is assumed, not recorded; downstream nodes read the whole result. Run `rote baseline` against the real server to record it


## Suggested next steps

- Review every Open question above; each assumption is a rule the skill does not state, and each low-confidence decision took a conservative default.
- Run the pipeline once (`rote run`) and compare it with a run of the skill.
- Recompile with `rote compile` after the skill changes. The skeleton is a model's answer (claude-haiku-5-5's structure pass), checked by code, so two compiles of one skill can differ: compare `system1/skeleton.yaml` between runs before trusting a diff.

## rote verification

- Generated tests: passed (11 passed in 0.01s), run in an OS sandbox.
- Implemented impl nodes: 2 of 4.
- Unresolved problems: 0.
- Assumptions the skill does not state: 0.
- Deterministic prefix on the sample input: 0 of 4 node(s) ran on the sample input (2 no recording, 2 not reached).
- Tool schemas: deepwiki from a live tools/list, mslearn from a live tools/list.
