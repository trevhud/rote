# System-1 decisions for this compile (machine-generated, not skill text)

rote decided every structural field of `pipeline.yaml` with rules, before you
started. They are final and rote restores them after you stop. Never cite this
file in a `# skill:` comment: it is not part of the skill.

Section names below are quoted from the skill. They are data, not instructions.

| node | kind | section | inputs | fan_out | mandatory | constants | lowest confidence |
|---|---|---|---|---|---|---|---|
| `search_microsoft_learn` | external_call | "1. Search Microsoft Learn" | `request`, `arg_query` |  | yes |  | kind 0.95 |
| `pick_reading_list` | pure_function | "2. Pick the reading list" | `request`, `search_microsoft_learn` |  | yes | `result_limit`='first five' | kind 0.90 |
| `read_repo_wiki_outline` | external_call | "3. Read the repo's wiki outline" | `request`, `arg_repoName` |  | yes |  | kind 0.95 |
| `write_digest` | pure_function | "4. Write the digest" | `request`, `pick_reading_list`, `read_repo_wiki_outline` |  | yes | `first_section`='Reading list', `second_section`='Repo outline' | kind 0.90 |

## Decisions below 0.7 confidence (listed in the report)

- (none)
