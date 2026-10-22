# Second held-out set: published skills

Eight knowledge-work skills that other people wrote and published, copied
here unmodified. Like [`examples/holdout/`](../holdout/README.md), they
exist to catch overfitting, with one difference: nobody on this project
wrote them, so they test whether the compiler handles skills in the
shapes people actually publish rather than the shapes our examples
taught it.

**Never use these to tune the compiler.** Do not read them while working
on the rubric, the prompts, System-1's rules or its decision models, do
not cite them as rubric examples, and do not tune against their scores.
A change that helps the top-level examples and hurts these is fitting
the examples. Measure against them once, at a checkpoint, after the
change is decided.

| Skill | Upstream | License | What it exercises |
| --- | --- | --- | --- |
| `apollo-sequence-load` | Apollo.io, via anthropics/knowledge-work-plugins | MIT | named MCP tools with explicit arguments, a confirm-before-spending gate, batched enrichment (10 per call) looped over the approved leads |
| `hiring-screener` | Anthropic (small-business) | Apache-2.0 | a rubric-weighted judgment over every application, four bands including "cannot assess", two approval gates (replies, then interview invites) |
| `invoice-chase` | Anthropic (small-business) | Apache-2.0 | per-customer loop over AR aging, a numeric payer score, a possibly-paid exclusion across processors, a PayPal 429 retry rule, one batch approval gate |
| `crm-hygiene-check` | Anthropic (sales) | Apache-2.0 | a seven-check rules table over every opportunity, suggested fixes from evidence, read-only |
| `call-prep` | Anthropic (sales) | Apache-2.0 | parallel reads (calendar, CRM, transcripts, email, chat) into one cited brief, no writes |
| `nda-triage` | Anthropic (legal) | Apache-2.0 | a ten-section clause screen and a GREEN / YELLOW / RED classification with routing |
| `ticket-triage` | Anthropic (customer-support) | Apache-2.0 | category and P1-P4 priority from a taxonomy, a duplicate check against support, KB and tracker, routing rules |
| `hf-best-model` | Hugging Face | Apache-2.0 | public no-auth Hub API calls per benchmark leaderboard, a device memory budget, rank and filter, `hub_repo_search` (Hugging Face MCP) as the fallback |

Each `<name>/NOTICE.md` records the source URL at a pinned commit, the
author, the license and the date fetched. Each `skill/sample-input.json`
is made up for rote: an input that uses the skill's own field names,
wrapped in the `{"input", "signals"}` envelope when the skill has a gate
(`rote.compiler.grounding` reads it, see
[docs/compile-bench.md](../../docs/compile-bench.md)).

Several upstream skills link to files outside their own directory
(`../../CONNECTORS.md`, `../../shared/*.md`). Those are not copied, so the
links do not resolve, which is also what a compile sees: it gets the
skill directory and nothing above it. Each NOTICE.md names the missing
files and where they live upstream.

There are no gold labels (`expected/pipeline.yaml`) yet, so `rote bench`
does not discover these skills. Its discovery requires both
`skill/SKILL.md` and `expected/pipeline.yaml`. Until labels exist, compile
one by hand with `rote compile examples/holdout2/<name>/skill`.
Discovery on `examples/` never reaches this directory, because
`holdout2/` is not itself a skill.

Refreshing a skill from upstream is a deliberate change: update the
files, the commit and the fetch date in its NOTICE.md together.
