---
name: setup
description: >-
  Walk the user through rote's first run: import the MCP servers they
  already use, compile one of their skills, and show that the compiled skill
  gives the same answer, faster and cheaper. Use when the user says "set up
  rote", "get started with rote", "try rote on my skills", "onboard me", or
  just installed rote and asks what to do first.
---

# Set up rote

You relay `rote setup`. rote asks the questions and makes the plans, and the
user answers them. You never answer a question or approve a plan yourself,
and you never pass `--yes`.

## 1. Run it

```sh
uvx --from rote-cli rote setup --agent-mode yes --json
```

It prints one JSON object on stdout. Exit 0 or 1 with `status: "done"`
means it finished: go to step 3.

## 2. Each stop: ask, then run `next`

Exit 2 is a stop with `status: "action_required"`. Tell the user what
`ask` says, in your own plain words, with the step it belongs to
(`step.index` of `step.total`). Keep money and time in it when the plan
spends any. Then wait for their answer.

| `reason` | What the user decides | What you run |
| --- | --- | --- |
| `missing_answer` | the question in `ask`, from `choices` (`skills` and `servers` describe them) | `next[0].command`, with the placeholder replaced by their answer as `answer_with` says |
| `confirmation_required` | yes or no to the plan | yes: `next[0].command`, no: `next[1].command` |
| `authorization_pending` | approve each `logins[].authorization_url` in their browser | once they say done: each `rote mcp login --complete` command in `next`, then the last `next` command |
| `person_required` | run `next[0].command` in their own terminal (you cannot) | once they say done: the last `next` command |

Run exactly the command `next` gives. rote remembers earlier answers, so it carries only the new one.
Never add a flag it does not name. An error (exit 1, `error`) names its fix:
tell the user, and run `rote setup` again once it is fixed.

When rote asks which skill to compile (`reason: missing_answer`, step
`skill`), read the SKILL.md of each choice before you answer, and
recommend one to the user by the rule below, with one line on why. Say
which choices it rules out.

<!-- rote:worth -->
### Is it worth compiling?

Compiling to the skill runtime pays off when most of the skill's work is
fixed steps: tool calls, lookups, filtering, arithmetic, formatting. The
compiled skill does those in code and stops for you only where the skill
needs judgment. Each stop is a round trip that re-reads the conversation,
so a skill that is mostly judgment (search, read, then write an answer)
runs slower and costs more compiled. Measured: Atlassian's
search-company-knowledge took 4x longer at 3x the cost, while
expense-audit ran 3.6x faster at 62% less.

Read the SKILL.md before compiling, and count the steps that need
judgment: choosing, ranking, summarizing, writing prose. If a run needs
more than two, tell the user compiling will not pay off and why, and do
not compile unless they still want to. rote checks the same thing after
its first step and stops (`--compile-anyway` overrides it), so a
misjudgment costs cents, not a whole compile. This applies to
`--runtime skill`: on a workflow runtime a judge is an API call, not a
round trip.
<!-- /rote:worth -->

## 3. The verdict

The last object has `answer` (did the compiled skill give the same answer),
`sides` (time, tokens and cost of each), and `next` (how to use the compiled
skill from an agent). Lead with `answer`, then the time and cost of each
side, then the next step.
