---
name: docs-digest
description: Build a short reading list on a Microsoft technology from Microsoft Learn, plus the wiki outline of the open-source repo that goes with it.
---

# Docs digest

The input is a topic (for example "Azure Functions durable orchestration")
and a GitHub repository in `owner/name` form.

## Tools

### mslearn

| Tool | Use |
| --- | --- |
| `microsoft_docs_search` | Search Microsoft Learn |

### deepwiki

| Tool | Use |
| --- | --- |
| `read_wiki_structure` | A repository's wiki outline |

## 1. Search Microsoft Learn

Call `microsoft_docs_search` on the `mslearn` MCP server with the topic as
the search text.

## 2. Pick the reading list

Keep the first 5 results. For each, keep its title and its link.

## 3. Read the repo's wiki outline

Call `read_wiki_structure` on the `deepwiki` MCP server for the repository.

## 4. Write the digest

Return a markdown document with two sections:

- **Reading list**: one bullet per kept result, the title linked to its URL.
- **Repo outline**: the wiki outline, as the server returned it.
