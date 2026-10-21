Trimmed real Google Discovery documents, fetched from the URLs in
`google_directory.json`: Gmail v1 and Drive v3 (revision 20261012), Sheets v4
and Admin SDK Directory v1 (20261014), People v1 (20261018) and OAuth2 v2
(20200213), the last four on 2026-10-20.

Only the methods the tests use and the schemas they reach are kept; every
value is the vendor's own:

- gmail: users.threads.list, users.threads.get
- drive: files.export, files.get, files.create (media upload), files.list
- sheets: spreadsheets.get, spreadsheets.batchUpdate, spreadsheets.values.append,
  spreadsheets.values.get (stored compact: batchUpdate reaches the whole
  Request schema)
- people: people.get, people.updateContact, people.connections.list,
  contactGroups.get (shares people.get's path; flatPath tells them apart)
- admin directory_v1: users.list, users.get, customers.chrome.printers.get,
  customers.chrome.printServers.get (one `{+name}` path, two flatPaths)
- oauth2 v2: userinfo.get

The directory fixture lists the APIs above plus `admin reports_v1`, so the
Admin URL resolution has a version to choose against.

Three vendors' own OpenAPI documents, fetched 2026-10-20 and trimmed to
the paths below and the components they reach. Prose and vendor
extensions rote never reads (`description`, `summary`, `example(s)`,
`externalDocs`, `x-*` except GitHub's `x-multi-segment`) are dropped;
every value kept is the vendor's own:

- `clinicaltrials_v2.openapi.json`: ClinicalTrials.gov
  (https://clinicaltrials.gov/api/oas/v2, info.version 2.0.5):
  `/studies`, `/studies/{nctId}`, `/version`, `/stats/size`.
- `stripe_v1.openapi.json`: Stripe (github.com/stripe/openapi,
  `openapi/spec3.json`): customers (list, search, retrieve, delete),
  charges, payment_intents, refunds, checkout sessions, subscriptions,
  files (at files.stripe.com).
- `github_rest.openapi.json`: GitHub REST (github/rest-api-description,
  `api.github.com.json`): issues and comments, issue search, repos,
  contents, git refs, commits, users, pulls, check runs, workflow
  dispatch, release assets (at uploads.github.com), org attestations.

`correct_calls/<vendor>/*.py` are requests written the way each vendor's
docs say to make them, against those operations. Every one must report
zero findings (`tests/test_openapi_specs.py`); the wrong-call tests are
one-line edits of them.

`anilist.graphql` is AniList's GraphQL schema as its own endpoint
(https://graphql.anilist.co) describes it by introspection, printed as SDL
by graphql-core, fetched 2026-10-20.

`github.graphql` is GitHub's public GraphQL schema
(https://docs.github.com/public/fpt/schema.docs.graphql), fetched
2026-10-20, with its descriptions dropped (1.56 MB to 630 KB). Its
introspection without descriptions is identical to the original's, except
that four ordering input types (the gist and pull request orders and
their field enums) and the four `orderBy` arguments that take them are
removed: their names contain a string `scripts/sanity-check.sh` refuses.
Nothing in the tests uses them.

`correct_calls/github_graphql` and `correct_calls/anilist` are GraphQL
requests written the way each vendor's docs say (`tests/test_graphql_specs.py`).
