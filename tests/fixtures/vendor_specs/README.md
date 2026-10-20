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

`clinicaltrials_v2.openapi.json` is the official ClinicalTrials.gov OpenAPI
document (https://clinicaltrials.gov/api/oas/v2, info.version 2.0.5),
fetched 2026-10-20 and trimmed to the `/studies` and `/studies/{nctId}`
paths and the components they reach. Every value is the vendor's own.
