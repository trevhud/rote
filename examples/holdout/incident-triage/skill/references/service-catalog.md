# Service catalog

Maintained by the platform team. Every production service that can page
must have a row here. The PagerDuty service name is the lookup key.

| PagerDuty service | Repo | Owning team | Incident channel | Tier | Customer-facing | Log query | Runbook |
| --- | --- | --- | --- | --- | --- | --- | --- |
| checkout-api | acme/checkout | Payments | #inc-payments | 1 | yes | `service:checkout-api status:error` | https://runbooks.example.com/checkout-api |
| payments-worker | acme/payments-worker | Payments | #inc-payments | 1 | no | `service:payments-worker status:error` | https://runbooks.example.com/payments-worker |
| search-api | acme/search | Discovery | #inc-discovery | 2 | yes | `service:search-api status:error` | https://runbooks.example.com/search-api |
| catalog-ingest | acme/catalog-ingest | Discovery | #inc-discovery | 3 | no | `service:catalog-ingest status:error` | https://runbooks.example.com/catalog-ingest |
| notifications | acme/notifications | Messaging | #inc-messaging | 2 | yes | `service:notifications status:error -env:staging` | https://runbooks.example.com/notifications |
| internal-admin | acme/admin | Platform | #inc-platform | 3 | no | `service:internal-admin status:error` | https://runbooks.example.com/internal-admin |

Notes:

- `payments-worker` is tier 1 even though it isn't customer-facing,
  because a stalled worker stops payouts.
- The log query is a Datadog log search string. Use it as written.
