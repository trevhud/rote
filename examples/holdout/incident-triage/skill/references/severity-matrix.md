# Severity matrix

Severity is decided by these rows, top to bottom. When more than one row
matches, the highest severity wins (SEV1 is highest).

| Severity | Monitor priority | Service tier | Customer-facing |
| --- | --- | --- | --- |
| SEV1 | P1 | 1 | any |
| SEV1 | P2 | 1 | yes |
| SEV2 | P1 | 2 or 3 | any |
| SEV2 | P2 | 1 or 2 | any |
| SEV2 | P3 | 1 | yes |
| SEV3 | P2 | 3 | any |
| SEV3 | P3 | any | any |
| SEV4 | P4 or P5 | any | any |

Severity to PagerDuty priority, for the update in step 9:

| Severity | PagerDuty priority |
| --- | --- |
| SEV1 | P1 |
| SEV2 | P2 |
| SEV3 | P3 |
| SEV4 | P4 |
