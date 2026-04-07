# Sales Email Templates

These tools manage the one-to-one email templates used in sequences. They use undocumented internal HubSpot APIs. **These are NOT the same as CMS Design Manager templates** (`hubspot_list_templates` etc.).

## Tools

| Tool | Purpose |
|---|---|
| `hubspot_search_sales_templates` | Search/list sales templates by name |
| `hubspot_get_sales_template` | Get template with subject, body HTML, and metadata |
| `hubspot_create_sales_template` | Create a new template with name, subject, and body |
| `hubspot_update_sales_template` | Update an existing template (fetches current, merges changes) |

## Workflow

```text
# List all sales templates
hubspot_search_sales_templates:
  query: ""    # empty for all, or search term

# Read a specific template
hubspot_get_sales_template:
  templateId: 249375455

# Create a new template for a campaign
hubspot_create_sales_template:
  name: "Q2 Rare Disease Outreach - Initial"
  subject: "Exploring RWD collaboration with {{ contact.company }}"
  body: "Hi {{ contact.firstname }},\n\nI noticed your work on..."

# Edit an existing template
hubspot_update_sales_template:
  templateId: 249375455
  subject: "Updated subject line"
  body: "Updated email body..."
```

## Template HTML Format

Template bodies use HTML. Key patterns from the HubSpot UI:
- Paragraphs: `<p style="margin:0;">text</p>`
- Line breaks: `<br class="hs-trailingbreak">`
- Bold: `<strong>text</strong>`
- Personalization: `{{ contact.firstname(there) }}`, `{{ contact.company }}`, `{{ contact.jobtitle }}`
- Wrapper: `<div style="" dir="auto" data-top-level="true">...</div>` (auto-added by create tool)

## HubSpot UI Links

To view/edit a template in the HubSpot UI:
```
https://app-na2.hubspot.com/templates/243878377/edit/{templateId}
```

## Email Copy Guidelines

### Sourcing Acme Experience

When writing email templates, Acme's experience in specific indications or therapy areas **must come from validated internal sources only**. Configure your own sources here — e.g. an internal research studies database and a public publications page. **Replace this section with real paths for your organization.**

Do not fabricate or assume Acme has experience in a therapy area without confirming it in one of these sources. If the user provides an external article to inform the campaign, use it for context on the target company's priorities and messaging angles, but **do not let it override or replace Acme's own validated experience claims**.

### Default Template Structure

Use this as the baseline structure for initial outreach emails. Adapt the company name, indication, and therapy area for each campaign, but keep the tone and flow consistent.

```
Hi [firstname],

I am reaching out to support [company]'s work in [condition] with real-world evidence generation.

Acme conducts research without the inefficiencies inherent in traditional research methods. Utilizing world-class AI, we recruit patients directly into studies, curate medical records from everywhere they receive care, and enable patients to participate in research virtually. This approach enables us to rapidly deploy research studies, achieve high patient retention year-over-year, and deliver high quality data compared to traditional approaches.

Would you like to learn more about our capabilities? If so, do you have availability in the next two weeks?

Best wishes,
[sender name]
```

Key principles:
- Lead with relevance to their specific work, not a generic intro about Acme
- Keep it under 200 words
- End with a clear, low-friction CTA
- Personalize using enrichment data (title, company pipeline, career history) where possible

## Internal API Notes

- Requires browser session cookies (5 cookies: `csrf.app`, `hubspotapi`, `hubspotapi-csrf`, `hubspotapi-strict`, `hubspotapi-lax`)
- May break without notice if HubSpot changes their UI
- Cookies expire and need periodic rotation via your own `rotate-hubspot-cookie` script
- Template changes do NOT affect contacts already enrolled — they use a snapshot from enrollment time
