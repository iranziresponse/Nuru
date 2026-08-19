# Connecting Nuru to Zapier, Make, or your own systems

Nuru doesn't (yet) have a published Zapier or Make app. Building one
means creating a developer account with them and going through their
review process, which isn't something that happens from a codebase
alone. What's here today, verified working: a generic webhook that any
automation tool with a "Webhooks" or "Catch Hook" trigger can receive,
and two other automation paths that need no third-party tool at all.

## What gets sent

When a document's approved fields are sent to a webhook, Nuru POSTs a
single JSON object. Every approved field appears as `"Label": "Value"`,
plus two fixed metadata keys. Real example, from an actual invoice run
through Nuru this session:

```json
{
  "Supplier Name": "Quantum Electronics",
  "Transaction Date": "2026-04-02",
  "Total Amount Due": "$1,339.20",
  "Tax Value (VAT/GST)": "$99.20",
  "_document": "invoice.pdf",
  "_type": "Invoice"
}
```

The field names depend on the document's inferred type: a receipt sends
`"Merchant"` / `"Purchase Date"` / `"Amount Paid"` / `"Payment Method"`
instead, a statement sends `"Account Name"` / `"Statement Period"` /
`"Closing Balance"`. Any custom fields added during review are included
under whatever label was given them. **Use the "Preview payload" button
on the review screen to see the exact JSON for the specific document
you're about to send**. Don't guess at field names when setting up a
Zap or Scenario; copy them from a real preview.

## Zapier setup

1. Create a Zap. For the trigger, search for **"Webhooks by Zapier"** and
   choose **"Catch Hook."**
2. Zapier gives you a URL like
   `https://hooks.zapier.com/hooks/catch/XXXXXXX/YYYYYYY/`. Paste that
   into Nuru's review screen when you choose "Send to a webhook"; it's
   remembered as the default for next time via `NURU_WEBHOOK_URL` (see
   `nuru.local.env.example`), so you only have to do this once.
3. Send one real document through Nuru to that webhook. Back in Zapier,
   click **"Test trigger"**. You'll see the actual field names/values
   from that document, ready to map into whatever action you build next
   (a row in a spreadsheet, a record in your accounting software, etc.).
4. Turn the Zap on.

## Make (Integromat) setup

1. Create a Scenario. Add a **"Webhooks"** module, choose
   **"Custom webhook,"** and create a new webhook; Make gives you a URL.
2. Paste that URL into Nuru the same way as above.
3. Send one document through; Make will show the received payload's
   structure, which you can then map in subsequent modules.
4. Turn the Scenario on.

## No-tool alternatives

- **Email**: send the approved fields directly to an inbox (an
  accounts-payable alias, for example) with no Zapier/Make account
  needed. Set `NURU_DEFAULT_EMAIL_TO` so it's pre-filled.
- **Archive**: rename and move the source PDF into an approved folder.
  Set `NURU_ARCHIVE_DIR` (and `NURU_ARCHIVE_ALLOWED_ROOTS` for
  additional approved locations); see the main README's Security
  section for why this is an allowlist rather than a free-text path.
- **Download as Excel**: no automation at all, just a spreadsheet.

## If you want a real published Zapier/Make app later

That's a business step, not a code one: create a developer account with
Zapier and/or Make, define the trigger/action there using this same
payload shape as the contract, and go through their app review process.
Nothing about the webhook payload above needs to change to support that;
it's the same data, just packaged as an installable integration
instead of a URL someone pastes in.
