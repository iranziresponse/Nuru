> **DRAFT, NOT LEGAL ADVICE.** Read [`legal/README.md`](README.md) before
> using this anywhere. Placeholders are marked `[LIKE THIS]`.

# Privacy Policy (Draft)

**Last updated:** [DATE]

This policy describes how [COMPANY NAME] ("we," "us") handles information
when you use Nuru (the "Service") to extract data from invoices, receipts,
and financial statements.

## 1. What we collect

- **The documents you upload**: PDF invoices, receipts, or financial
  statements, and the text/data our system extracts from them (vendor or
  merchant name, dates, amounts, account identifiers, and any fields you
  add or edit during review).
- **Account and usage information**, if applicable: [LOGIN/ACCOUNT DETAILS
  IF THIS BECOMES A REAL SAAS PRODUCT WITH ACCOUNTS; NOT YET IMPLEMENTED].
- **Audit metadata**: for each document, the time it was scanned, the
  document type our system inferred, which automation action was taken
  (sent to a webhook, emailed, archived, or downloaded), the destination
  you specified for that action (e.g. an email address or webhook URL),
  whether it succeeded, and the IP address the request came from. This
  metadata **does not include the extracted field values themselves**;
  see Section 3.

We do not run your documents through any third-party AI or machine
learning API. Extraction is performed by a system we built and run
ourselves.

## 2. How long we keep it

- **The uploaded document itself** is deleted permanently the moment you
  complete an action for it (send, email, archive, download, or discard).
  If a document is scanned but never carried through to completion, it is
  automatically deleted after [24 HOURS BY DEFAULT, CONFIGURABLE]
  regardless.
- **Audit metadata** (Section 1, third bullet) is retained for
  [RETENTION PERIOD NOT YET DEFINED; DECIDE AND FILL IN; NO AUTOMATIC
  ROTATION IS CURRENTLY IMPLEMENTED] to support internal controls and
  troubleshooting. It does not include the content of your documents.

## 3. What we don't keep

We do not retain the extracted field values (vendor names, amounts, dates,
etc.) once your document has been processed and its action completed. Our
audit trail records *that* an action happened and *whether it succeeded*,
not the financial data itself.

## 4. Who we share it with

Only the destinations **you** specify when you choose an automation
action:

- A **webhook** URL you provide (for example, a Zapier or Make.com
  endpoint you control).
- An **email address** you provide, sent via our configured mail
  provider.
- A **file storage location** you (or your administrator) has approved.

We do not sell your data or share it with any other third party. [ADD ANY
SUBPROCESSORS HERE ONCE THE HOSTING/EMAIL/INFRASTRUCTURE PROVIDERS ARE
FINALIZED, e.g. cloud hosting provider, transactional email provider,
each with its own DPA.]

## 5. Security

We apply the following technical safeguards: destination validation on
outbound webhook requests (to prevent the service being used to probe
internal network addresses), an allowlist for file-archive destinations,
CSRF protection and rate limiting on all data-changing requests, standard
security response headers, and a cap on upload size. [DESCRIBE ENCRYPTION
AT REST, HOSTING PROVIDER SECURITY CERTIFICATIONS, ETC. ONCE THOSE ARE IN
PLACE; NOT YET IMPLEMENTED.]

No system is perfectly secure, and we can't guarantee absolute security.

## 6. Your rights

[FILL IN BASED ON APPLICABLE LAW AND WHERE YOUR USERS ARE LOCATED, e.g.
GDPR/UK GDPR data subject rights (access, correction, deletion,
portability, objection) if you have EU/UK users; CCPA/CPRA rights if you
have California users. This section cannot be completed without knowing
who your users are and which laws apply; see legal/README.md.]

## 7. Contact

Questions about this policy: [CONTACT EMAIL].

## 8. Changes to this policy

[STANDARD "WE MAY UPDATE THIS POLICY" LANGUAGE; HAVE COUNSEL DRAFT THE
EXACT NOTICE MECHANISM YOU'LL ACTUALLY FOLLOW.]
