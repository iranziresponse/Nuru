# About the documents in this folder

**These are drafts, not legal advice, and not something to publish as-is.**
I'm not a lawyer, and neither document here has been reviewed by one. What
they *are*: an accurate description of what Nuru's code actually does today,
what it collects, how long it keeps it, who it's shared with, what
security controls exist, written in the shape of a Privacy Policy and
Terms of Service so a real attorney has something concrete to start from
instead of generic boilerplate that doesn't match the product.

Before either document is published anywhere a real user will see it:

1. **Have a licensed attorney review and revise both**, especially the
   liability, indemnification, and dispute-resolution language in the ToS,
   since those are legal judgment calls, not engineering ones, and the draft
   language here is deliberately conservative/incomplete rather than
   guessed at.
2. **Confirm the business model assumption.** Both drafts assume Nuru is
   offered as a hosted service to customers (matching the original "PRD"
   framing: enterprise accounting teams, LinkedIn-facing). If it's
   actually self-hosted software a business runs internally for its own
   staff, both documents need real changes to who "you," "we," and "the
   Service" refer to.
3. **Confirm jurisdiction and applicable law.** Neither document names a
   governing law or jurisdiction; that's a business decision, not
   something to default.
4. **If any user is in the EU/UK or California**, GDPR/UK GDPR and
   CCPA/CPRA have specific requirements (a lawful basis for processing,
   data subject rights, a formal DPA with any subprocessor) beyond what a
   general privacy policy covers. Say so explicitly to whoever reviews
   this so they scope for it.
5. **Get a signed Data Processing Agreement (DPA)** with any customer that
   asks for one before handling their data. This is standard for
   B2B/enterprise vendors and will come up as soon as a serious prospect
   evaluates Nuru.

See also [`docs/soc2-readiness.md`](../docs/soc2-readiness.md) for what a
SOC 2 process would actually require beyond these two documents.
