> **DRAFT, NOT LEGAL ADVICE.** Read [`legal/README.md`](README.md) before
> using this anywhere. Placeholders are marked `[LIKE THIS]`. The
> liability and warranty sections in particular are legal judgment calls
> that need a real attorney, not engineering defaults; what's here is
> deliberately conservative rather than filled in with a guess.

# Terms of Service (Draft)

**Last updated:** [DATE]

These Terms govern your use of Nuru (the "Service"), provided by [COMPANY
NAME] ("we," "us"). By using the Service, you agree to these Terms.

## 1. What the Service does

Nuru reads PDF invoices, receipts, and financial statements you upload,
extracts fields like vendor, date, and total amount, and, after you
review and approve them, sends that data to a destination you choose
(a webhook, an email address, or a file archive location) or lets you
download it as a spreadsheet.

## 2. Your responsibilities

- You're responsible for having the right to upload and process any
  document you submit to the Service.
- You're responsible for reviewing extracted data before approving it.
  The Service surfaces a confidence indicator and flags fields it isn't
  sure about, but **you** are the one who confirms it's correct before
  it's sent anywhere.
- You're responsible for the accuracy of any destination (email address,
  webhook URL, file path) you configure. Data sent to a destination you
  specified is sent as instructed.

## 3. Accuracy disclaimer

Extraction is performed by an automated system and **is not guaranteed to
be accurate**. [THIS NEEDS A REAL DECISION: what accuracy do you actually
commit to, if any? No benchmark against real-world documents exists yet;
see the readiness audit. Do not publish an accuracy claim you haven't
measured.] The Service is a review-and-approve tool, not a substitute for
your own verification of financial data before you rely on it.

## 4. Data handling

See the [Privacy Policy](DRAFT_PRIVACY_POLICY.md) for what we collect, how
long we keep it, and who we share it with. In short: source documents are
deleted once you complete an action for them (or automatically after a
bounded period if a review is abandoned); we don't retain the extracted
financial data itself beyond that.

## 5. Acceptable use

You agree not to use the Service to:
- Upload documents you don't have the right to process.
- Attempt to disrupt, overload, or gain unauthorized access to the
  Service or its infrastructure.
- Use the webhook or email automation features to send unsolicited
  communications or to target systems you don't have permission to
  interact with.

## 6. Availability

[NO UPTIME COMMITMENT HAS BEEN DEFINED. Decide whether you're offering an
SLA. If this is meant for enterprise customers, they will likely ask for
one, and that's a business + engineering-operations commitment, not just
legal text.]

## 7. Limitation of liability

[THIS IS THE SECTION THAT MOST NEEDS AN ATTORNEY. A tool that
mis-extracts a financial figure and sends it somewhere real carries real
liability exposure. Do not publish this section without counsel defining
the actual limitation-of-liability, indemnification, and disclaimer-of-
warranties language, and confirming what insurance coverage backs it;
see the readiness audit's Business & Legal section.]

## 8. Termination

[DEFINE: under what conditions can either party end the relationship, and
what happens to a customer's data on termination? Should align with the
retention commitments in the Privacy Policy.]

## 9. Governing law

[NOT YET SPECIFIED. A business decision, typically wherever the company
is incorporated.]

## 10. Contact

Questions about these Terms: [CONTACT EMAIL].
