# SOC 2 readiness: a gap analysis, not a certification plan

This is a reference for what a SOC 2 report would actually require,
mapped against what Nuru currently has. It's the thing enterprise buyers
handling financial data will ask for most often — treat this as prep
work for that conversation, not a substitute for engaging a real auditor.

**What SOC 2 actually is, briefly:** an independent auditor's report
against the AICPA's Trust Services Criteria. Every SOC 2 report covers
**Security** (the "Common Criteria," mandatory); **Availability**,
**Processing Integrity**, **Confidentiality**, and **Privacy** are
optional add-on categories you choose based on what your customers care
about. A **Type I** report attests controls are designed correctly at a
point in time; a **Type II** — the one most enterprise buyers actually
want — attests they operated effectively over a period, typically 3–12
months. You cannot get a Type II faster than the observation period you
choose; that's the real timeline driver, not the audit itself.

## Common Criteria (Security) — mandatory for any SOC 2 report

| Criteria | What it covers | Status |
|---|---|---|
| CC1 — Control Environment | Organizational structure, board oversight, hiring/background-check policies, code of conduct | **Missing.** Organizational, not code — needs a real company structure to exist first. |
| CC2 — Communication & Information | Documented policies communicated to staff; incident communication plan | **Missing.** No written information security policy exists yet. |
| CC3 — Risk Assessment | A documented process for identifying and assessing risk (this readiness audit is informal groundwork for that, not a substitute for it) | **Partial.** This audit and the fixes that followed are real risk-identification work; it isn't a formal, repeatable risk assessment process. |
| CC4 — Monitoring Activities | Ongoing monitoring that controls are working (log review, vulnerability scanning cadence) | **Missing.** The audit trail (below) is a building block; nothing reviews it on a schedule yet. |
| CC5 — Control Activities | Policies translating risk assessment into concrete controls | **Partial.** The security fixes in this session *are* control activities; they aren't yet backed by the CC1–CC4 process that's supposed to justify and track them. |
| CC6 — Logical & Physical Access | Access control, authentication, encryption, network security | **Partial** — see detail below. |
| CC7 — System Operations | Vulnerability management, incident detection/response, backup/recovery | **Partial** — see detail below. |
| CC8 — Change Management | Controlled process for deploying changes (code review, testing, approval) | **Partial.** A real test suite exists (60 tests) and every change this session was verified before shipping; there's no formal review/approval gate since this is currently a one-person project. |
| CC9 — Risk Mitigation | Vendor/subprocessor risk management, business continuity planning | **Missing.** No vendor risk process; no documented business continuity or disaster recovery plan. |

### CC6 (Access) in detail

| Control | Status |
|---|---|
| Authentication | Have (optional): `NURU_ACCESS_PASSWORD` gates the app via HTTP Basic Auth. **Gap:** single shared credential, no per-user identity, no MFA. |
| Authorization / least privilege | Missing: no role concept — anyone with the password can do everything. |
| Encryption in transit | Missing by default: plain HTTP. A reverse proxy can add TLS, but that's not configured out of the box. |
| Encryption at rest | Missing: cached documents and the audit log are stored as plain files. |
| Audit logging | Have: `audit.py` records who (IP), what, when, and outcome for every scan and automation. **Gap:** metadata only (by design, for data minimization — see below), no log integrity protection (append-only file, not tamper-evident), no automated review. |

### CC7 (Operations) in detail

| Control | Status |
|---|---|
| Vulnerability management | Partial: dependencies aren't on an automated scanning/patching cadence. |
| Incident detection | Missing: no error monitoring/alerting — failures currently go to a terminal's stdout. |
| Backup / recovery | Partial: the trained model and vocabulary are version-controlled; the pending-review state and audit log are not backed up. |
| Rate limiting / abuse prevention | Have: Flask-Limiter caps requests per route. |
| Input validation / injection defense | Have: upload size cap, SSRF guard on webhook destinations, archive-path allowlist, CSRF protection. |

## Optional categories

- **Availability** — no uptime commitment or infrastructure redundancy exists yet (single local process). Only pursue this category once there's a real hosting/on-call story.
- **Processing Integrity** — directly relevant to Nuru's core value prop ("did it extract the data correctly"), but requires a *measured* accuracy benchmark against real documents, which doesn't exist yet (see the readiness audit's accuracy section).
- **Confidentiality** — the ephemeral-deletion design (source files purged after automation, now with a TTL backstop for abandoned reviews) is a genuine strength here; encryption at rest is the missing piece.
- **Privacy** — needs the actual Privacy Policy finalized (see `legal/`) and a defined data subject rights process before this category is realistic.

## The realistic sequence

1. Decide on scope (Security only, or Security + one or two add-ons) — Confidentiality is the most natural second category given what's already built.
2. Write the actual policies CC1/CC2 require (a written InfoSec policy is usually the first thing an auditor asks for).
3. Close the CC6/CC7 gaps above that are pure engineering (encryption at rest, error monitoring, backup).
4. Engage an auditor for a **Type I** first — it validates control *design* without waiting through an observation period, and surfaces gaps before committing to Type II.
5. Run the **Type II** observation period (3–12 months) with controls actually operating, not just designed.

None of this is a code deliverable I can hand you finished. What's in
this repo today — the audit trail, the security hardening, the test
suite — are real, verifiable inputs to CC5/CC6/CC7/CC8; the
organizational layer (CC1–CC4, CC9) has to be built by the business, not
generated.
