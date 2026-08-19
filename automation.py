"""The three "Driven Automation" actions Nuru can take once a user has
reviewed and approved a document's extracted fields: send it to an
external webhook, email it, or archive the source PDF under a clean name.

Every function returns (ok: bool, message: str) instead of raising, so the
caller can show the result inline rather than crashing the review flow.
Standard library only, no new dependencies.
"""

import json
import os
import re
import shutil
import smtplib
import urllib.error
import urllib.request
from email.message import EmailMessage

WEBHOOK_TIMEOUT_SECONDS = 10

_LOCAL_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nuru.local.env")


def _load_local_env_file():
    """Read KEY=VALUE settings (SMTP credentials, etc.) from a local file
    that never gets committed to git, so they persist across restarts
    without needing to be set in the shell every time. Real environment
    variables, if already set, always win over the file."""
    if not os.path.exists(_LOCAL_ENV_PATH):
        return
    with open(_LOCAL_ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_local_env_file()

_FALLBACK_ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "archive")
DEFAULT_ARCHIVE_DIR = os.environ.get("NURU_ARCHIVE_DIR") or _FALLBACK_ARCHIVE_DIR


def default_webhook_url():
    """A pre-filled webhook URL from NURU_WEBHOOK_URL, if one is set, so
    the review screen doesn't need it retyped for every document. Still
    overridable per document in the UI."""
    return os.environ.get("NURU_WEBHOOK_URL", "")


def default_email_to():
    """A pre-filled destination address from NURU_DEFAULT_EMAIL_TO, mirroring
    the webhook/archive defaults, e.g. a standing accounts-payable inbox."""
    return os.environ.get("NURU_DEFAULT_EMAIL_TO", "")


def send_webhook(url, payload):
    """POST the approved fields as JSON to a user-provided destination URL
    (an accounting platform endpoint, or a Zapier/Make "Catch Webhook"
    trigger URL)."""
    if not url or not url.startswith(("http://", "https://")):
        return False, "That doesn't look like a valid URL. It should start with http:// or https://."

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=WEBHOOK_TIMEOUT_SECONDS) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        return False, f"The destination rejected the request (status {exc.code})."
    except urllib.error.URLError:
        return False, "Couldn't reach that address. Please check the URL and try again."
    except Exception:
        return False, "Something went wrong sending this. Please check the URL and try again."

    if 200 <= status < 300:
        return True, "Sent."
    return False, f"The destination responded with status {status}."


def _smtp_config():
    host = os.environ.get("NURU_SMTP_HOST")
    user = os.environ.get("NURU_SMTP_USER")
    password = os.environ.get("NURU_SMTP_PASSWORD")
    if not (host and user and password):
        return None
    return {
        "host": host,
        "port": int(os.environ.get("NURU_SMTP_PORT", "587")),
        "user": user,
        "password": password,
        "from_addr": os.environ.get("NURU_SMTP_FROM", user),
    }


def email_is_configured():
    return _smtp_config() is not None


def send_email(to_address, subject, rows):
    """Format the approved fields as a readable summary and send it via
    the SMTP account configured through NURU_SMTP_* environment variables.
    `rows` is the same list of {label, value} dicts the review grid holds."""
    if not to_address or "@" not in to_address:
        return False, "That doesn't look like a valid email address."

    config = _smtp_config()
    if config is None:
        return False, (
            "Email sending isn't set up yet. Set NURU_SMTP_HOST, NURU_SMTP_USER, "
            "and NURU_SMTP_PASSWORD as environment variables (and restart Nuru), "
            "then try again."
        )

    lines = [f"{row['label']}: {row['value'] or 'Not provided'}" for row in rows]
    body = "\n".join(lines)

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = config["from_addr"]
    message["To"] = to_address
    message.set_content(body)

    try:
        with smtplib.SMTP(config["host"], config["port"], timeout=WEBHOOK_TIMEOUT_SECONDS) as server:
            server.starttls()
            server.login(config["user"], config["password"])
            server.send_message(message)
    except smtplib.SMTPAuthenticationError:
        return False, "The email account rejected those credentials."
    except Exception as exc:
        print(f"  error sending email: {exc}")
        return False, "Something went wrong sending this email. Please try again."

    return True, f"Sent to {to_address}."


_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def _clean_filename_part(text, fallback):
    text = (text or "").strip()
    if not text:
        return fallback
    return _UNSAFE_FILENAME_CHARS.sub("_", text).strip("_") or fallback


def build_archive_filename(rows, original_filename):
    """SupplierName_Date.pdf, built from whatever the user approved. Falls
    back to the original filename's pieces when a field is missing."""
    name_field = next((r for r in rows if r["label"] in
                        ("Supplier Name", "Merchant", "Account Name")), None)
    date_field = next((r for r in rows if r["label"] in
                        ("Transaction Date", "Purchase Date", "Statement Period")), None)

    name_part = _clean_filename_part(name_field["value"] if name_field else "", "Document")
    date_part = _clean_filename_part(date_field["value"] if date_field else "", "")

    ext = os.path.splitext(original_filename)[1] or ".pdf"
    if date_part:
        return f"{name_part}_{date_part}{ext}"
    return f"{name_part}{ext}"


def archive_file(pdf_path, dest_dir, filename):
    """Move the source PDF into a designated storage directory under a
    clean corporate-standard name, creating the directory if needed."""
    dest_dir = dest_dir or DEFAULT_ARCHIVE_DIR
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except OSError as exc:
        print(f"  error creating archive folder {dest_dir}: {exc}")
        return False, "Couldn't create that destination folder. Please check the path and try again."

    dest_path = os.path.join(dest_dir, filename)
    if os.path.exists(dest_path):
        stem, ext = os.path.splitext(filename)
        i = 2
        while os.path.exists(os.path.join(dest_dir, f"{stem} ({i}){ext}")):
            i += 1
        dest_path = os.path.join(dest_dir, f"{stem} ({i}){ext}")

    try:
        shutil.move(pdf_path, dest_path)
    except OSError as exc:
        print(f"  error archiving {pdf_path}: {exc}")
        return False, "Couldn't move the file to that folder. Please try again."

    return True, f"Archived as {os.path.basename(dest_path)}."
