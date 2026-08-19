"""The three "Driven Automation" actions Nuru can take once a user has
reviewed and approved a document's extracted fields: send it to an
external webhook, email it, or archive the source PDF under a clean name.

Every function returns (ok: bool, message: str) instead of raising, so the
caller can show the result inline rather than crashing the review flow.
Standard library only, no new dependencies.
"""

import ipaddress
import json
import os
import re
import shutil
import smtplib
import socket
import urllib.error
import urllib.request
from email.message import EmailMessage
from urllib.parse import urlparse

import errors

_logger = errors.get_logger()

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


# Always blocked, regardless of NURU_WEBHOOK_ALLOW_PRIVATE: no webhook has a
# legitimate reason to target link-local space (this is where cloud provider
# metadata endpoints like 169.254.169.254 live), multicast, or reserved
# ranges. Loopback and private (RFC1918) ranges are allowed by default since
# a self-hosted receiver on the same machine or LAN is a normal setup for a
# single-user tool; set NURU_WEBHOOK_PUBLIC_ONLY=true to also block those,
# which you want the moment more than one trusted person can trigger a send.
def _ip_is_always_blocked(ip):
    return ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified


def _ip_is_blocked_unless_public_only_is_off(ip):
    return ip.is_loopback or ip.is_private


def _webhook_host_is_safe(hostname):
    """Resolve the hostname and check every address it could mean. Blocks
    the common cases (a literal internal IP, a hostname that resolves to
    one). Does not defend a sophisticated DNS-rebinding attack, where the
    name resolves safely here but differently at actual connect time —
    that needs a transport that connects to a pinned, pre-validated IP,
    which this stdlib implementation doesn't do."""
    public_only = os.environ.get("NURU_WEBHOOK_PUBLIC_ONLY", "").lower() in ("1", "true", "yes")
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if _ip_is_always_blocked(ip):
            return False
        if public_only and _ip_is_blocked_unless_public_only_is_off(ip):
            return False
    return True


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A URL that passes the safety check can still redirect to one that
    wouldn't. Refusing to follow redirects at all closes that off — no
    real webhook receiver needs Nuru to follow one."""

    def redirect_request(self, *args, **kwargs):
        return None


_NO_REDIRECT_OPENER = urllib.request.build_opener(_NoRedirect)


def send_webhook(url, payload):
    """POST the approved fields as JSON to a user-provided destination URL
    (an accounting platform endpoint, or a Zapier/Make "Catch Webhook"
    trigger URL)."""
    if not url or not url.startswith(("http://", "https://")):
        return False, "That doesn't look like a valid URL. It should start with http:// or https://."

    hostname = urlparse(url).hostname
    if not hostname or not _webhook_host_is_safe(hostname):
        return False, "That destination isn't allowed."

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with _NO_REDIRECT_OPENER.open(request, timeout=WEBHOOK_TIMEOUT_SECONDS) as response:
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
        errors.report_exception(exc, action="send_email", to=to_address)
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


def allowed_archive_dirs():
    """The archive destinations Nuru is permitted to write to: the default
    folder, plus anything an administrator has explicitly added via
    NURU_ARCHIVE_ALLOWED_ROOTS (os.pathsep-separated). A free-text path
    typed by whoever's using the review screen is never enough on its own
    — otherwise "designated secure storage directory" means nothing and
    the app will write a file anywhere it has filesystem permission to."""
    roots = [DEFAULT_ARCHIVE_DIR]
    extra = os.environ.get("NURU_ARCHIVE_ALLOWED_ROOTS", "")
    roots += [r.strip() for r in extra.split(os.pathsep) if r.strip()]
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for root in roots:
        resolved = os.path.realpath(root)
        if resolved not in seen:
            seen.add(resolved)
            unique.append(root)
    return unique


def _is_allowed_archive_dir(path):
    resolved = os.path.realpath(path)
    for root in allowed_archive_dirs():
        root_resolved = os.path.realpath(root)
        if resolved == root_resolved or resolved.startswith(root_resolved + os.sep):
            return True
    return False


def archive_file(pdf_path, dest_dir, filename):
    """Move the source PDF into a designated storage directory under a
    clean corporate-standard name, creating the directory if needed."""
    dest_dir = dest_dir or DEFAULT_ARCHIVE_DIR
    if not _is_allowed_archive_dir(dest_dir):
        return False, "That folder isn't one of the allowed archive locations."
    try:
        os.makedirs(dest_dir, exist_ok=True)
    except OSError as exc:
        errors.report_exception(exc, action="archive_mkdir", dest_dir=dest_dir)
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
        errors.report_exception(exc, action="archive_move", pdf_path=pdf_path, dest_dir=dest_dir)
        return False, "Couldn't move the file to that folder. Please try again."

    return True, f"Archived as {os.path.basename(dest_path)}."
