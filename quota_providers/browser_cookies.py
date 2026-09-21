"""Local browser-cookie helpers for the Grok quota provider.

Firefox stores cookies in cookies.sqlite. Chrome stores them encrypted in
Cookies SQLite. We copy each database first because the live browser may keep
the file locked. Only grok.com domains are selected and cookie values never
leave this process.
"""
from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, NoReturn, Optional

try:
    import cryptography  # noqa: F401  # dependency probe

    _HAS_CRYPTO = True
except ImportError:  # pragma: no cover - environment-dependent
    _HAS_CRYPTO = False

logger = logging.getLogger(__name__)

# Typed failures that must abort the whole Chrome import instead of
# degrading to the next profile database.
_HARD_CHROME_FAILURES = frozenset({"chrome-crypto-missing"})


class ChromeCookieError(ValueError):
    """Typed Chrome import failure.

    Subclasses ``ValueError`` so the decrypt primitive can raise typed
    failures directly (no string re-parsing at the call site), while any
    unexpected ``ValueError`` still degrades to ``chrome-decrypt-failed``.
    """

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _raise_chrome_tcc() -> NoReturn:
    """Single spelling of the typed TCC-denied failure."""
    raise ChromeCookieError("chrome-tcc-denied") from None


def _is_tcc_error(exc: BaseException) -> bool:
    if isinstance(exc, PermissionError):
        return True
    if isinstance(exc, OSError) and getattr(exc, "errno", None) == 1:
        return True
    return "not permitted" in str(exc).lower()


# Chrome expires_utc is microseconds since 1601-01-01 UTC.
_CHROME_EPOCH_DELTA_S = 11_644_473_600
_CHROME_KEY_SALT = b"saltysalt"
_CHROME_KEY_ITERS = 1003
# Linux browsers derive the Safe Storage key with a single PBKDF2 round.
_CHROME_KEY_ITERS_LINUX = 1
_CHROME_CBC_IV = b" " * 16


@contextmanager
def _copied_sqlite(source: Path) -> Iterator[sqlite3.Connection]:
    fd, temp_name = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        shutil.copyfile(source, temp_name)
        conn = sqlite3.connect(temp_name)
        try:
            yield conn
        finally:
            conn.close()
    finally:
        try:
            os.unlink(temp_name)
        except OSError:
            pass


def _cookie_header(pairs: Iterator[tuple[str, str]]) -> Optional[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name, value in pairs:
        if not name or not value or name in seen:
            continue
        seen.add(name)
        out.append(f"{name}={value}")
    return "; ".join(out) if out else None


def _firefox_cookie_dbs() -> list[Path]:
    roots: list[Path] = []
    # Windows: %APPDATA%\Mozilla\Firefox\Profiles
    appdata = os.environ.get("APPDATA")
    if appdata:
        roots.append(Path(appdata) / "Mozilla" / "Firefox" / "Profiles")
    # macOS: ~/Library/Application Support/Firefox/Profiles
    roots.append(Path.home() / "Library" / "Application Support" / "Firefox" / "Profiles")
    # Linux: ~/.mozilla/firefox
    roots.append(Path.home() / ".mozilla" / "firefox")
    dbs: list[Path] = []
    for root in roots:
        dbs.extend(Path(p) for p in glob.glob(str(root / "*" / "cookies.sqlite")))
    return dbs


def load_firefox_grok_cookies() -> Optional[str]:
    now = int(time.time())
    for source in _firefox_cookie_dbs():
        try:
            with _copied_sqlite(source) as conn:
                rows = conn.execute(
                    """
                    SELECT name, value
                    FROM moz_cookies
                    WHERE (host = 'grok.com' OR host LIKE '%.grok.com')
                      AND (expiry = 0 OR expiry > ?)
                    ORDER BY host, path, name
                    """,
                    (now,),
                ).fetchall()
        except (OSError, sqlite3.Error):
            continue
        header = _cookie_header((name, value) for name, value in rows)
        if header:
            return header
    return None


def _linux_config_home() -> Path:
    """Config dir where Linux Chromium/Chrome/Brave keep their profiles."""
    configured = os.environ.get("XDG_CONFIG_HOME")
    return Path(configured) if configured else Path.home() / ".config"


def _chrome_user_data_roots() -> list[Path]:
    """Existing browser user-data roots on this platform.

    macOS reads the Keychain and Linux the login keyring (secret-tool), so both
    are supported; Windows is not (DPAPI key).
    """
    if sys.platform == "darwin":
        candidates = [Path.home() / "Library" / "Application Support" / "Google" / "Chrome"]
    elif sys.platform.startswith("linux"):
        config = _linux_config_home()
        candidates = [
            config / "chromium",
            config / "google-chrome",
            config / "google-chrome-beta",
            config / "google-chrome-unstable",
            config / "BraveSoftware" / "Brave-Browser",
            config / "BraveSoftware" / "Brave-Browser-Beta",
            config / "BraveSoftware" / "Brave-Browser-Dev",
        ]
    else:
        return []
    roots: list[Path] = []
    for path in candidates:
        try:
            if path.exists():
                roots.append(path)
        except OSError as exc:
            if _is_tcc_error(exc):
                _raise_chrome_tcc()
    return roots


def _chrome_last_used_profile(root: Path) -> Optional[str]:
    local_state = root / "Local State"
    try:
        data = json.loads(local_state.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    profile = data.get("profile") if isinstance(data, dict) else None
    if not isinstance(profile, dict):
        return None
    last = profile.get("last_used") or profile.get("last_used_profile_directory")
    return last if isinstance(last, str) and last else None


def _safe_chrome_profile_dir(root: Path, name: str) -> Optional[Path]:
    """Join a Local State / listing name only if it stays inside ``root``."""
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        return None
    candidate = root / name
    try:
        candidate.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    try:
        return candidate if candidate.is_dir() else None
    except OSError as exc:
        if _is_tcc_error(exc):
            _raise_chrome_tcc()
        return None


def chrome_profile_dirs(root: Path) -> list[Path]:
    names: list[str] = []
    last = _chrome_last_used_profile(root)
    if last:
        names.append(last)
    names.append("Default")
    saw_tcc = False
    try:
        extra = sorted(
            p.name
            for p in root.iterdir()
            if p.is_dir() and (p.name == "Default" or p.name.startswith("Profile "))
        )
    except OSError as exc:
        extra = []
        if _is_tcc_error(exc):
            saw_tcc = True
    ordered: list[Path] = []
    seen = set()
    for name in names + extra:
        if name in seen:
            continue
        seen.add(name)
        try:
            path = _safe_chrome_profile_dir(root, name)
        except ChromeCookieError:
            raise
        except OSError as exc:
            if _is_tcc_error(exc):
                saw_tcc = True
            continue
        if path is not None:
            ordered.append(path)
    if not ordered and saw_tcc:
        _raise_chrome_tcc()
    return ordered


def chrome_cookie_dbs() -> list[Path]:
    dbs: list[Path] = []
    seen: set[Path] = set()
    saw_tcc = False
    for root in _chrome_user_data_roots():
        try:
            profiles = chrome_profile_dirs(root)
        except OSError as exc:
            if _is_tcc_error(exc):
                saw_tcc = True
            continue
        for profile in profiles:
            for candidate in (
                profile / "Network" / "Cookies",
                profile / "Cookies",
            ):
                try:
                    if candidate.is_file() and candidate not in seen:
                        seen.add(candidate)
                        dbs.append(candidate)
                except OSError as exc:
                    if _is_tcc_error(exc):
                        saw_tcc = True
    if not dbs and saw_tcc:
        _raise_chrome_tcc()
    return dbs


# Linux keeps the "Safe Storage" password in the login keyring; these are the
# lookups the browsers themselves use (newest xdg:schema form last).
_LINUX_SECRET_LOOKUPS: dict[str, tuple[tuple[str, str], ...]] = {
    "chrome": (
        ("application", "chrome"),
        ("xdg:schema", "chrome_libsecret_os_crypt_password_v2"),
    ),
    "chromium": (
        ("application", "chromium"),
        ("xdg:schema", "chromium_libsecret_os_crypt_password_v2"),
    ),
    "brave": (
        ("application", "brave"),
        ("xdg:schema", "brave_libsecret_os_crypt_password_v2"),
    ),
}
# Chromium started without a keyring (--password-store=basic) uses this
# well-known password; a wrong candidate simply fails to decrypt.
_LINUX_FALLBACK_PASSWORD = "peanuts"


def _linux_keyring_password(app: str) -> Optional[str]:
    if not shutil.which("secret-tool"):
        return None
    for attribute, value in _LINUX_SECRET_LOOKUPS.get(app, ()):
        try:
            completed = subprocess.run(
                ["secret-tool", "lookup", attribute, value],
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if completed.returncode == 0:
            password = (completed.stdout or "").strip()
            if password:
                return password
    return None


def _chrome_safe_storage_password(app: str = "chrome") -> Optional[str]:
    """Safe Storage password for one browser, or None when unavailable."""
    if sys.platform.startswith("linux"):
        return _linux_keyring_password(app)
    if sys.platform != "darwin":
        return None
    if app != "chrome":
        # Only Google Chrome's Keychain item is read on macOS.
        return None
    try:
        completed = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-w",
                "-s",
                "Chrome Safe Storage",
                "-a",
                "Chrome",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    password = (completed.stdout or "").strip()
    return password or None


def _chrome_key_iterations() -> int:
    """PBKDF2 rounds for the Safe Storage password.

    macOS Chrome derives with 1003 rounds; Linux Chromium/Chrome/Brave use one
    round (verified live: 1003 fails on Linux v11 blobs, 1 decrypts them).
    """
    if sys.platform.startswith("linux"):
        return _CHROME_KEY_ITERS_LINUX
    return _CHROME_KEY_ITERS


def _derive_chrome_key(password: str) -> bytes:
    if not _HAS_CRYPTO:
        raise ChromeCookieError("chrome-crypto-missing")
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    return PBKDF2HMAC(
        algorithm=hashes.SHA1(),
        length=16,
        salt=_CHROME_KEY_SALT,
        iterations=_chrome_key_iterations(),
    ).derive(password.encode("utf-8"))


def _chrome_aes_keys(app: str = "chrome") -> list[bytes]:
    """Candidate AES keys for one browser's cookie DB, best guess first."""
    keys: list[bytes] = []
    password = _chrome_safe_storage_password(app)
    if password:
        keys.append(_derive_chrome_key(password))
    if sys.platform.startswith("linux"):
        keys.append(_derive_chrome_key(_LINUX_FALLBACK_PASSWORD))
    if not keys:
        raise ChromeCookieError("chrome-keychain-denied")
    return keys


def _chrome_aes_key() -> bytes:
    """Single-key form (macOS Keychain path)."""
    return _chrome_aes_keys("chrome")[0]


def _strip_chrome_host_hash(plaintext: bytes, host_key: Optional[str] = None) -> bytes:
    """Chrome 127+ / cookie DB v24+ prefixes SHA256(host_key) before the value."""
    if len(plaintext) < 32:
        return plaintext
    prefix, rest = plaintext[:32], plaintext[32:]
    if host_key is not None:
        expected = hashlib.sha256(host_key.encode("utf-8")).digest()
        if prefix == expected:
            return rest
    try:
        plaintext.decode("utf-8")
        return plaintext
    except UnicodeDecodeError:
        try:
            rest.decode("utf-8")
            return rest
        except UnicodeDecodeError:
            return plaintext


def decrypt_chrome_cookie_value(
    key: bytes, blob: bytes, host_key: Optional[str] = None
) -> str:
    """Decrypt one Chrome cookie value.

    Raises ``ChromeCookieError`` with a kebab-case reason for every typed
    failure (``chrome-app-bound``, ``chrome-unknown-prefix``,
    ``chrome-decrypt-failed``, ``chrome-crypto-missing``).
    """
    if not _HAS_CRYPTO:
        raise ChromeCookieError("chrome-crypto-missing")
    if blob.startswith(b"v20"):
        raise ChromeCookieError("chrome-app-bound")
    if not blob.startswith((b"v10", b"v11")):
        raise ChromeCookieError("chrome-unknown-prefix")
    ciphertext = blob[3:]
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    decryptor = Cipher(algorithms.AES(key), modes.CBC(_CHROME_CBC_IV)).decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()
    if not padded:
        raise ChromeCookieError("chrome-decrypt-failed")
    pad = padded[-1]
    if pad < 1 or pad > 16 or padded[-pad:] != bytes([pad] * pad):
        raise ChromeCookieError("chrome-decrypt-failed")
    plaintext = _strip_chrome_host_hash(padded[:-pad], host_key)
    try:
        return plaintext.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ChromeCookieError("chrome-decrypt-failed") from exc


def _chrome_expires_ok(expires_utc: object, now: float) -> bool:
    if expires_utc in (None, 0):
        return True
    try:
        expiry = int(str(expires_utc))
    except (TypeError, ValueError):
        return False
    if expiry <= 0:
        return True
    unix = (expiry / 1_000_000) - _CHROME_EPOCH_DELTA_S
    return unix > now


def _chrome_expiry_cutoff(now: float) -> int:
    return int((now + _CHROME_EPOCH_DELTA_S) * 1_000_000)


def _chrome_app_for_path(path: Path) -> str:
    """Which browser's Safe Storage secret encrypts this cookie DB."""
    lowered = str(path).lower()
    if "brave" in lowered:
        return "brave"
    if "chromium" in lowered:
        return "chromium"
    return "chrome"


def load_chrome_grok_cookies() -> Optional[str]:
    """Cookie header for grok.com from a local browser profile, or None.

    macOS reads the Keychain, Linux the login keyring (secret-tool); Windows is
    unsupported (DPAPI).
    """
    if sys.platform != "darwin" and not sys.platform.startswith("linux"):
        return None
    try:
        sources = chrome_cookie_dbs()
    except ChromeCookieError:
        raise
    except OSError as exc:
        if _is_tcc_error(exc):
            _raise_chrome_tcc()
        return None

    now = time.time()
    cutoff = _chrome_expiry_cutoff(now)
    keys_by_app: dict[str, list[bytes]] = {}
    decrypt_reason: Optional[str] = None

    for source in sources:
        try:
            with _copied_sqlite(source) as conn:
                rows = conn.execute(
                    """
                    SELECT name, host_key, encrypted_value, expires_utc
                    FROM cookies
                    WHERE (host_key = 'grok.com' OR host_key LIKE '%.grok.com')
                      AND (expires_utc IS NULL OR expires_utc <= 0 OR expires_utc > ?)
                    ORDER BY host_key, name
                    """,
                    (cutoff,),
                ).fetchall()
        except OSError as exc:
            if _is_tcc_error(exc):
                _raise_chrome_tcc()
            continue
        except sqlite3.Error:
            logger.debug(
                "browser_cookies ▸ unreadable Chrome cookie DB (skip): %s",
                source,
                exc_info=True,
            )
            continue

        pairs: list[tuple[str, str]] = []
        for name, host_key, blob, expires_utc in rows:
            if not name or not _chrome_expires_ok(expires_utc, now):
                continue
            if not isinstance(blob, (bytes, bytearray)):
                continue
            app = _chrome_app_for_path(source)
            keys = keys_by_app.get(app)
            if keys is None:
                # Resolved lazily: a profile with no grok cookies must not fail
                # just because no keyring secret is available.
                keys = _chrome_aes_keys(app)
                keys_by_app[app] = keys
            value: Optional[str] = None
            for candidate in keys:
                try:
                    value = decrypt_chrome_cookie_value(
                        candidate, bytes(blob), host_key=str(host_key or "")
                    )
                except ChromeCookieError as exc:
                    if exc.reason in _HARD_CHROME_FAILURES:
                        raise
                    # Soft failure: record the typed reason and try the next
                    # candidate key / source database instead of aborting.
                    decrypt_reason = exc.reason
                    value = None
                    continue
                except Exception as exc:
                    raise ChromeCookieError("chrome-decrypt-failed") from exc
                if value:
                    break
            if value:
                pairs.append((str(name), value))
        header = _cookie_header(iter(pairs))
        if header:
            return header

    if decrypt_reason:
        raise ChromeCookieError(decrypt_reason)
    return None
