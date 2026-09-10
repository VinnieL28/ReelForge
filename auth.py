"""
Authentication, roles and per-user sandboxing.

Two deliberate decisions worth stating up front.

**Hashing.** A single unsalted SHA-256 of a password is not a password hash --
a consumer GPU works through billions of SHA-256 guesses per second, so a
stolen users.json is a stolen password list. What is stored here is
PBKDF2-HMAC-SHA256: the same SHA-256 primitive, salted per user and stretched
over 240,000 iterations, which is the standard way to store a password with
SHA-256 and is what `hashlib` is built for. Plain SHA-256 digests are still
accepted on login so an existing file keeps working, and they are transparently
re-hashed to PBKDF2 the first time that user signs in.

**Where users live.** `users.json` beside the app, or environment variables
when that file is absent -- a container should not have to bake credentials
into its image or mount a file to have a working admin account.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from functools import lru_cache
from typing import Any

from paths import EXPORTS_ROOT, USERS_FILE, ensure_dir

# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

ADMIN = "admin"
CREATOR = "creator"

ROLES: dict[str, dict[str, Any]] = {
    ADMIN: {
        "label": "Admin",
        "blurb": "Every engine, the API configuration and user management.",
        "modes": ("commentary", "minimalist", "narrative", "batch", "reel",
                  "duel", "admin"),
        "manage_users": True,
        "manage_keys": True,
    },
    CREATOR: {
        "label": "Creator",
        "blurb": "The three generation studios. Own files only.",
        "modes": ("commentary", "minimalist", "narrative"),
        "manage_users": False,
        "manage_keys": False,
    },
}

# "member" is accepted as a synonym so either word works in a config file.
ROLE_ALIASES = {"member": CREATOR, "user": CREATOR, "creator": CREATOR, "admin": ADMIN}

DEFAULT_ROLE = CREATOR


def normalise_role(role: Any) -> str:
    return ROLE_ALIASES.get(str(role or "").strip().lower(), DEFAULT_ROLE)


def allowed_modes(role: str) -> tuple[str, ...]:
    return tuple(ROLES[normalise_role(role)]["modes"])


def can(role: str, capability: str) -> bool:
    """Capability check -- `can(role, "manage_users")`."""
    return bool(ROLES[normalise_role(role)].get(capability, False))


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

PBKDF2_ITERATIONS = 240_000
_HASH_PREFIX = "pbkdf2_sha256"


def hash_password(password: str, *, iterations: int = PBKDF2_ITERATIONS,
                  salt: bytes | None = None) -> str:
    """Returns `pbkdf2_sha256$<iterations>$<salt_b64>$<hash_b64>`."""
    if not password:
        raise ValueError("Password must not be empty.")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "$".join((
        _HASH_PREFIX, str(iterations),
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(digest).decode("ascii"),
    ))


def sha256_hex(password: str) -> str:
    """A bare SHA-256 digest, for reading legacy entries and for tests."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(stored: str, password: str) -> bool:
    """
    Constant-time check against either hash format.

    `hmac.compare_digest` rather than `==`: string comparison returns early on
    the first differing byte, which leaks how much of a guess was right.
    """
    stored = (stored or "").strip()
    if not stored or not password:
        return False

    if stored.startswith(_HASH_PREFIX + "$"):
        try:
            _, raw_iterations, raw_salt, raw_digest = stored.split("$", 3)
            iterations = int(raw_iterations)
            salt = base64.b64decode(raw_salt)
            expected = base64.b64decode(raw_digest)
        except (ValueError, TypeError):
            return False
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(candidate, expected)

    # Legacy: "sha256$<hex>" or a bare 64-character hex digest.
    legacy = stored.split("$", 1)[1] if stored.startswith("sha256$") else stored
    if re.fullmatch(r"[0-9a-fA-F]{64}", legacy):
        return hmac.compare_digest(legacy.lower(), sha256_hex(password))
    return False


def needs_upgrade(stored: str) -> bool:
    """True when a stored hash is legacy and should be re-written on login."""
    return bool(stored) and not str(stored).startswith(_HASH_PREFIX + "$")


# ---------------------------------------------------------------------------
# The user store
# ---------------------------------------------------------------------------

USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,30}$")


def normalise_username(username: str) -> str:
    return str(username or "").strip().lower()


def valid_username(username: str) -> bool:
    return bool(USERNAME_RE.fullmatch(normalise_username(username)))


def _read_json(path: Any) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def users_from_env() -> dict[str, dict[str, Any]]:
    """
    Users declared in the environment, for deployments with no writable file.

      REELFORGE_ADMIN_USER / REELFORGE_ADMIN_PASSWORD_HASH  (or _PASSWORD)
      REELFORGE_USERS = {"ana": {"password_hash": "...", "role": "creator"}}

    A hash is strongly preferred over a plaintext password: anything in an env
    var shows up in `docker inspect` and in the process listing.
    """
    users: dict[str, dict[str, Any]] = {}

    blob = os.environ.get("REELFORGE_USERS", "").strip()
    if blob:
        try:
            parsed = json.loads(blob)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            for name, record in parsed.items():
                if not isinstance(record, dict) or not valid_username(name):
                    continue
                stored = str(record.get("password_hash") or "")
                if not stored and record.get("password"):
                    stored = hash_password(str(record["password"]))
                if stored:
                    users[normalise_username(name)] = {
                        "password_hash": stored,
                        "role": normalise_role(record.get("role")),
                        "created_at": str(record.get("created_at") or ""),
                        "source": "env",
                    }

    admin = normalise_username(os.environ.get("REELFORGE_ADMIN_USER", ""))
    if admin and valid_username(admin):
        stored = os.environ.get("REELFORGE_ADMIN_PASSWORD_HASH", "").strip()
        if not stored and os.environ.get("REELFORGE_ADMIN_PASSWORD"):
            stored = hash_password(os.environ["REELFORGE_ADMIN_PASSWORD"])
        if stored:
            users[admin] = {"password_hash": stored, "role": ADMIN,
                            "created_at": "", "source": "env"}

    return users


def load_users(path: Any = None) -> dict[str, dict[str, Any]]:
    """
    Every known user: the file, plus anything declared in the environment.

    Environment entries win. That is what makes a locked-out admin
    recoverable -- set the env var, restart, sign in.
    """
    path = path or USERS_FILE
    stored = _read_json(path)
    raw = stored.get("users") if isinstance(stored.get("users"), dict) else stored

    users: dict[str, dict[str, Any]] = {}
    if isinstance(raw, dict):
        for name, record in raw.items():
            if not isinstance(record, dict) or not valid_username(name):
                continue
            users[normalise_username(name)] = {
                "password_hash": str(record.get("password_hash") or ""),
                "role": normalise_role(record.get("role")),
                "created_at": str(record.get("created_at") or ""),
                "last_login": str(record.get("last_login") or ""),
                "source": "file",
            }

    users.update(users_from_env())
    return users


def save_users(users: dict[str, dict[str, Any]], path: Any = None) -> None:
    """
    Writes the file-backed users, atomically, with the env-only ones dropped.

    Atomic because a half-written users.json locks everyone out: the replace is
    the only moment the real file changes.
    """
    path = path or USERS_FILE
    keep = {
        name: {
            "password_hash": record["password_hash"],
            "role": normalise_role(record.get("role")),
            "created_at": str(record.get("created_at") or ""),
            "last_login": str(record.get("last_login") or ""),
        }
        for name, record in sorted(users.items())
        if record.get("source") != "env" and record.get("password_hash")
    }

    temp = f"{path}.tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump({"users": keep}, handle, indent=2, ensure_ascii=False)
    os.replace(temp, path)
    try:
        os.chmod(path, 0o600)       # no-op on Windows, meaningful on Linux
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap(path: Any = None) -> dict[str, Any]:
    """
    Guarantees at least one admin exists.

    Returns {"created": bool, "username": str, "password": str}. The generated
    password is returned once, for the caller to print to the server console --
    never to the browser, where anyone who can reach the login page would read
    it.
    """
    path = path or USERS_FILE
    users = load_users(path)
    if any(normalise_role(record.get("role")) == ADMIN for record in users.values()):
        return {"created": False, "username": "", "password": ""}

    username = normalise_username(os.environ.get("REELFORGE_ADMIN_USER", "") or "admin")
    if not valid_username(username):
        username = "admin"
    password = os.environ.get("REELFORGE_ADMIN_PASSWORD", "") or secrets.token_urlsafe(12)

    users[username] = {
        "password_hash": hash_password(password),
        "role": ADMIN,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_login": "",
        "source": "file",
    }
    save_users(users, path)
    return {"created": True, "username": username, "password": password}


# ---------------------------------------------------------------------------
# Sign-in
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _decoy_hash() -> str:
    """
    One throwaway hash, derived once.

    Deriving it per call would make the unknown-user path do two PBKDF2 rounds
    against a real login's one, which is a 2x timing gap pointing straight at
    "no such account" -- the opposite of what the decoy is for.
    """
    return hash_password(secrets.token_urlsafe(16))


def authenticate(username: str, password: str, path: Any = None) -> dict[str, Any] | None:
    """
    Checks a sign-in and returns {"username", "role"} or None.

    A missing user still runs a full PBKDF2 verification against a dummy hash,
    so "no such user" and "wrong password" take the same time and the login
    form cannot be used to enumerate accounts.
    """
    path = path or USERS_FILE
    name = normalise_username(username)
    users = load_users(path)
    record = users.get(name)

    if record is None:
        verify_password(_decoy_hash(), str(password))
        return None

    if not verify_password(str(record.get("password_hash") or ""), str(password)):
        return None

    if record.get("source") != "env":
        record = dict(record)
        if needs_upgrade(str(record.get("password_hash"))):
            record["password_hash"] = hash_password(str(password))
        record["last_login"] = time.strftime("%Y-%m-%d %H:%M:%S")
        users[name] = {**record, "source": "file"}
        try:
            save_users(users, path)
        except OSError:
            pass        # a read-only deployment still signs in fine

    return {"username": name, "role": normalise_role(record.get("role"))}


# ---------------------------------------------------------------------------
# Login rate limiting
#
# This lives here rather than in app.py for a mechanical reason: Streamlit
# re-executes the main script top to bottom on every interaction, so a
# module-level counter defined there is reset by the very click it is meant to
# count. An imported module is executed once and cached in sys.modules, so its
# state survives reruns -- and survives across browser sessions, which is what
# makes this a control rather than a hint.
#
# It is still per-process and per-username. It does nothing about an attempt
# spread across many usernames or many replicas; that belongs at the reverse
# proxy.
# ---------------------------------------------------------------------------

LOGIN_MAX_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 30.0

_LOGIN_FAILURES: dict[str, tuple[int, float]] = {}
_LOGIN_LOCK = threading.Lock()


def login_locked_for(username: str) -> float:
    """Seconds remaining before this account may try again."""
    with _LOGIN_LOCK:
        _, until = _LOGIN_FAILURES.get(normalise_username(username), (0, 0.0))
    return max(0.0, until - time.time())


def note_login_failure(username: str) -> float:
    """Records a failed attempt; returns the cooldown it triggered, or 0."""
    name = normalise_username(username)
    with _LOGIN_LOCK:
        count, until = _LOGIN_FAILURES.get(name, (0, 0.0))
        count += 1
        if count >= LOGIN_MAX_ATTEMPTS:
            until = time.time() + LOGIN_LOCKOUT_SECONDS
            count = 0
        _LOGIN_FAILURES[name] = (count, until)
    return max(0.0, until - time.time())


def clear_login_failures(username: str = "") -> None:
    """Forgets failures for one account, or all of them when given nothing."""
    with _LOGIN_LOCK:
        if username:
            _LOGIN_FAILURES.pop(normalise_username(username), None)
        else:
            _LOGIN_FAILURES.clear()


# ---------------------------------------------------------------------------
# User management (admin)
# ---------------------------------------------------------------------------

def add_user(username: str, password: str, role: str = DEFAULT_ROLE,
             path: Any = None) -> dict[str, Any]:
    path = path or USERS_FILE
    name = normalise_username(username)
    if not valid_username(name):
        raise ValueError(
            "Usernames are 2-31 characters, lowercase, starting with a letter or "
            "digit; letters, digits, dot, dash and underscore after that."
        )
    if len(str(password)) < 8:
        raise ValueError("Passwords must be at least 8 characters.")

    users = load_users(path)
    if name in users:
        raise ValueError(f"'{name}' already exists.")

    users[name] = {
        "password_hash": hash_password(password),
        "role": normalise_role(role),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_login": "",
        "source": "file",
    }
    save_users(users, path)
    ensure_dir(user_exports_dir(name))
    return {"username": name, "role": normalise_role(role)}


def set_password(username: str, password: str, path: Any = None) -> None:
    path = path or USERS_FILE
    name = normalise_username(username)
    users = load_users(path)
    if name not in users:
        raise ValueError(f"No such user: {name}")
    if users[name].get("source") == "env":
        raise ValueError(f"'{name}' comes from the environment; change the env var instead.")
    if len(str(password)) < 8:
        raise ValueError("Passwords must be at least 8 characters.")
    users[name]["password_hash"] = hash_password(password)
    save_users(users, path)


def set_role(username: str, role: str, path: Any = None) -> None:
    path = path or USERS_FILE
    name = normalise_username(username)
    users = load_users(path)
    if name not in users:
        raise ValueError(f"No such user: {name}")
    if users[name].get("source") == "env":
        raise ValueError(f"'{name}' comes from the environment; change the env var instead.")

    wanted = normalise_role(role)
    if wanted != ADMIN and _admin_count(users) == 1 and \
            normalise_role(users[name].get("role")) == ADMIN:
        raise ValueError("This is the only admin. Promote someone else first.")

    users[name]["role"] = wanted
    save_users(users, path)


def delete_user(username: str, path: Any = None) -> None:
    path = path or USERS_FILE
    name = normalise_username(username)
    users = load_users(path)
    if name not in users:
        raise ValueError(f"No such user: {name}")
    if users[name].get("source") == "env":
        raise ValueError(f"'{name}' comes from the environment; remove the env var instead.")
    if normalise_role(users[name].get("role")) == ADMIN and _admin_count(users) == 1:
        raise ValueError("This is the only admin. Promote someone else first.")

    del users[name]
    save_users(users, path)
    # Their exports are deliberately left on disk. Deleting an account should
    # not silently destroy the work; an operator can remove the folder.


def _admin_count(users: dict[str, dict[str, Any]]) -> int:
    return sum(1 for record in users.values() if normalise_role(record.get("role")) == ADMIN)


# ---------------------------------------------------------------------------
# Per-user sandbox
# ---------------------------------------------------------------------------

_UNSAFE = re.compile(r"[^a-z0-9._-]+")


def safe_slug(username: str) -> str:
    """
    A username reduced to something that cannot escape the exports folder.

    `valid_username` already rejects separators, but this runs anyway: the
    directory name is the entire sandbox boundary, and it should not depend on
    validation somewhere else having been called first.
    """
    slug = _UNSAFE.sub("-", normalise_username(username)).strip("-.") or "unassigned"
    return slug[:40]


def user_exports_dir(username: str) -> str:
    """`exports/<username>/` -- created on demand."""
    return str(EXPORTS_ROOT / safe_slug(username))


def list_user_dirs() -> list[str]:
    """Every user folder that exists on disk, for the admin overview."""
    if not EXPORTS_ROOT.is_dir():
        return []
    return sorted(entry.name for entry in EXPORTS_ROOT.iterdir() if entry.is_dir())
