# dropbox_utils.py
# -*- coding: utf-8 -*-
"""Dropbox helpers for Streamlit (Access Token based).
Includes upload/download for RAG cache files.
ASCII-safe, Python 3.8+ compatible.
"""
from __future__ import annotations

import io
import os
import time
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st

try:
    import dropbox
    from dropbox.files import WriteMode, FileMetadata
except Exception as e:
    st.error("Dropbox SDK is required. Please add 'dropbox>=11.36.0' to requirements.txt")
    raise

# --------------------
# Authentication
# --------------------
@st.cache_resource(show_spinner=False)
def authenticate_dropbox() -> "dropbox.Dropbox":
    """Authenticate using an access token from Streamlit secrets.
    Expect secrets.DROPBOX_ACCESS_TOKEN (string).
    """
    token = st.secrets.get("DROPBOX_ACCESS_TOKEN", None)
    if not token:
        raise RuntimeError("DROPBOX_ACCESS_TOKEN is missing in secrets.")
    dbx = dropbox.Dropbox(token, timeout=120)
    # Quick check: get current account
    try:
        _ = dbx.users_get_current_account()
    except Exception as e:
        raise RuntimeError(f"Dropbox auth failed: {e}")
    return dbx

# --------------------
# Helpers
# --------------------

def _retry(callable_fn, max_tries: int = 3, base_delay: float = 0.8):
    last_err = None
    for i in range(max_tries):
        try:
            return callable_fn()
        except Exception as e:
            last_err = e
            time.sleep(base_delay * (2 ** i))
    if last_err:
        raise last_err


def format_file_size(size_val: Optional[int]) -> str:
    if size_val is None:
        return "-"
    try:
        size = int(size_val)
    except Exception:
        return str(size_val)
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    val = float(size)
    while val >= 1024.0 and i < len(units) - 1:
        val = val / 1024.0
        i += 1
    if i == 0:
        return f"{int(val)} {units[i]}"
    return f"{val:.1f} {units[i]}"


def _normalize_app_path(folder_path: Optional[str], app_scope: str = "app_folder") -> str:
    """Normalize a provided folder path for Dropbox API.
    - For app_scope='app_folder': root must be "" (empty string). Subfolders are like "/sub".
    - For app_scope='full_dropbox': absolute path starting with "/" is required.
    This function also strips accidental URLs and trims whitespace.
    """
    p = (folder_path or "").strip()

    # If user pasted a URL by mistake, reject clearly
    if p.startswith("http://") or p.startswith("https://"):
        raise ValueError("DROPBOX_FOLDER_PATH must be a path (e.g., \"/subfolder\"), not a URL.")

    if app_scope == "app_folder":
        # Root of the app folder must be ""
        if p in ("", "/", None):
            return ""
        # If someone passed "/Apps/<AppName>/X", reduce to "/X"
        if p.startswith("/Apps/"):
            parts = p.split("/", 3)  # ['', 'Apps', '<AppName>', 'maybe/rest']
            if len(parts) >= 4 and parts[3]:
                p = "/" + parts[3]
            else:
                # exactly the app root -> normalize to ""
                return ""
        if not p.startswith("/"):
            p = "/" + p
        return p
    else:
        # Full Dropbox scope
        if p in ("", None):
            return "/"
        return p if p.startswith("/") else "/" + p


def _ensure_folder(dbx: "dropbox.Dropbox", folder_path: str, app_scope: str = "app_folder"):
    """Create folder if it doesn't exist (idempotent)."""
    norm = _normalize_app_path(folder_path, app_scope=app_scope)
    # Root cannot/need not be created
    if norm == "":
        return
    try:
        dbx.files_get_metadata(norm)
    except dropbox.exceptions.ApiError:
        dbx.files_create_folder_v2(norm, autorename=False)


# --------------------
# Self-test & diagnostics
# --------------------

def self_test(dbx: "dropbox.Dropbox") -> Dict[str, Any]:
    """Return quick diagnostics to help identify scope/path issues."""
    info: Dict[str, Any] = {}
    try:
        acc = dbx.users_get_current_account()
        info["account_id"] = getattr(acc, "account_id", None)
        info["name"] = getattr(getattr(acc, "name", None), "display_name", None)
    except Exception as e:
        info["account_error"] = str(e)

    # Try listing app root ("" for App Folder)
    try:
        root_list = dbx.files_list_folder("", recursive=False)
        info["root_ok"] = True
        info["root_count"] = len(root_list.entries)
    except Exception as e:
        info["root_ok"] = False
        info["root_error"] = str(e)
    return info


# --------------------
# File listing & I/O
# --------------------

def list_files_in_folder(dbx: "dropbox.Dropbox", folder_path: str, app_scope: str = "app_folder") -> List[Dict[str, Any]]:
    """List files in a folder (non-recursive). Return Drive-like dicts."""
    api_path = _normalize_app_path(folder_path, app_scope=app_scope)

    res = _retry(lambda: dbx.files_list_folder(api_path, recursive=False))
    entries = list(res.entries)
    while res.has_more:
        res = _retry(lambda: dbx.files_list_folder_continue(res.cursor))
        entries.extend(res.entries)

    out: List[Dict[str, Any]] = []
    for e in entries:
        if isinstance(e, FileMetadata):
            out.append({
                "id": e.id,
                "name": e.name,
                "size": int(getattr(e, "size", 0) or 0),
                "mimeType": None,
                "modifiedTime": e.server_modified.isoformat() if getattr(e, "server_modified", None) else "",
                "path_lower": getattr(e, "path_lower", "") or "",
                "path_display": getattr(e, "path_display", "") or "",
            })
    return out


def download_file(dbx: "dropbox.Dropbox", path_or_id: str, app_scope: str = "app_folder") -> io.BytesIO:
    """Download a file by path (preferred)."""
    # Assume path_or_id is a path from list_files_in_folder
    api_path = _normalize_app_path(path_or_id, app_scope=app_scope)
    # Special case: if api_path == "" it's invalid for files_download; must be an actual file path
    if api_path == "":
        raise ValueError("download_file() requires a file path, not the root folder.")
    md, resp = _retry(lambda: dbx.files_download(api_path))
    data = resp.content
    bio = io.BytesIO(data)
    bio.seek(0)
    return bio


def upload_file(dbx: "dropbox.Dropbox", folder_path: str, local_path: str, app_scope: str = "app_folder") -> str:
    """Upload or overwrite a file to a folder. Returns the uploaded path."""
    norm_folder = _normalize_app_path(folder_path, app_scope=app_scope)
    _ensure_folder(dbx, norm_folder, app_scope=app_scope)

    filename = os.path.basename(local_path)
    if norm_folder == "":
        dest_path = f"/{filename}"
    else:
        dest_path = f"{norm_folder.rstrip('/')}/{filename}"

    with open(local_path, "rb") as f:
        data = f.read()
    _retry(lambda: dbx.files_upload(data, dest_path, mode=WriteMode("overwrite")))
    return dest_path


# --------------------
# RAG cache helpers (embeddings + FAISS)
# --------------------

def _find_file_by_name(dbx: "dropbox.Dropbox", folder_path: str, filename: str, app_scope: str = "app_folder") -> Optional[str]:
    files = list_files_in_folder(dbx, folder_path, app_scope=app_scope)
    for f in files:
        if f.get("name") == filename:
            return f.get("path_display") or f.get("path_lower")
    return None


def download_embeddings_from_dropbox(
    dbx: "dropbox.Dropbox",
    folder_path: str,
    embeddings_name: str = "embeddings_meta.pkl",
    faiss_name: str = "faiss_index.bin",
    app_scope: str = "app_folder",
) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"embeddings_path": None, "faiss_path": None}

    emb_path = _find_file_by_name(dbx, folder_path, embeddings_name, app_scope=app_scope)
    if emb_path:
        buf = download_file(dbx, emb_path, app_scope=app_scope)
        with open(embeddings_name, "wb") as f:
            f.write(buf.getvalue())
        out["embeddings_path"] = os.path.abspath(embeddings_name)

    f_path = _find_file_by_name(dbx, folder_path, faiss_name, app_scope=app_scope)
    if f_path:
        buf = download_file(dbx, f_path, app_scope=app_scope)
        with open(faiss_name, "wb") as f:
            f.write(buf.getvalue())
        out["faiss_path"] = os.path.abspath(faiss_name)

    return out


def upload_embeddings_to_dropbox(
    dbx: "dropbox.Dropbox",
    folder_path: str,
    embeddings_path: str = "embeddings_meta.pkl",
    faiss_path: str = "faiss_index.bin",
    app_scope: str = "app_folder",
) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"embeddings_uploaded": None, "faiss_uploaded": None}
    if os.path.exists(embeddings_path):
        out["embeddings_uploaded"] = upload_file(dbx, folder_path, embeddings_path, app_scope=app_scope)
    if os.path.exists(faiss_path):
        out["faiss_uploaded"] = upload_file(dbx, folder_path, faiss_path, app_scope=app_scope)
    return out
