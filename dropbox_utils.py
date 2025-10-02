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
from typing import Dict, Any, List, Optional

import streamlit as st

try:
    import dropbox
    from dropbox.files import WriteMode
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


def _ensure_folder(dbx: "dropbox.Dropbox", folder_path: str):
    """Create folder if it doesn't exist (idempotent)."""
    if not folder_path or folder_path == "/":
        return
    if not folder_path.startswith("/"):
        folder_path = "/" + folder_path
    try:
        dbx.files_get_metadata(folder_path)
    except dropbox.exceptions.ApiError:
        # Try create
        dbx.files_create_folder_v2(folder_path, autorename=False)


# --------------------
# File listing & I/O
# --------------------

def list_files_in_folder(dbx: "dropbox.Dropbox", folder_path: str) -> List[Dict[str, Any]]:
    """List files in a Dropbox folder (non-recursive). Return Drive-like dicts."""
    if not folder_path.startswith("/"):
        folder_path = "/" + folder_path

    res = _retry(lambda: dbx.files_list_folder(folder_path, recursive=False))
    entries = list(res.entries)
    while res.has_more:
        res = _retry(lambda: dbx.files_list_folder_continue(res.cursor))
        entries.extend(res.entries)

    out: List[Dict[str, Any]] = []
    for e in entries:
        if isinstance(e, dropbox.files.FileMetadata):
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


def download_file(dbx: "dropbox.Dropbox", path_or_id: str) -> io.BytesIO:
    """Download a file by path (preferred)."""
    path = path_or_id
    if not path.startswith("/"):
        path = "/" + path
    md, resp = _retry(lambda: dbx.files_download(path))
    data = resp.content
    bio = io.BytesIO(data)
    bio.seek(0)
    return bio


def upload_file(dbx: "dropbox.Dropbox", folder_path: str, local_path: str) -> str:
    """Upload or overwrite a file to a folder. Returns the uploaded path."""
    if not folder_path.startswith("/"):
        folder_path = "/" + folder_path
    _ensure_folder(dbx, folder_path)

    filename = os.path.basename(local_path)
    dest_path = f"{folder_path.rstrip('/')}/{filename}"

    with open(local_path, "rb") as f:
        data = f.read()
    _retry(lambda: dbx.files_upload(data, dest_path, mode=WriteMode("overwrite")))
    return dest_path


# --------------------
# RAG cache helpers (embeddings + FAISS)
# --------------------

def _find_file_by_name(dbx: "dropbox.Dropbox", folder_path: str, filename: str) -> Optional[str]:
    files = list_files_in_folder(dbx, folder_path)
    for f in files:
        if f.get("name") == filename:
            return f.get("path_display") or f.get("path_lower")
    return None


def download_embeddings_from_dropbox(
    dbx: "dropbox.Dropbox",
    folder_path: str,
    embeddings_name: str = "embeddings_meta.pkl",
    faiss_name: str = "faiss_index.bin",
) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"embeddings_path": None, "faiss_path": None}

    emb_path = _find_file_by_name(dbx, folder_path, embeddings_name)
    if emb_path:
        buf = download_file(dbx, emb_path)
        with open(embeddings_name, "wb") as f:
            f.write(buf.getvalue())
        out["embeddings_path"] = os.path.abspath(embeddings_name)

    f_path = _find_file_by_name(dbx, folder_path, faiss_name)
    if f_path:
        buf = download_file(dbx, f_path)
        with open(faiss_name, "wb") as f:
            f.write(buf.getvalue())
        out["faiss_path"] = os.path.abspath(faiss_name)

    return out


def upload_embeddings_to_dropbox(
    dbx: "dropbox.Dropbox",
    folder_path: str,
    embeddings_path: str = "embeddings_meta.pkl",
    faiss_path: str = "faiss_index.bin",
) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {"embeddings_uploaded": None, "faiss_uploaded": None}
    if os.path.exists(embeddings_path):
        out["embeddings_uploaded"] = upload_file(dbx, folder_path, embeddings_path)
    if os.path.exists(faiss_path):
        out["faiss_uploaded"] = upload_file(dbx, folder_path, faiss_path)
    return out
