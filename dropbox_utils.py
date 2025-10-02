# dropbox_utils.py
path = "/" + path


md, resp = _retry(lambda: dbx.files_download(path))
data = resp.content
bio = io.BytesIO(data)
bio.seek(0)
return bio




def upload_file(dbx: dropbox.Dropbox, folder_path: str, local_path: str) -> str:
"""Upload or overwrite a file to a folder. Returns the uploaded path."""
if not folder_path.startswith("/"):
folder_path = "/" + folder_path
_ensure_folder(dbx, folder_path)


filename = os.path.basename(local_path)
dest_path = f"{folder_path.rstrip('/')}/{filename}"


with open(local_path, "rb") as f:
_retry(lambda: dbx.files_upload(f.read(), dest_path, mode=WriteMode("overwrite")))
return dest_path




# --------------------
# RAG cache helpers (embeddings + FAISS)
# --------------------


def _find_file_by_name(dbx: dropbox.Dropbox, folder_path: str, filename: str) -> Optional[str]:
"""Return a display path for the first match, else None."""
files = list_files_in_folder(dbx, folder_path)
for f in files:
if f.get("name") == filename:
return f.get("path_display") or f.get("path_lower")
return None




def download_embeddings_from_dropbox(dbx: dropbox.Dropbox,
folder_path: str,
embeddings_name: str = "embeddings_meta.pkl",
faiss_name: str = "faiss_index.bin") -> Dict[str, Optional[str]]:
out = {"embeddings_path": None, "faiss_path": None}


emb_path = _find_file_by_name(dbx, folder_path, embeddings_name)
if emb_path:
buf = download_file(dbx, emb_path)
with open(embeddings_name, "wb") as f:
f.write(buf.getvalue())
out["embeddings_path"] = os.path.abspath(embeddings_name)


faiss_path = _find_file_by_name(dbx, folder_path, faiss_name)
if faiss_path:
buf = download_file(dbx, faiss_path)
with open(faiss_name, "wb") as f:
f.write(buf.getvalue())
out["faiss_path"] = os.path.abspath(faiss_name)


return out




def upload_embeddings_to_dropbox(dbx: dropbox.Dropbox,
folder_path: str,
embeddings_path: str = "embeddings_meta.pkl",
faiss_path: str = "faiss_index.bin") -> Dict[str, Optional[str]]:
out = {"embeddings_uploaded": None, "faiss_uploaded": None}
if os.path.exists(embeddings_path):
out["embeddings_uploaded"] = upload_file(dbx, folder_path, embeddings_path)
if os.path.exists(faiss_path):
out["faiss_uploaded"] = upload_file(dbx, folder_path, faiss_path)
return out
