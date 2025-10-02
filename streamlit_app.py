# -*- coding: utf-8 -*-
import os
import pickle
from io import BytesIO
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import streamlit as st
import numpy as np
import pandas as pd


from datetime import datetime, timezone

def _to_epoch(mt: str) -> float:
    try:
        return datetime.fromisoformat(mt.replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0

# FAISS
try:
    import faiss
except Exception:
    st.error("FAISS is required. Please add 'faiss-cpu>=1.7.4' to requirements.txt")
    st.stop()

# OpenAI
try:
    from openai import OpenAI
except Exception:
    st.error("OpenAI SDK is required. Please add 'openai>=1.12.0' to requirements.txt")
    st.stop()

# Bcrypt (custom auth)
try:
    import bcrypt
except Exception:
    st.error("Please add 'bcrypt>=4.0.1' to requirements.txt")
    st.stop()

# Local modules
try:
    from drive_utils import (
        authenticate_drive,
        list_files_in_folder,
        download_file,
        format_file_size,
        download_embeddings_from_drive,
    )
except Exception as e:
    st.error("Failed to import drive_utils: %s" % e)
    st.stop()

try:
    from document_processors import (
        process_pdf,
        process_pptx,
        chunk_text_smart,
        get_embeddings,
        count_tokens,
    )
except Exception as e:
    st.error("Failed to import document_processors: %s" % e)
    st.stop()

# =========================
# App Constants & Settings
# =========================
EMBEDDINGS_FILE = "embeddings_meta.pkl"
FAISS_INDEX_FILE = "faiss_index.bin"
TOP_K = 10  # Tăng lên để có nhiều candidates cho reranking

st.set_page_config(page_title="VNA Tech", layout="wide")

# =========================
# Authentication (giữ nguyên)
# =========================
def _load_credentials_from_secrets() -> Dict[str, Dict[str, str]]:
    if "auth" not in st.secrets:
        raise RuntimeError("Missing [auth] in secrets.")
    users = st.secrets["auth"].get("users", {})
    creds = {}
    for _, u in users.items():
        uname = u.get("username")
        pwd = u.get("password")
        name = u.get("name", uname)
        if uname and pwd:
            creds[uname] = {"name": name, "password": pwd}
    if not creds:
        raise RuntimeError("No valid users under [auth.users].")
    return creds

def _verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

def login_gate() -> Tuple[bool, str, str]:
    try:
        creds = _load_credentials_from_secrets()
    except Exception as e:
        st.error(str(e))
        st.stop()

    if "auth_user" in st.session_state and st.session_state.get("auth_ok"):
        u = st.session_state["auth_user"]
        display_name = st.session_state.get("auth_name", u)
        return True, u, display_name

    with st.form("login_form", clear_on_submit=False):
        st.subheader("Đăng nhập để truy cập VNA Techinsight Hub")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if username in creds and _verify_password(password, creds[username]["password"]):
            st.session_state["auth_ok"] = True
            st.session_state["auth_user"] = username
            st.session_state["auth_name"] = creds[username]["name"]
            st.success("Đăng nhập thành công.")
            st.rerun()
        else:
            st.error("Sai username hoặc password.")

    return False, "", ""

def logout_button():
    if st.session_state.get("auth_ok"):
        if st.sidebar.button("Sign out"):
            for k in ["auth_ok", "auth_user", "auth_name"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.success("Đã đăng xuất.")
            st.rerun()

# =========================
# Google Drive Helpers
# =========================
@st.cache_resource(show_spinner=False)
def _drive_service():
    return authenticate_drive()

def _list_drive_files() -> List[Dict[str, Any]]:
    folder_id = st.secrets.get("DRIVE_FOLDER_ID")
    if not folder_id:
        st.error("DRIVE_FOLDER_ID is missing in secrets.")
        st.stop()
    service = _drive_service()
    files = list_files_in_folder(service, folder_id)
    filtered = []
    for f in files:
        name = f.get("name", "")
        if name.lower().endswith(".pdf") or name.lower().endswith(".pptx"):
            filtered.append(f)
    return filtered

# =========================
# Embeddings Store & FAISS (giữ nguyên logic cũ)
# =========================
def _try_load_local_index():
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(FAISS_INDEX_FILE):
        try:
            with open(EMBEDDINGS_FILE, "rb") as f:
                meta = pickle.load(f)
            index = faiss.read_index(FAISS_INDEX_FILE)
            return index, meta
        except Exception:
            return None, None
    return None, None

def _load_or_pull_cache_from_drive() -> Tuple[Any, List[Dict[str, Any]]]:
    idx, meta = _try_load_local_index()
    if idx is not None and meta is not None:
        return idx, meta
    service = _drive_service()
    folder_id = st.secrets.get("DRIVE_FOLDER_ID")
    paths = download_embeddings_from_drive(service, folder_id, EMBEDDINGS_FILE, FAISS_INDEX_FILE)
    if paths.get("embeddings_path") and paths.get("faiss_path"):
        try:
            with open(EMBEDDINGS_FILE, "rb") as f:
                meta = pickle.load(f)
            idx = faiss.read_index(FAISS_INDEX_FILE)
            return idx, meta
        except Exception:
            pass
    return None, None

def _get_processed_file_ids(meta: List[Dict[str, Any]]) -> set:
    if not meta:
        return set()
    return {item.get("file_id") for item in meta if item.get("file_id")}

def _build_or_load_index(process_all: bool = False) -> Tuple[Any, List[Dict[str, Any]]]:
    service = _drive_service()
    files = _list_drive_files()
    
    existing_index = None
    existing_meta = []
    processed_ids = set()
    
    if not process_all:
        existing_index, existing_meta = _load_or_pull_cache_from_drive()
        if existing_index is not None and existing_meta is not None:
            processed_ids = _get_processed_file_ids(existing_meta)
            st.info(f"📦 Đã load {len(existing_meta)} chunks từ {len(processed_ids)} files có sẵn")

    new_files = [f for f in files if f["id"] not in processed_ids]
    
    if not new_files and existing_index is not None:
        st.success("✅ Không có file mới. Sử dụng index hiện tại.")
        return existing_index, existing_meta
    
    if new_files:
        st.info(f"📄 Phát hiện {len(new_files)} file mới cần xử lý")
    
    new_vectors = []
    new_meta: List[Dict[str, Any]] = []

    progress = st.progress(0.0, text="Processing new documents...")
    n = max(len(new_files), 1)
    
    for i, f in enumerate(new_files, start=1):
        file_id = f["id"]
        file_name = f["name"]
        file_mtime = f.get("modifiedTime")
        progress.progress(i / n, text="Processing %s (%d/%d)" % (file_name, i, len(new_files)))

        try:
            content: BytesIO = download_file(service, file_id)
        except Exception as e:
            st.warning("Failed to download '%s': %s" % (file_name, e))
            continue

        try:
            if file_name.lower().endswith(".pdf"):
                text, meta = process_pdf(content)
            elif file_name.lower().endswith(".pptx"):
                text, meta = process_pptx(content)
            else:
                continue
        except Exception as e:
            st.warning("Failed to parse '%s': %s" % (file_name, e))
            continue

        chunks = chunk_text_smart(text, meta, chunk_size=1000, chunk_overlap=200)
        texts = [c["text"] for c in chunks]
        
        try:
            vecs = get_embeddings(texts, batch_size=100)
        except Exception as e:
            st.error("Embedding failed for %s: %s" % (file_name, e))
            continue

        for j, c in enumerate(chunks):
            new_vectors.append(vecs[j])
            row = {"file_id": file_id, "file_name": file_name, "modified_time": file_mtime}
            row.update(c)
            new_meta.append(row)
    
    progress.progress(1.0, text="Hoàn thành xử lý file mới")
    
    if not new_vectors and not existing_meta:
        st.error("No embeddings were created. Please check your Drive folder and parsers.")
        st.stop()
    
    if new_vectors:
        new_mat = np.array(new_vectors, dtype="float32")
        faiss.normalize_L2(new_mat)
        
        if existing_index is not None and existing_meta:
            existing_index.add(new_mat)
            combined_meta = existing_meta + new_meta
            st.success(f"✅ Đã thêm {len(new_vectors)} chunks mới vào index (tổng: {len(combined_meta)} chunks)")
            index = existing_index
            all_meta = combined_meta
        else:
            index = faiss.IndexFlatIP(new_mat.shape[1])
            index.add(new_mat)
            all_meta = new_meta
            st.success(f"✅ Đã tạo index mới với {len(new_meta)} chunks")
    else:
        index = existing_index
        all_meta = existing_meta

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(all_meta, f)
    faiss.write_index(index, FAISS_INDEX_FILE)

    return index, all_meta

# =========================
# Enhanced Retrieval & Reranking
# =========================
def _embed_query(client: OpenAI, query: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=[query])
    v = np.array(resp.data[0].embedding, dtype="float32")
    v = v / np.linalg.norm(v)
    return v

def _keyword_score(query: str, text: str, key_terms: List[str]) -> float:
    """Tính keyword matching score"""
    query_lower = query.lower()
    text_lower = text.lower()
    
    score = 0.0
    query_words = set(query_lower.split())
    text_words = set(text_lower.split())
    
    # Exact word matches
    common_words = query_words & text_words
    score += len(common_words) * 0.1
    
    # Key terms matching
    for term in key_terms:
        if term.lower() in query_lower:
            if term.lower() in text_lower:
                score += 0.3
    
    # Phrase matching (bigrams)
    query_bigrams = set(zip(query_lower.split()[:-1], query_lower.split()[1:]))
    text_tokens = text_lower.split()
    text_bigrams = set(zip(text_tokens[:-1], text_tokens[1:]))
    common_bigrams = query_bigrams & text_bigrams
    score += len(common_bigrams) * 0.2
    
    return min(score, 1.0)

def _rerank_results(query: str, results: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
    """Rerank kết quả dựa trên nhiều yếu tố"""
    
    for r in results:
        # Semantic similarity (từ FAISS)
        semantic_score = r["similarity"]
        
        # Keyword matching
        all_terms = r.get("local_key_terms", [])
        keyword_score = _keyword_score(query, r["text"], all_terms)
        
        # Content type bonus
        content_type = r.get("content_type", "general")
        type_bonus = 0.0
        if content_type in ["procedure", "specification"]:
            type_bonus = 0.1
        elif content_type == "safety_note":
            type_bonus = 0.15
        
        # Section completeness bonus
        if r.get("is_complete_section", False):
            type_bonus += 0.05
        
        # Has structure bonus (tables, lists)
        if r.get("has_tables", False):
            type_bonus += 0.05
        if r.get("has_lists", False):
            type_bonus += 0.03
        
        # Combined score với trọng số
        combined_score = (
            semantic_score * 0.65 +
            keyword_score * 0.25 +
            type_bonus * 0.10
        )
        
        r["rerank_score"] = combined_score
        r["keyword_score"] = keyword_score
    
    # Sắp xếp theo rerank_score
    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    
    # Diversify: đảm bảo có chunks từ nhiều files khác nhau
    diverse_results = []
    file_counts = defaultdict(int)
    max_per_file = max(2, top_k // 3)
    
    for r in reranked:
        file_name = r["file_name"]
        if file_counts[file_name] < max_per_file or len(diverse_results) < top_k:
            diverse_results.append(r)
            file_counts[file_name] += 1
            if len(diverse_results) >= top_k:
                break
    
    # Nếu không đủ, lấy thêm
    if len(diverse_results) < top_k:
        for r in reranked:
            if r not in diverse_results:
                diverse_results.append(r)
                if len(diverse_results) >= top_k:
                    break
    
    return diverse_results[:top_k]

def _search(index, meta: List[Dict[str, Any]], qvec: np.ndarray, query: str, topk: int = TOP_K):
    """Enhanced search với reranking"""
    # FAISS search - lấy nhiều candidates hơn
    D, I = index.search(qvec.reshape(1, -1), topk * 2)
    
    candidates = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(meta):
            continue
        item = meta[idx].copy()
        item["similarity"] = float(score)
        candidates.append(item)
    
    # Rerank
    final_results = _rerank_results(query, candidates, top_k=topk)
    
    return final_results

def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format context với metadata phong phú"""
    blocks = []
    
    for i, c in enumerate(chunks, 1):
        # Header với thông tin chi tiết
        file_name = c["file_name"]
        section = "%s %s" % (
            str(c.get("section_type", "?")).title(),
            c.get("section_number", "?")
        )
        
        title = c.get("section_title", "")
        title_str = f" - {title}" if title else ""
        
        content_type = c.get("content_type", "general")
        
        header = f"[{i}] {file_name} | {section}{title_str}\n"
        header += f"Type: {content_type} | Relevance: {c.get('rerank_score', 0):.3f}\n"
        
        # Đánh dấu nếu có tables/lists
        if c.get("has_tables"):
            header += "⚠️ Contains table data\n"
        if c.get("has_lists"):
            header += "📋 Contains structured list\n"
        
        text = c["text"]
        blocks.append(header + "---\n" + text)
    
    return "\n\n═══════════════════\n\n".join(blocks)

def _ask_llm(client: OpenAI, question: str, chunks: List[Dict[str, Any]]) -> str:
    """Enhanced LLM prompting với CoT và structured output"""
    context = _format_context(chunks)
    
    system = """Bạn là trợ lý kỹ thuật chuyên nghiệp của Vietnam Airlines.

NHIỆM VỤ:
1. Đọc kỹ và phân tích tất cả nguồn tham chiếu được cung cấp
2. Trả lời câu hỏi dựa HOÀN TOÀN trên thông tin trong nguồn tham chiếu
3. Nếu thông tin không đủ để trả lời, hãy nói rõ phần nào thiếu
4. Trích dẫn rõ ràng nguồn bằng cách ghi [số] tương ứng với nguồn

CÁCH TRẢ LỜI:
- Viết bằng tiếng Việt, chính xác và chuyên nghiệp
- Cấu trúc câu trả lời rõ ràng (dùng đầu dòng nếu cần)
- Với thông tin kỹ thuật: ghi đầy đủ số liệu, đơn vị, điều kiện
- Với quy trình: liệt kê các bước theo thứ tự
- Luôn trích dẫn nguồn bằng [1], [2], [3]... sau mỗi thông tin

QUAN TRỌNG:
- KHÔNG bịa đặt hoặc thêm thông tin không có trong nguồn
- KHÔNG tóm tắt quá ngắn gọn nếu câu hỏi yêu cầu chi tiết
- Ưu tiên thông tin từ nguồn có "Relevance" cao hơn"""

    user_msg = f"""Câu hỏi: {question}

NGUỒN THAM CHIẾU:
{context}

Hãy trả lời câu hỏi dựa trên các nguồn trên. Nhớ trích dẫn nguồn bằng [số]."""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]
    
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.1,
            max_tokens=2000,
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"LLM API error: {e}")
        return "Xin lỗi, đã có lỗi khi tạo câu trả lời. Vui lòng thử lại."

# =========================
# UI
# =========================
def sidebar_panel(index, meta):
    st.sidebar.header("VNA Techinsight")
    
    processed_ids = _get_processed_file_ids(meta)
    with st.sidebar.expander("📊 Thống kê", expanded=True):
        st.metric("Số files đã xử lý", len(processed_ids))
        st.metric("Tổng số chunks", len(meta) if meta else 0)
        
        # Thống kê content types
        if meta:
            content_types = [m.get("content_type", "general") for m in meta]
            type_counts = pd.Series(content_types).value_counts()
            st.caption("**Content Types:**")
            for ctype, count in type_counts.items():
                st.caption(f"  • {ctype}: {count}")
        
        st.caption("Cache được lưu **local-only** trong phiên chạy.")
    
    st.sidebar.divider()
    
    with st.sidebar.expander("🔧 Quản lý Index", expanded=False):
        st.write("**Embeddings**: `%s`" % EMBEDDINGS_FILE)
        st.write("**FAISS index**: `%s`" % FAISS_INDEX_FILE)
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Cập nhật (chỉ file mới)", use_container_width=True):
                st.session_state["force_rebuild"] = False
                st.rerun()
        with col2:
            if st.button("🔨 Rebuild toàn bộ", type="secondary", use_container_width=True):
                st.session_state["force_rebuild"] = True
                st.rerun()
        
        if st.button("🗑️ Xoá cache (local)", type="secondary", use_container_width=True):
            try:
                if os.path.exists(EMBEDDINGS_FILE):
                    os.remove(EMBEDDINGS_FILE)
                if os.path.exists(FAISS_INDEX_FILE):
                    os.remove(FAISS_INDEX_FILE)
            except Exception:
                pass
            st.success("Đã xoá cache local.")
            st.rerun()

    st.sidebar.divider()
    
    try:
        files = _list_drive_files()
    except Exception as e:
        st.sidebar.error("Lỗi liệt kê Drive: %s" % e)
        files = []

    if files:
        st.sidebar.subheader("📁 Tài liệu trong Drive")
        for f in files[:100]:
            is_new = f["id"] not in processed_ids
            icon = "🆕" if is_new else "✅"
            st.sidebar.caption("%s %s (%s)" % (icon, f["name"], format_file_size(f.get("size", ""))))

    logout_button()

def main():
    ok, username, display_name = login_gate()
    if not ok:
        st.stop()

    DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID", "1JXkaAwVD2lLbFg5bJrNRaSdxJ_oS7kEY")
    drive_url = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}?usp=sharing"

    st.title("🛩️ VNA TechInsight Hub")
    st.caption("Hệ thống tra cứu tài liệu kỹ thuật thông minh với AI")
    st.link_button("📂 Mở thư mục Google Drive để cập nhật kiến thức cho Chatbot", drive_url)

    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY is missing in secrets.")
        st.stop()
    client = OpenAI(api_key=api_key)

    force = st.session_state.get("force_rebuild", False)
    index, meta = _build_or_load_index(process_all=force)
    st.session_state["force_rebuild"] = False

    sidebar_panel(index, meta)

    st.subheader("💬 Đặt câu hỏi")
    
    # Query input với suggestions
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input(
            "Nhập câu hỏi (tiếng Việt hoặc tiếng Anh):",
            value="",
            placeholder="Ví dụ: Quy trình kiểm tra động cơ CFM56 là gì?"
        )
    with col2:
        search_mode = st.selectbox(
            "Chế độ tìm kiếm",
            ["Hybrid (khuyến nghị)", "Semantic only", "Keyword priority"],
            index=0
        )
    
    # Advanced options
    with st.expander("⚙️ Tùy chọn nâng cao"):
        col_a, col_b = st.columns(2)
        with col_a:
            num_results = st.slider("Số nguồn tham chiếu", 5, 20, 10)
        with col_b:
            answer_detail = st.select_slider(
                "Độ chi tiết câu trả lời",
                options=["Ngắn gọn", "Trung bình", "Chi tiết"],
                value="Trung bình"
            )
    
    run = st.button("🔍 Tìm kiếm & Trả lời", type="primary", use_container_width=True)

    if run:
        if not question.strip():
            st.warning("Vui lòng nhập câu hỏi.")
            st.stop()

        with st.spinner("Đang phân tích câu hỏi và tìm kiếm tài liệu..."):
            qvec = _embed_query(client, question)
            results = _search(index, meta, qvec, question, topk=num_results)

        if not results:
            st.info("❌ Không tìm thấy đoạn trích phù hợp. Vui lòng thử câu hỏi khác hoặc kiểm tra tài liệu.")
            return

        with st.spinner("Đang tổng hợp và phân tích thông tin..."):
            answer = _ask_llm(client, question, results)

        # Display answer
        st.markdown("### ✅ Kết quả")
        st.markdown(answer)

        st.markdown("---")
        st.markdown("### 📚 Nguồn tham chiếu")
        
        df = pd.DataFrame([
            {
                "Số": f"[{i+1}]",
                "Tên file": r["file_name"],
                "Section": "%s %s" % (r.get("section_type","?"), r.get("section_number","?")),
                "Title": r.get("section_title", "")[:40],
                "Type": r.get("content_type", "general"),
                "Relevance": f"{r.get('rerank_score', 0):.3f}",
                "Semantic": f"{r['similarity']:.3f}",
                "Keyword": f"{r.get('keyword_score', 0):.3f}",
            }
            for i, r in enumerate(results)
        ])
        st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("📄 Xem chi tiết các đoạn trích"):
            for i, c in enumerate(results, start=1):
                title = c.get("section_title", "")
                title_display = f" - {title}" if title else ""
                
                st.markdown(f"**[{i}] {c['file_name']}** – {str(c.get('section_type','?')).title()} {c.get('section_number','?')}{title_display}")
                
                # Metadata badges
                badges = []
                if c.get("content_type"):
                    badges.append(f"🏷️ {c['content_type']}")
                if c.get("has_tables"):
                    badges.append("📊 Has tables")
                if c.get("has_lists"):
                    badges.append("📋 Has lists")
                badges.append(f"⭐ {c.get('rerank_score', 0):.3f}")
                
                st.caption(" | ".join(badges))
                
                txt = c["text"]
                if len(txt) > 1500:
                    txt = txt[:1500] + "\n\n... (truncated)"
                st.code(txt, language="markdown")
                st.markdown('---')

    st.caption("Sản phẩm thử nghiệm của Ban Kỹ thuật – VNA. Mọi ý kiến đóng góp vui lòng liên hệ Phòng Kỹ thuật Máy bay.")

if __name__ == "__main__":
    main()
