#!/usr/bin/env python3
from __future__ import annotations
from embeddings import E5Embeddings


import os
import json
import glob
import argparse
import shutil
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from langchain.docstore.document import Document
from langchain_chroma import Chroma


MANUAL_JSON = "data/manual_chunks/manual_chunks.json"
TRANSCRIPTS_DIR = "data/transcripts"

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ableton")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")

WIN_SEC_DEFAULT = 40
STRIDE_SEC_DEFAULT = 30


def load_manual_docs(path: str) -> List[Document]:
    if not os.path.exists(path):
        print(f"[WARN] No manual found at {path}")
        return []
    data = json.load(open(path, "r", encoding="utf-8"))
    docs: List[Document] = []
    for row in data:
        txt = (row.get("text") or "").strip()
        if not txt:
            continue
        src = row.get("source", "manual")
        docs.append(Document(page_content=txt, metadata={"type": "manual", "source": src}))
    return docs


def _load_segments(fp: str) -> List[Dict]:
    try:
        segs = json.load(open(fp, "r", encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Cannot read JSON file: {fp} -> {e}")

    if not isinstance(segs, list):
        raise ValueError(f"Waiting for a list of segments {fp}")

    cleaned: List[Dict] = []
    for s in segs:
        if "start" not in s:
            continue
        t = s.get("text")
        if not t:
            continue
        try:
            start = float(s["start"])
        except Exception:
            continue
        txt = str(t).strip()
        if not txt:
            continue
        cleaned.append({"start": start, "text": txt})
    cleaned.sort(key=lambda x: x["start"])
    return cleaned


def load_video_windows(
    dirpath: str,
    window_sec: int,
    stride_sec: int,
    min_chars: int = 20,
) -> List[Document]:
    if not os.path.isdir(dirpath):
        print(f"[WARN] No transcripts dir at {dirpath}")
        return []

    docs: List[Document] = []
    fps = sorted(glob.glob(os.path.join(dirpath, "*.json")))

    for fp in tqdm(fps, desc="Indexing videos"):
        vid = Path(fp).stem

        try:
            segs = _load_segments(fp)
        except Exception as e:
            print(f"[WARN] Skip {fp}: {e}")
            continue

        if not segs:
            continue
        max_t = int(segs[-1]["start"] + window_sec)
        n = len(segs)
        left = 0 
        t = 0
        while t <= max_t:
            start, end = t, t + window_sec
            t += stride_sec

            while left < n and segs[left]["start"] < start:
                left += 1

            chunk_texts = []
            i = left
            while i < n and segs[i]["start"] < end:
                chunk_texts.append(segs[i]["text"])
                i += 1

            if not chunk_texts:
                continue

            content = " ".join(x for x in chunk_texts if x)
            content = " ".join(content.split())

            if len(content) < min_chars:
                continue

            docs.append(Document(
                page_content=content,
                metadata={
                    "type": "video",
                    "source": f"https://www.youtube.com/watch?v={vid}",
                    "video_id": vid,
                    "start": float(start),
                    "end": float(end),
                }
            ))

    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="Delete and rebuild Chroma")
    ap.add_argument("--win", type=int, default=WIN_SEC_DEFAULT, help="size of windows (s)")
    ap.add_argument("--stride", type=int, default=STRIDE_SEC_DEFAULT, help="Stride (s)")
    args = ap.parse_args()

    if args.rebuild and os.path.isdir(CHROMA_DIR):
        print(f"[INFO] Removing {CHROMA_DIR} ...")
        shutil.rmtree(CHROMA_DIR)

    manual_docs = load_manual_docs(MANUAL_JSON)
    video_docs = load_video_windows(TRANSCRIPTS_DIR, window_sec=args.win, stride_sec=args.stride)
    all_docs = manual_docs + video_docs

    print(f"[INFO] Manual docs: {len(manual_docs)} | Video docs: {len(video_docs)} | Total: {len(all_docs)}")

    if not all_docs:
        print("[ERROR] No documents to index.")
        return

    emb = E5Embeddings(model_name=EMBED_MODEL)
    vs = Chroma(collection_name=COLLECTION_NAME, persist_directory=CHROMA_DIR, embedding_function=emb)

    batch = 256
    for i in range(0, len(all_docs), batch):
        vs.add_documents(all_docs[i:i+batch])
        print(f"[INFO] Indexed {min(i+batch, len(all_docs))}/{len(all_docs)}")

    print("[INFO] Chroma persisted to", CHROMA_DIR)

if __name__ == "__main__":
    main()
