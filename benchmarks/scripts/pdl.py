#!/usr/bin/env python3
"""Parallel, resumable, hash-verified download of one Hugging Face file.

The link throttles each connection, so N concurrent range requests are much faster than one.
Progress is kept in <dest>.parts (finished chunk indices), so re-running resumes.
On completion the SHA-256 is checked against HF's x-linked-etag; mismatch -> file renamed .bad.

Usage: pdl.py <repo> <filename> <dest_dir> [workers=8]
"""
import concurrent.futures as cf
import hashlib
import os
import sys
import threading
import time

import httpx

CHUNK = 32 << 20


def main():
    repo, fname, dest_dir = sys.argv[1:4]
    workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    url = f"https://huggingface.co/{repo}/resolve/main/{fname}"
    dest = os.path.join(dest_dir, fname)
    parts_path = dest + ".parts"

    h = httpx.head(url, follow_redirects=False, timeout=30).headers
    size = int(h["x-linked-size"])
    sha = h["x-linked-etag"].strip('"')
    n = -(-size // CHUNK)
    done = set()
    if os.path.exists(parts_path):
        done = {int(x) for x in open(parts_path).read().split()}
    if not os.path.exists(dest):
        with open(dest, "wb") as f:
            f.truncate(size)
    lock = threading.Lock()
    t0, got0 = time.time(), len(done) * CHUNK
    print(f"{fname}: {size/2**30:.2f} GiB, {n} chunks, {len(done)} already done, sha256 {sha[:12]}…", flush=True)

    def fetch(i):
        lo, hi = i * CHUNK, min(size, (i + 1) * CHUNK) - 1
        for attempt in range(50):
            try:
                with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(60, read=120)) as c:
                    r = c.get(url, headers={"Range": f"bytes={lo}-{hi}"})
                    if r.status_code == 429:
                        time.sleep(30); continue
                    r.raise_for_status()
                    data = r.content
                if len(data) != hi - lo + 1:
                    raise IOError(f"short read {len(data)}")
                with open(dest, "r+b") as f:
                    f.seek(lo); f.write(data)
                with lock:
                    done.add(i)
                    with open(parts_path, "a") as p:
                        p.write(f"{i}\n")
                    if len(done) % 10 == 0 or len(done) == n:
                        mb_s = (len(done) * CHUNK - got0) / max(1, time.time() - t0) / 2**20
                        eta = (n - len(done)) * CHUNK / 2**20 / max(mb_s, 0.01) / 60
                        print(f"  {len(done)}/{n} chunks  {mb_s:.2f} MB/s  eta {eta:.0f} min", flush=True)
                return
            except Exception as e:
                time.sleep(min(60, 2 ** min(attempt, 6)))
        raise RuntimeError(f"chunk {i} failed")

    todo = [i for i in range(n) if i not in done]
    with cf.ThreadPoolExecutor(workers) as ex:
        list(ex.map(fetch, todo))

    print("verifying sha256…", flush=True)
    hs = hashlib.sha256()
    with open(dest, "rb") as f:
        while b := f.read(16 << 20):
            hs.update(b)
    if hs.hexdigest() == sha:
        os.remove(parts_path)
        print(f"OK {dest} sha256 verified", flush=True)
    else:
        os.rename(dest, dest + ".bad")
        print(f"HASH MISMATCH {hs.hexdigest()} != {sha}; renamed to .bad", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
