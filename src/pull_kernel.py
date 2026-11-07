"""Download a kernel's output files with resume-on-failure (the stock CLI restarts from zero)."""
from __future__ import annotations

import os
import re
import sys
import time

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def list_files(api: KaggleApi, owner: str, slug: str):
    files, token = [], None
    while True:
        req = ApiListKernelSessionOutputRequest()
        req.user_name, req.kernel_slug = owner, slug
        api._set_paging(req, 50, token)
        with api.build_kaggle_client() as kaggle:
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
        files += list(resp.files or [])
        token = resp.next_page_token
        if not token:
            return files


def fetch(url: str, path: str, tries: int = 40) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    done = os.path.getsize(path) if os.path.exists(path) else 0
    for _ in range(tries):
        try:
            head = requests.get(url, headers={"Range": f"bytes={done}-"}, stream=True, timeout=60)
            if head.status_code == 416:
                return
            mode = "ab" if head.status_code == 206 else "wb"
            if mode == "wb":
                done = 0
            with open(path, mode) as f:
                for chunk in head.iter_content(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
            total = int(head.headers.get("Content-Range", "/0").split("/")[-1] or 0)
            if head.status_code == 200 or (total and done >= total):
                return
        except requests.RequestException as e:
            print(f"  retry at {done:,} bytes: {type(e).__name__}", flush=True)
            time.sleep(3)
    raise RuntimeError(f"gave up on {path}")


if __name__ == "__main__":
    kernel, dest = sys.argv[1], sys.argv[2]
    pattern = re.compile(sys.argv[3]) if len(sys.argv) > 3 else None
    owner, slug = kernel.split("/")
    api = KaggleApi()
    api.authenticate()
    for item in list_files(api, owner, slug):
        if pattern and not pattern.search(item.file_name):
            continue
        out = os.path.join(dest, item.file_name)
        fetch(item.url, out)
        print(f"{item.file_name}: {os.path.getsize(out):,} bytes", flush=True)
