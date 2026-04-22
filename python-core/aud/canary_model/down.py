from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/canary-qwen-2.5b",
    local_dir="D:/canary_model",
    local_dir_use_symlinks=False,
    resume_download=True,
)