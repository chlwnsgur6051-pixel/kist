# Git workflow

The working directory is nested inside an existing repository rooted at `/home/junhyeok/projects`. Keep that repository and its history intact; do not initialize another `.git` directory inside either project.

The supplied GitHub repository holds the workspace projects as top-level directories: `heteroassociation/` and `qwen_lora_benchmark/`. Commit work in the enclosing repository, then push the workspace branch:

```bash
cd /home/junhyeok/projects
git push kist master
```

The local remote `kist` points to `https://github.com/chlwnsgur6051-pixel/kist.git`. The initial remote inspection found no published branches. Future project changes should be committed in the existing local repository and published with the command above.

Stage intended project paths only. Python caches and model checkpoint files are excluded by project `.gitignore` files. Keep experiment outputs and failed runs according to `docs/experiment_protocol.md` and the Qwen benchmark README.
