# Git workflow

The working directory is nested inside an existing repository rooted at `/home/junhyeok/projects`. Keep that repository and its history intact; do not initialize another `.git` directory inside this project.

The supplied GitHub repository is intended to present this project at its own root. The local checkout appears as the `heteroassociation/` subdirectory in the enclosing repository, so synchronize it with Git's subtree command after committing the local changes:

```bash
cd /home/junhyeok/projects
git subtree push --prefix=heteroassociation kist master
```

The local remote name `kist` should point to `https://github.com/chlwnsgur6051-pixel/kist.git`. The initial remote inspection found no published branches. Future research changes should be committed in the existing local repository, then published with the subtree command above. Do not use a plain `git push` for the enclosing workspace repository: that would publish its workspace-level layout rather than this project's root layout.

Stage paths under `heteroassociation/` only. Do not include neighboring workspace folders such as `qwen_lora_benchmark/`. Python caches are excluded by this project's `.gitignore`; experiment outputs and failed runs should remain recorded according to `docs/experiment_protocol.md`.
