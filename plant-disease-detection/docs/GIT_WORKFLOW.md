# Git & GitHub Workflow — Learn By Doing This Project

You're new to Git, so this walks through real commands, in order, using
*this* project. Type them yourself rather than copy-pasting blindly —
that's how it sticks.

## 0. Install & configure (once per computer)
```bash
git --version          # check it's installed
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

## 1. Turn this folder into a git repository
```bash
cd plant-disease-detection
git init
git status              # shows all files as "untracked" — git sees them but isn't tracking changes yet
```

## 2. Your first commit
```bash
git add .                          # stage everything (the .gitignore already excludes venv/data/db/etc.)
git status                         # now files show as "staged" (green)
git commit -m "Initial project scaffold: Flask app, DB models, training script"
git log                            # see your commit
```

**Why `.gitignore` matters:** without it you'd accidentally commit your
virtual environment, your dataset, and your trained model file — possibly
gigabytes of stuff that doesn't belong in version control. Open
`.gitignore` and read through it now so you know what's excluded and why.

## 3. Create a GitHub repo and connect it
1. On github.com, click **New repository**. Name it `plant-disease-detection`.
   **Don't** initialize it with a README (you already have one) — that
   avoids a merge conflict on your very first push.
2. GitHub will show you commands like these — run them:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/plant-disease-detection.git
   git branch -M main
   git push -u origin main
   ```
3. Refresh the GitHub page — your code is there.

`git remote add origin <url>` links your local repo to GitHub.
`-u origin main` remembers that link so future pushes are just `git push`.

## 4. The daily loop
Every time you sit down to work:
```bash
git status              # what's changed?
git pull                # get any updates (matters more once you collaborate)
# ... do work ...
git add <files>          # or `git add .` for everything
git commit -m "Describe what you did, present tense, short"
git push
```

Commit **often**, in small logical chunks — "add upload form" and
"wire up database save" as two commits, not one giant "did stuff" commit.
Small commits make it possible to undo a single bad change without
losing everything else.

## 5. Branching — practice this deliberately

Right now everything is on `main`. For real practice, build the next
feature on a **branch** instead of directly on `main`:

```bash
git checkout -b add-delete-button     # creates AND switches to a new branch
# ... make changes (e.g. add a delete route + button) ...
git add .
git commit -m "Add ability to delete a prediction from history"
git push -u origin add-delete-button
```

Then on GitHub: open a **Pull Request** from `add-delete-button` into
`main`. Even working solo, review your own diff in the PR view — it's a
different (better) way of catching mistakes than reading in your editor.
Click **Merge**, then back in your terminal:

```bash
git checkout main
git pull                              # get the merged change locally
git branch -d add-delete-button       # delete the now-merged local branch
```

Repeat this branch → commit → PR → merge cycle for each feature in the
"Feature ideas" list in the main README. By the third one it'll feel
natural.

## 6. Useful commands for when things go sideways
```bash
git diff                 # see unstaged changes before committing
git log --oneline         # compact commit history
git checkout -- file.py   # discard uncommitted changes to one file
git reset HEAD~1          # undo the last commit, keep the changes unstaged
```

## 7. Commit message convention (optional but good habit)
```
feat: add delete button to history table
fix: prediction confidence showing as 0%
docs: update README setup steps
refactor: move image validation into helper function
```

## Common beginner mistakes to avoid
- **Committing secrets**: never commit `.env` files with API keys (already gitignored here).
- **Committing the dataset/model**: huge files bloat the repo forever, even if you delete them later (already gitignored).
- **One giant commit at the end**: defeats the purpose of version control. Commit as you go.
- **Working directly on `main` forever**: fine solo at first, but practice branches — it's how real teams work.
