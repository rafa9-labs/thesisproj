---
name: pr-creator
description: Stage changes, create a properly-named branch, commit with conventional commit message, push, and create a PR with summary of changes. Reads AGENTS.md git workflow rules. Never merges own PRs. Always runs test-runner first.
---

# Skill: /pr-creator

**Trigger:** User types `/pr-creator`.

**Objective:** Create a complete, reviewable PR following project conventions. Never merge own PRs. Always validate first.

**Protocol:**

1. **Read AGENTS.md / CLAUDE.md git rules:**
   - Current branch convention: `feature/phase<N>-<descriptive-kebab-case>` or `sprint<N>/<sub-task-id>`
   - Remote: `origin`
   - Always push before ending session.

2. **Pre-flight checks (MANDATORY):**
   - Run `/test-runner` first. If any step fails, STOP and report. Do not create PR.
   - Run `git status` to see changes.
   - Run `git diff --staged` and `git diff` to see what will be committed.

3. **Branch creation:**
   - Derive branch name from the current task/sprint:
     - If working on Sprint 9 task: `sprint9/pyinstaller-build-pipeline`
     - If working on a phase: `feature/phase2-api-bridge`
   - Ask user to confirm branch name before creating.

4. **Stage and commit:**
   - Stage relevant files only (no .env, no secrets, no large binary files).
   - Conventional commit format: `type(scope): description`
     - Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
     - Example: `feat(electron): add PyInstaller spec for Python bundling`
   - Never use `--no-verify` or skip hooks.

5. **Push and create PR:**
   - Push: `git push -u origin <branch>`
   - Create PR via `gh pr create`:
     - Title: conventional commit style
     - Body: summary of changes, files touched, test results
     - Base: determine from ROADMAP (current phase branch or main)

6. **Rules:**
   - **NEVER merge own PRs.** Always leave for review.
   - **NEVER commit secrets** (.env, credentials, API keys).
   - **NEVER use --force push** to main/master.
   - If commit fails due to hooks, fix the issue and create a NEW commit (do not amend unless explicitly requested).
