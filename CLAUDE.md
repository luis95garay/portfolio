# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a static portfolio repository for a machine learning engineer. It contains only:
- `README.md` — the portfolio content displayed on GitHub
- `public/` — image assets referenced by the README

There is no build system, framework, package manager, or test suite. All content is Markdown with embedded image references.

## Structure

- `README.md` — portfolio showcasing 9 ML projects (bank churn, chatbot, image classification, sentiment analysis, object detection, time series, clustering, graph theory, generative AI)
- `public/*.png` — screenshots and result images for each project

## Working in This Repo

- Edit `README.md` directly to add, update, or reorder projects
- Add new project images to `public/` and reference them in README as `![alt](public/filename.png)`
- Image paths are relative to the repo root (GitHub renders them correctly)
- Each project entry follows this pattern:
  ```markdown
  ## N) Project Title
  - Objective/description
  - Tools: ...
    [Link text](https://github.com/luis95garay/repo-name)
    ![alt text](public/image.png)
  ```
