
---
name: fruit-ml-project
description: Develops, maintains, trains, evaluates and documents a fruit quality and ripeness classification machine learning project. Use when working on dataset processing, image preprocessing, feature extraction, ML models, evaluation, prediction or project documentation.
---

# Fruit Quality and Ripeness Classification Skill

## 1. Project Context

The project classifies fruit quality and ripeness using machine learning and image processing.

Before performing any task, read:

- `docs/AGENT_PROJECT_SPEC.md`

This document is the primary project specification.

## 2. Mandatory Rules

- Inspect the existing repository before modifying anything.
- Follow the current architecture, framework and coding conventions.
- Do not create a new project when an existing project is available.
- Do not delete, overwrite or reorganize datasets without permission.
- Do not invent dataset statistics, experimental results or model accuracy.
- Do not assume fruit categories or labels without inspecting the actual dataset.
- Clearly distinguish verified facts from assumptions.
- Preserve existing working functionality.

## 3. Development Workflow

For every task:

1. Read the project specification.
2. Inspect the repository structure.
3. Identify relevant source files, datasets and dependencies.
4. Understand the current implementation.
5. Propose a concise implementation plan.
6. Implement the requested changes.
7. Run relevant tests and verify the results.
8. Update documentation when necessary.
9. Summarize modified files, commands and test results.

## 4. Machine Learning Requirements

When working on the ML pipeline:

- Inspect and validate the dataset before training.
- Check image formats, dimensions, labels and class distribution.
- Prevent data leakage between training, validation and test sets.
- Apply preprocessing consistently during training and inference.
- Compare suitable machine learning models.
- Use appropriate evaluation metrics.
- Save model configurations and reproducible experiment information.
- Never report metrics that were not actually measured.

## 5. Image Processing

Consider the following techniques when appropriate:

- RGB, HSV and Lab color spaces.
- Image resizing and normalization.
- Noise reduction and filtering.
- Histogram analysis.
- Otsu thresholding and segmentation.
- Color, texture and shape feature extraction.
- GLCM and LBP.

Do not add unnecessary processing steps without evaluating their impact.

## 6. Model Evaluation

Use appropriate metrics:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-score
- ROC/AUC when applicable

Explain the limitations of each model and compare results using the same evaluation dataset.

## 7. Documentation

Keep project documentation in `docs/`.

When updating the report:

- Follow the chapter structure in `docs/AGENT_PROJECT_SPEC.md`.
- Use actual implementation details.
- Use actual experimental results.
- Identify missing information instead of fabricating it.

## 8. Completion Report

After each task, report:

- What was inspected.
- What was changed.
- Files created or modified.
- Commands and tests executed.
- Actual test results.
- Remaining issues.
- Recommended next steps.
