---
title: Welcome to the LibreYOLO Articles Section
description: Introducing the LibreYOLO blog, where we will share tutorials, release notes, and deep dives on MIT-licensed object detection.
date: 2026-06-10
author: LibreYOLO Team
tags: [LibreYOLO, announcement, object detection]
---

Welcome to the new articles section of the LibreYOLO website! This is where we will publish tutorials, release announcements, and technical deep dives about the project.

## What is LibreYOLO?

LibreYOLO is an MIT-licensed training and inference engine for state-of-the-art YOLO object detection models. Unlike AGPL-licensed alternatives, you can use it freely in proprietary, closed-source commercial applications.

Getting started takes just a few lines of Python:

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

model = LibreYOLO("LibreYOLOXs.pt")
results = model(SAMPLE_IMAGE, save=True)
```

## What to expect here

- **Tutorials**: step-by-step guides for training, exporting, and deploying models
- **Release notes**: what's new in each LibreYOLO version
- **Deep dives**: explainability, architecture details, and benchmarks

## Stay in touch

Check out the [documentation](/docs), browse the [model zoo](/models), or star the project on [GitHub](https://github.com/Libre-YOLO/libreyolo).
