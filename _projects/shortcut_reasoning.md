---
layout: page
title: Shortcut-Aware Reasoning Training
description: Gradient-aware training that detects and mitigates shortcut reasoning in language models
importance: 1
category: work
github: https://github.com/HongyuCao3/short-cut-aware-data-centric-reasoning
related_publications: true
---

This project targets shortcut reasoning in language models, where predictions rely on
surface pattern matching, memorization, or keyword correlations rather than logical
inference. The method, Shortcut-Aware Reasoning Training, identifies shortcut-promoting
training samples through gradient misalignment with the reasoning objective and through
answer-token concentration, then applies gradient surgery to adjust the training dynamics.
The repository provides the data-centric training pipeline and the evaluation protocol on
controlled reasoning benchmarks. {% cite cao2026mitigating %}
