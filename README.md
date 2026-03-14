# Deep Research Pipeline using DSPy

This repository contains an experiment implementing a Deep Research Pipeline using DSPy, inspired by the tutorial below.

Source: https://www.cmpnd.ai/blog/learn-dspy-deep-research.html

## Overview

DSPy provides a structured way to build LLM-powered pipelines. Instead of manually chaining prompts together, developers can define modular, optimizable workflows that can be compiled and improved over time.

This experiment reproduces the deep research workflow from the blog and explores how the same pipeline behaves when executed with different model providers.

## Experiment Setup

The same DSPy pipeline was executed using two different model providers:

- Gemini models
- OpenAI models

Implementation details:

- `predict.py` uses Gemini
- `react.py`, `clarifier.py`, and `workflow.py` use OpenAI

The pipeline structure remains the same, allowing us to compare how different models perform within the same workflow.

DSPy makes this relatively straightforward by allowing model providers to be swapped with minimal code changes.

For additional supported providers, see:
https://docs.litellm.ai/docs/providers

## Goals of the Experiment

This project aims to explore:

- Differences in reasoning patterns across models
- How structured LLM workflows behave across providers
- The benefits of programmatic LLM pipelines compared to prompt chaining

## Pipeline Concept

The deep research pipeline follows a structured workflow:

User Query
   ↓
Clarification / Question Refinement
   ↓
Research & Reasoning
   ↓
Structured Response Generation

Each stage is implemented as a DSPy module, enabling modular experimentation and easier optimization.

## Observations

Initial experiments suggest that running the same structured pipeline across different models can lead to noticeable differences in:

- Reasoning depth
- Answer structure
- Response consistency

Further analysis will be added as additional experiments are completed.

## Future Work

Planned improvements include:

- Adding a CSV comparison of outputs across different models
- Introducing evaluation metrics for response quality
- Testing additional providers supported by LiteLLM
- Automating experiment runs and result collection
