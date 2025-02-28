# Custom LLM Demo

This repository demonstrates how to integrate a custom LLM (Large Language Model) with Nemo Guardrails.

## Overview

Nemo Guardrails provides a framework for adding safety controls and behavioral guidelines to language models. This demo shows how to integrate your own custom LLM implementation with the Nemo Guardrails framework.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Nemo Guardrails package installed

### Usage

1. Create a custom LLM instance by following the implementation in `llm_providers/custom_llm.py`
2. Initialize the Rails instance as shown in `test_llm.py` to test the conversation flow
3. Alternatively, start an interactive chat session by running:
   ```bash
   nemoguardrails chat
   ```

## Directory Structure

```
custom_llm_demo/
├── llm_providers/
│   └── custom_llm.py    # Custom LLM implementation
├── test_llm.py          # Example usage and tests
└── README.md