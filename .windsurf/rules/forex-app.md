# PinguForex App Constitution

## 1. Architectural Integrity
- **Architecture**: Move from Procedural Python to a **Modular API-First** structure.
- **Data Factory**: All Oanda/CSV logic must live in a standalone `DataHandler`.
- **Configurability**: Use `Pydantic` models for all indicators and Oanda credentials.
- **Exportability**: The engine must be UI-agnostic so we can export it to a Next.js or Streamlit dashboard later.

## 2. Computational Efficiency
- **Vectorization First**: Replace loops with `pandas` vectorized operations (`.rolling()`, `.shift()`, `.apply()`).
- **Caching**: Implement a decorator for indicator calculations to prevent redundant math.

## 3. Strict Development Gate
- **Divide & Conquer**: Never refactor more than one module at a time. Split complex tasks into multiple, smaller prompts to prevent timeouts.
- **Strict Output Limitation**: Output ONLY the new or updated variables and functions. Do NOT rewrite package declarations, imports, or the rest of the file.
- **Language**: Always write prompts and code comments in English.
- **Confirmation**: Always provide a "Plan of Action" and wait for a 'Go' before writing code.
