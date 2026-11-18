# Project preferences

- Before making any changes to the repository, describe the proposed changes and wait for explicit confirmation from the user.
- Put documentation in `docs/` and prioritize code examples to explain behavior and usage.
- Keep changes small, direct, and limited to the requested work. Preserve existing names, structure, and style where practical.
- Assume ordinary internal inputs are valid. Avoid excessive defensive validation, numeric-limit checks, and speculative error handling. Keep CUDA API error handling.
- If a large rework is necessary, stop before starting it, explain why, and discuss the scope with the user.
- Avoid incidental nitpicks or cleanup such as changing `int` to `size_t`, adding casts, or renaming helpers unless required for the requested behavior or a concrete correctness issue. Prioritize readability and small diffs.
- Do not run tests unless the user asks. Clearly distinguish compilation checks from tests.
- Use braces for `if`/`else` bodies, including single-statement bodies. Exception: very short one-line guards such as `if (token_ids.empty()) return;` are fine. Do not omit braces just to shorten ordinary conditionals.
