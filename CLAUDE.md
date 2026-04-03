# Kore

Zig "standard library" of utilities missing from the real stdlib. Uses **Zig tip-of-master** (currently 0.16.x+).

## Build & Test

- Build and test **per-submodule**: `cd src/<module> && zig build test`
- Each submodule has its own `build.zig` / `build.zig.zon`
- GPU + OpenCL is always available on dev machines -- just run ML tests normally
- When adding a new submodule, only create the submodule files -- do NOT wire into root `build.zig` / `kore.zig`

## Zig API Pitfalls

- This project uses bleeding-edge Zig. **Do not guess APIs** -- if unsure, ask me directly
- Do NOT run excessive greps/finds trying to figure out a Zig stdlib API
- Common mistakes to avoid:
  - `std.Io` (capitalized) not `std.io`
  - RNG lives in `std.Io.random`, NOT `std.crypto`

## Code Style

- Sparse doc comments -- only on non-obvious functions. Code should be self-documenting
- Inferred error sets (`!T`) are fine -- no need to enumerate unless there's a reason
- Tests: include inline tests for non-trivial logic, skip for simple wrappers/glue
- OpenCL kernels: always separate `.cl` files in `kernels/` dir, loaded via `@embedFile`
- SIMD: always use `@Vector` for portability (must run on both x86/AVX2 and ARM/NEON)

## Workflow

- Propose approach/API before writing code for anything non-trivial
- Commit messages: short imperative style (e.g., `add BitReader`, `fix matmul padding`)
