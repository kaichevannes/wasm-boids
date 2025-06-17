Boids implementation in wasm.

## Publishing
1. `RUSTFLAGS='--cfg getrandom_backend="wasm_js" -C target-feature=+atomics,+bulk-memory' wasm-pack build --release --target web --scope kaichevannes`
2. `cd pkg && bun publish`
