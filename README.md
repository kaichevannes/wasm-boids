Boids implementation in wasm.

## Publishing
1. `RUSTFLAGS='--cfg getrandom_backend="wasm_js" -C target-feature=+atomics,+bulk-memory' wasm-pack build --release --target web --scope kaichevannes`
2. `cd pkg && bun publish`
3. Replace files in pkg/package.json with:
```
"files": [
    "wasm_boids_bg.wasm",
    "wasm_boids_bg.wasm.d.ts",
    "wasm_boids.js",
    "wasm_boids.d.ts",
    "snippets"
],
```
