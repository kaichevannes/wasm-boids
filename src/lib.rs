mod bluenoise;
pub mod boid;
mod boid_factory;
mod grid;
pub mod universe;
pub use universe::Universe;

pub use wasm_bindgen_rayon::init_thread_pool;

extern crate console_error_panic_hook;
use std::panic;

#[wasm_bindgen]
pub fn init_panic_hook() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
}
