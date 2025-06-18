mod bluenoise;
pub mod boid;
mod boid_factory;
mod grid;
pub mod universe;
pub use universe::Universe;

pub use wasm_bindgen_rayon::init_thread_pool;

use console_error_panic_hook;
pub fn init_panic_hook() {
    console_error_panic_hook::set_once();
}
