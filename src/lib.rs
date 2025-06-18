mod bluenoise;
pub mod boid;
mod boid_factory;
mod grid;
pub mod universe;
pub use universe::Universe;

pub use wasm_bindgen_rayon::init_thread_pool;
