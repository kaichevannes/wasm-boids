#[wasm_bindgen]
pub struct UniverseConfig {
    boid_count: u32,
    /// boids per unit^2
    density: u32,
    noise_fraction: f32,
    attraction_weighting: u32,
    alignment_weighting: u32,
    separation_weighting: u32,
    attraction_radius: f32,
    alignment_radius: f32,
    separation_radius: f32,
}

impl UniverseConfig {
    /// Creates a config based on the values from Maruyama et al.
    pub fn maruyama() -> UniverseConfig {
        UniverseConfig {
            boid_count: 100,
            density: 600,
            noise_fraction: 0.05,
            attraction_weighting: 4,
            alignment_weighting: 30,
            separation_weighting: 1,
            attraction_radius: 0.05,
            alignment_radius: 0.05,
            separation_radius: 0.01,
        }
    }

    /// Creates a config based on the values from Zhang et al.
    pub fn zhang() -> UniverseConfig {
        UniverseConfig {
            boid_count: 100,
            density: 100,
            noise_fraction: 0.05,
            attraction_weighting: 10,
            alignment_weighting: 1,
            separation_weighting: 400,
            attraction_radius: 0.05,
            alignment_radius: 0.05,
            separation_radius: 0.05,
        }
    }
}

impl Default for UniverseConfig {
    fn default() -> Self {
        UniverseConfig::maruyama()
    }
}

fn grid_size(&self) -> u32 {
    (self.boids.len() / self.config.density).isqrt()
}

pub fn boids_ptr(&self) -> *const Boid {
    self.boids.as_ptr()
}
