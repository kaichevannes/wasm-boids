use crate::boid::Boid;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub enum Preset {
    Basic,
    Maruyama,
    Zhang,
}

#[wasm_bindgen]
pub struct Builder {
    boid_count: Option<u32>,
    create_boid: Box<dyn Fn() -> Boid>,
}

#[wasm_bindgen]
impl Builder {
    pub fn from_preset(preset: Preset) -> Builder {
        match preset {
            Preset::Basic => Builder::default().boid_count(100),
            Preset::Maruyama => Builder::default().boid_count(100),
            Preset::Zhang => Builder::default().boid_count(100),
        }
    }

    pub fn boid_count(mut self, count: u32) -> Self {
        self.boid_count = Some(count);
        self
    }

    pub fn build(self) -> Universe {
        let boid_count = self.boid_count.expect("Missing field: boid_count");
        Universe {
            boids: (0..boid_count).map(|_| (self.create_boid)()).collect(),
            create_boid: self.create_boid,
        }
    }
}

impl Default for Builder {
    fn default() -> Self {
        Builder {
            boid_count: None,
            create_boid: Box::new(|| Boid {
                position: (0.0, 0.0),
                velocity: (0.0, 0.0),
                acceleration: (0.0, 0.0),
            }),
        }
    }
}

#[wasm_bindgen]
pub struct Universe {
    boids: Vec<Boid>,
    create_boid: Box<dyn FnMut() -> Boid>,
}

#[wasm_bindgen]
impl Universe {
    /// Returns the default [`universe::Builder`].
    ///
    /// Use this method when creating a custom [`Universe`]
    ///
    /// For presets, see [`Universe.build_from_preset`] and [`universe::Builder.from_preset`].
    pub fn builder() -> Builder {
        Builder::default()
    }

    /// Construct a new Universe from a [`universe::Preset`].
    ///
    /// To modify the preset, use [`universe::Builder.from_preset`].
    pub fn build_from_preset(preset: Preset) -> Universe {
        Builder::from_preset(preset).build()
    }

    /// Advance time by one tick.
    ///
    /// This will perform a state update for every Boid in the universe.
    pub fn tick(&mut self) {
        for boid in self.boids.iter_mut() {
            boid.position = (1.0, 1.0);
        }
    }
}

impl Universe {
    pub fn get_boids(&self) -> &Vec<Boid> {
        &self.boids
    }
}
