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
    create_boid: Box<dyn FnMut() -> Boid>,
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

    pub fn build(mut self) -> Universe {
        let boid_count = self.boid_count.expect("Missing field: boid_count");
        Universe {
            boids: (0..boid_count).map(|_| (self.create_boid)()).collect(),
            create_boid: self.create_boid,
        }
    }
}

impl Builder {
    pub fn create_boid<F>(mut self, f: F) -> Self
    where
        F: FnMut() -> Boid + 'static,
    {
        self.create_boid = Box::new(f);
        self
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

#[cfg(test)]
mod tests {
    use crate::{boid::Boid, *};
    use std::collections::VecDeque;

    #[test]
    fn builds_with_expected_number_of_boids() {
        let universe_with_no_boids = universe::Builder::from_preset(universe::Preset::Basic)
            .boid_count(0)
            .build();
        assert!(universe_with_no_boids.get_boids().is_empty());

        let universe_with_single_boid = universe::Builder::from_preset(universe::Preset::Basic)
            .boid_count(1)
            .build();
        assert_eq!(1, universe_with_single_boid.get_boids().len());

        let universe_with_one_hundred_boids =
            universe::Builder::from_preset(universe::Preset::Basic)
                .boid_count(100)
                .build();
        assert_eq!(100, universe_with_one_hundred_boids.get_boids().len());
    }

    #[test]
    fn builds_with_expected_boid_values() {
        let boids: Vec<Boid> = (0..3)
            .map(|n| {
                let x = n as f32;
                Boid {
                    position: (x, x),
                    velocity: (x, x),
                    acceleration: (x, x),
                }
            })
            .collect();
        let universe = universe::Builder::from_preset(universe::Preset::Basic)
            .boid_count(boids.len() as u32)
            .create_boid({
                let mut boids_copy: VecDeque<Boid> = boids.clone().into();
                move || boids_copy.pop_front().unwrap()
            })
            .build();
        assert!(
            universe
                .get_boids()
                .iter()
                .zip(boids.iter())
                .all(|(expected, actual)| expected.position == actual.position)
        )
    }

    #[test]
    fn tick_changes_boid_positions() {
        let mut universe = Universe::build_from_preset(universe::Preset::Basic);
        let original_positions: Vec<(f32, f32)> = universe
            .get_boids()
            .iter()
            .map(|boid| boid.position)
            .collect();
        universe.tick();
        let updated_positions: Vec<(f32, f32)> = universe
            .get_boids()
            .iter()
            .map(|boid| boid.position)
            .collect();
        assert!(
            original_positions
                .iter()
                .zip(updated_positions.iter())
                .all(|(before, after)| before != after)
        );
    }

    #[test]
    fn builds_with_unique_boid_states() {
        let universe = Universe::build_from_preset(universe::Preset::Basic);
        let mut seen_positions = vec![];
        let mut seen_velocities = vec![];
        let mut seen_accelerations = vec![];
        for boid in universe.get_boids() {
            assert!(!seen_positions.contains(&boid.position));
            assert!(!seen_velocities.contains(&boid.velocity));
            assert!(!seen_accelerations.contains(&boid.acceleration));

            seen_positions.push(boid.position);
            seen_velocities.push(boid.velocity);
            seen_accelerations.push(boid.acceleration);
        }
    }
}
