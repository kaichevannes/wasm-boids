use crate::{
    bluenoise::{BlueNoise, Sample},
    boid::Boid,
    grid::{Grid, NaiveGrid, Point},
};
use wasm_bindgen::prelude::*;

pub trait BoidFactory {
    fn create_n(&mut self, grid: &dyn Grid<Boid>, number_of_boids: u32) -> Vec<Boid>;
}

struct BlueNoiseBoidFactory {
    noise: BlueNoise,
}

impl BlueNoiseBoidFactory {
    fn new() -> Self {
        Self {
            noise: BlueNoise::new(),
        }
    }
}

impl BoidFactory for BlueNoiseBoidFactory {
    fn create_n(&mut self, grid: &dyn Grid<Boid>, number_of_boids: u32) -> Vec<Boid> {
        let boids = grid.get_points();
        let mut existing_positions = Vec::new();
        let mut existing_velocities = Vec::new();
        let mut existing_accelerations = Vec::new();
        boids.iter().for_each(|b| {
            existing_positions.push(Sample(b.position.0, b.position.1));
            existing_velocities.push(Sample(b.velocity.0, b.velocity.1));
            existing_accelerations.push(Sample(b.acceleration.0, b.acceleration.1));
        });

        let mut new_position_grid = NaiveGrid::new(grid.get_size());
        new_position_grid.set_points(&existing_positions.iter().collect::<Vec<&Sample>>());
        let mut new_velocities_grid = NaiveGrid::new(grid.get_size());
        new_velocities_grid.set_points(&existing_velocities.iter().collect::<Vec<&Sample>>());
        let mut new_accelerations_grid = NaiveGrid::new(grid.get_size());
        new_accelerations_grid.set_points(&existing_accelerations.iter().collect::<Vec<&Sample>>());

        let new_positions = self.noise.generate(&new_position_grid, number_of_boids);
        let new_velocities = self.noise.generate(&new_velocities_grid, number_of_boids);
        let new_accelerations = self
            .noise
            .generate(&new_accelerations_grid, number_of_boids);

        let mut result = Vec::new();
        for i in 0..number_of_boids {
            result.push(Boid {
                position: new_positions.get(i as usize).unwrap().xy(),
                velocity: new_velocities.get(i as usize).unwrap().xy(),
                acceleration: new_accelerations.get(i as usize).unwrap().xy(),
            });
        }

        result
    }
}

#[wasm_bindgen]
pub enum Preset {
    Basic,
    Maruyama,
    Zhang,
}

#[wasm_bindgen]
pub struct Builder {
    number_of_boids: Option<u32>,
    boid_factory: Box<dyn BoidFactory>,
}

#[wasm_bindgen]
impl Builder {
    pub fn from_preset(preset: Preset) -> Builder {
        match preset {
            Preset::Basic => Builder::default().number_of_boids(100),
            Preset::Maruyama => Builder::default().number_of_boids(100),
            Preset::Zhang => Builder::default().number_of_boids(100),
        }
    }

    pub fn number_of_boids(mut self, count: u32) -> Self {
        self.number_of_boids = Some(count);
        self
    }

    pub fn build(mut self) -> Universe {
        let number_of_boids = self
            .number_of_boids
            .expect("Missing field: number_of_boids");
        Universe {
            boids: (self.boid_factory).create_n(&NaiveGrid::<Boid>::new(100.0), number_of_boids),
            boid_factory: self.boid_factory,
        }
    }
}

impl Builder {
    pub fn boid_factory(mut self, factory: Box<dyn BoidFactory>) -> Self {
        self.boid_factory = factory;
        self
    }
}

impl Default for Builder {
    fn default() -> Self {
        Builder {
            number_of_boids: None,
            boid_factory: Box::new(BlueNoiseBoidFactory::new()),
        }
    }
}

#[wasm_bindgen]
pub struct Universe {
    boids: Vec<Boid>,
    boid_factory: Box<dyn BoidFactory>,
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
    use crate::universe::BoidFactory;
    use crate::{boid::Boid, *};
    use std::collections::VecDeque;

    struct TestBoidFactory {
        boids: Vec<Boid>,
    }

    impl BoidFactory for TestBoidFactory {
        fn create_n(&mut self, grid: &dyn grid::Grid<Boid>, number_of_boids: u32) -> Vec<Boid> {
            let mut result = Vec::new();
            (0..number_of_boids)
                .map(|_| result.push(self.boids.pop().expect("No more test boids in self.boids")));
            result
        }
    }

    #[test]
    fn builds_with_expected_number_of_boids() {
        let universe_with_no_boids = universe::Builder::from_preset(universe::Preset::Basic)
            .number_of_boids(0)
            .build();
        assert!(universe_with_no_boids.get_boids().is_empty());

        let universe_with_single_boid = universe::Builder::from_preset(universe::Preset::Basic)
            .number_of_boids(1)
            .build();
        assert_eq!(1, universe_with_single_boid.get_boids().len());

        let universe_with_one_hundred_boids =
            universe::Builder::from_preset(universe::Preset::Basic)
                .number_of_boids(100)
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
            .number_of_boids(boids.len() as u32)
            .boid_factory(Box::new(TestBoidFactory {
                boids: boids.clone(),
            }))
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
