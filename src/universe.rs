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

    fn generate_samples_from_existing(
        &mut self,
        grid: &dyn Grid<Boid>,
        existing_samples: Vec<Sample>,
        number_of_samples_to_generate: u32,
    ) -> Vec<Sample> {
        let mut new_samples_grid = NaiveGrid::new(grid.get_size());
        new_samples_grid.set_points(&existing_samples.iter().collect::<Vec<&Sample>>());
        self.noise
            .generate(&new_samples_grid, number_of_samples_to_generate)
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

        let mut result = Vec::new();
        let new_positions =
            self.generate_samples_from_existing(grid, existing_positions, number_of_boids);
        let new_velocities =
            self.generate_samples_from_existing(grid, existing_velocities, number_of_boids);
        let new_accelerations =
            self.generate_samples_from_existing(grid, existing_accelerations, number_of_boids);

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
    density: Option<f32>,
    noise_fraction: Option<f32>,
    attraction_weighting: Option<u32>,
    alignment_weighting: Option<u32>,
    separation_weighting: Option<u32>,
    attraction_radius: Option<f32>,
    alignment_radius: Option<f32>,
    separation_radius: Option<f32>,
    boid_factory: Box<dyn BoidFactory>,
}

#[wasm_bindgen]
impl Builder {
    pub fn from_preset(preset: Preset) -> Builder {
        match preset {
            Preset::Basic => Builder::default()
                .number_of_boids(100)
                .density(1.0)
                .noise_fraction(0.05)
                .attraction_weighting(1)
                .alignment_weighting(1)
                .separation_weighting(1)
                .attraction_radius(1.0)
                .alignment_radius(1.0)
                .separation_radius(1.0),
            Preset::Maruyama => Builder::default()
                .number_of_boids(100)
                .density(600.0)
                .noise_fraction(0.0)
                .attraction_weighting(4)
                .alignment_weighting(30)
                .separation_weighting(1)
                .attraction_radius(0.05)
                .alignment_radius(0.05)
                .separation_radius(0.01),
            Preset::Zhang => Builder::default()
                .number_of_boids(100)
                .density(100.0)
                .noise_fraction(0.05)
                .attraction_weighting(10)
                .alignment_weighting(1)
                .separation_weighting(400)
                .attraction_radius(0.05)
                .alignment_radius(0.05)
                .separation_radius(0.05),
        }
    }

    pub fn number_of_boids(mut self, count: u32) -> Self {
        self.number_of_boids = Some(count);
        self
    }

    pub fn density(mut self, density: f32) -> Self {
        self.density = Some(density);
        self
    }

    pub fn noise_fraction(mut self, noise_fraction: f32) -> Self {
        if !(0.0..=1.0).contains(&noise_fraction) {
            panic!("Noise fraction must be between 0 and 1");
        }
        self.noise_fraction = Some(noise_fraction);
        self
    }

    pub fn attraction_weighting(mut self, weighting: u32) -> Self {
        self.attraction_weighting = Some(weighting);
        self
    }

    pub fn alignment_weighting(mut self, weighting: u32) -> Self {
        self.alignment_weighting = Some(weighting);
        self
    }

    pub fn separation_weighting(mut self, weighting: u32) -> Self {
        self.separation_weighting = Some(weighting);
        self
    }

    pub fn attraction_radius(mut self, radius: f32) -> Self {
        self.attraction_radius = Some(radius);
        self
    }

    pub fn alignment_radius(mut self, radius: f32) -> Self {
        self.alignment_radius = Some(radius);
        self
    }

    pub fn separation_radius(mut self, radius: f32) -> Self {
        self.separation_radius = Some(radius);
        self
    }

    pub fn build(mut self) -> Universe {
        let number_of_boids = self
            .number_of_boids
            .expect("Missing field: number_of_boids");
        Universe {
            boids: (self.boid_factory).create_n(&NaiveGrid::<Boid>::new(100.0), number_of_boids),
            boid_factory: self.boid_factory,
            density: self.density.expect("Must provide density"),
            noise_fraction: self.noise_fraction.expect("Must provide noise_fraction"),
            attraction_weighting: self
                .attraction_weighting
                .expect("Must provide attraction_weighting"),
            alignment_weighting: self
                .alignment_weighting
                .expect("Must provide alignment_weighting"),
            separation_weighting: self
                .separation_weighting
                .expect("Must provide separation_weighting"),
            attraction_radius: self
                .attraction_radius
                .expect("Must provide attraction_radius"),
            alignment_radius: self
                .alignment_radius
                .expect("Must provide alignment_radius"),
            separation_radius: self
                .separation_radius
                .expect("Must provide separation_radius"),
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
            density: None,
            noise_fraction: None,
            attraction_weighting: None,
            alignment_weighting: None,
            separation_weighting: None,
            attraction_radius: None,
            alignment_radius: None,
            separation_radius: None,
        }
    }
}

#[wasm_bindgen]
pub struct Universe {
    boids: Vec<Boid>,
    boid_factory: Box<dyn BoidFactory>,
    density: f32,
    noise_fraction: f32,
    attraction_weighting: u32,
    alignment_weighting: u32,
    separation_weighting: u32,
    attraction_radius: f32,
    alignment_radius: f32,
    separation_radius: f32,
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

    struct TestBoidFactory {
        boids: Vec<Boid>,
    }

    impl BoidFactory for TestBoidFactory {
        fn create_n(&mut self, grid: &dyn grid::Grid<Boid>, number_of_boids: u32) -> Vec<Boid> {
            let mut result = Vec::new();
            (0..number_of_boids).for_each(|_| {
                result.push(self.boids.pop().expect("No more test boids in self.boids"))
            });
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

    #[test]
    fn attraction_rule() {
        let b1 = Boid {
            position: (1.0, 1.0),
            velocity: (0.0, 0.0),
            acceleration: (0.0, 0.0),
        };
        let b2 = Boid {
            position: (2.0, 1.0),
            velocity: (0.0, 0.0),
            acceleration: (0.0, 0.0),
        };
        let universe = universe::Builder::from_preset(universe::Preset::Basic)
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b1, b2],
            }))
            .number_of_boids(2)
            .attraction_weighting(1)
            .alignment_weighting(0)
            .separation_weighting(0)
            .build();
    }
}
