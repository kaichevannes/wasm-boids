use crate::{
    bluenoise::{BlueNoise, Sample},
    boid::{Boid, Vec2},
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
        new_samples_grid.set_points(existing_samples);
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
                position: new_positions.get(i as usize).unwrap().xy().into(),
                velocity: new_velocities.get(i as usize).unwrap().xy().into(),
                acceleration: new_accelerations.get(i as usize).unwrap().xy().into(),
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
    grid_size: Option<f32>,
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
                .grid_size(100.0)
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
        self.grid_size = None;
        self.density = Some(density);
        self
    }

    pub fn grid_size(mut self, size: f32) -> Self {
        self.density = None;
        self.grid_size = Some(size);
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
        let mut grid;
        let attraction_weighting = self
            .attraction_weighting
            .expect("Must provide attraction_weighting");
        let alignment_weighting = self
            .alignment_weighting
            .expect("Must provide alignment_weighting");
        let separation_weighting = self
            .separation_weighting
            .expect("Must provide separation_weighting");
        let total_weighting = attraction_weighting + alignment_weighting + separation_weighting;

        let density = self.density.unwrap_or_else(|| {
            if number_of_boids == 0 {
                panic!("Must set a density when creating a grid with no boids.")
            }
            number_of_boids as f32
                / self
                    .grid_size
                    .expect("Either density or grid_size must be set.")
                    .powi(2)
        });

        grid = NaiveGrid::new((number_of_boids as f32 / density).sqrt());
        grid.set_points(self.boid_factory.create_n(&grid, number_of_boids));
        Universe {
            density,
            noise_fraction: self.noise_fraction.expect("Must provide noise_fraction"),
            attraction_weighting: attraction_weighting as f32 / total_weighting as f32,
            alignment_weighting: alignment_weighting as f32 / total_weighting as f32,
            separation_weighting: separation_weighting as f32 / total_weighting as f32,
            attraction_radius: self
                .attraction_radius
                .expect("Must provide attraction_radius"),
            alignment_radius: self
                .alignment_radius
                .expect("Must provide alignment_radius"),
            separation_radius: self
                .separation_radius
                .expect("Must provide separation_radius"),
            boid_factory: self.boid_factory,
            grid: Box::new(grid),
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
            grid_size: None,
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
    density: f32,
    noise_fraction: f32,
    attraction_weighting: f32,
    alignment_weighting: f32,
    separation_weighting: f32,
    attraction_radius: f32,
    alignment_radius: f32,
    separation_radius: f32,
    grid: Box<dyn Grid<Boid>>,
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
        let mut boids = Vec::new();
        for boid in self.grid.get_points().iter() {
            let acceleration = boid.acceleration
                + self.attraction_acceleration(boid) * self.attraction_weighting
                + self.alignment_acceleration(boid) * self.alignment_weighting
                + self.separation_acceleration(boid) * self.separation_weighting;
            // should be normalised, or bounded by a max velocity.
            let velocity = boid.velocity + acceleration;
            // make sure still in bounds of the grid.
            let position = boid.position + velocity;

            boids.push(Boid {
                position,
                velocity,
                acceleration,
            });
        }
        self.grid.set_points(boids);
    }
}

impl Universe {
    pub fn get_boids(&self) -> &[Boid] {
        self.grid.get_points()
    }

    fn attraction_acceleration(&self, boid: &Boid) -> Vec2 {
        let neighbors = self.grid.neighbors(boid, self.attraction_radius);
        if neighbors.is_empty() {
            return Vec2(0.0, 0.0);
        }

        let total_position = neighbors.iter().fold(Vec2(0.0, 0.0), |acc, n| {
            acc + self.wrapped_position(boid.position, n.position)
        });
        let average_position = total_position / neighbors.len();
        average_position - boid.position
    }

    fn alignment_acceleration(&self, boid: &Boid) -> Vec2 {
        let neighbors = self.grid.neighbors(boid, self.alignment_radius);
        if neighbors.is_empty() {
            return Vec2(0.0, 0.0);
        }

        let total_velocity = neighbors
            .iter()
            .fold(Vec2(0.0, 0.0), |acc, n| acc + n.velocity);
        let average_velocity = total_velocity / neighbors.len();
        average_velocity - boid.velocity
    }

    fn separation_acceleration(&self, boid: &Boid) -> Vec2 {
        let neighbors = self.grid.neighbors(boid, self.attraction_radius);
        if neighbors.is_empty() {
            return Vec2(0.0, 0.0);
        }

        let total_position = neighbors.iter().fold(Vec2(0.0, 0.0), |acc, n| {
            acc + self.wrapped_position(boid.position, n.position)
        });
        let average_position = total_position / neighbors.len();
        boid.position - average_position
    }

    fn wrapped_position(&self, starting: Vec2, other: Vec2) -> Vec2 {
        let grid_size = self.grid.get_size();
        let (x1, y1) = starting.into();
        let (x2, y2) = other.into();

        let result_x;
        if x2 - x1 > grid_size / 2.0 {
            result_x = x2 - grid_size;
        } else if x2 - x1 < -grid_size / 2.0 {
            result_x = x2 + grid_size;
        } else {
            result_x = x2;
        }

        let result_y;
        if y2 - y1 > grid_size / 2.0 {
            result_y = y2 - grid_size;
        } else if y2 - y1 < -grid_size / 2.0 {
            result_y = y2 + grid_size;
        } else {
            result_y = y2;
        }

        Vec2(result_x, result_y)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use crate::boid::Vec2;
    use crate::grid::Point;
    use crate::universe::BoidFactory;
    use crate::{boid::Boid, *};

    struct TestBoidFactory {
        boids: VecDeque<Boid>,
    }

    impl BoidFactory for TestBoidFactory {
        fn create_n(&mut self, grid: &dyn grid::Grid<Boid>, number_of_boids: u32) -> Vec<Boid> {
            let mut result = Vec::new();
            (0..number_of_boids).for_each(|_| {
                result.push(
                    self.boids
                        .pop_front()
                        .expect("No more test boids in self.boids"),
                )
            });
            result
        }
    }

    #[test]
    fn builds_with_expected_number_of_boids() {
        let universe_with_single_boid = universe::Builder::from_preset(universe::Preset::Basic)
            .number_of_boids(1)
            .build();
        assert_eq!(1, universe_with_single_boid.get_boids().len());

        let universe_with_one_hundred_boids =
            universe::Builder::from_preset(universe::Preset::Basic)
                .number_of_boids(100)
                .build();
        assert_eq!(100, universe_with_one_hundred_boids.get_boids().len());

        let universe_with_one_thousand_boids =
            universe::Builder::from_preset(universe::Preset::Basic)
                .number_of_boids(1000)
                .build();
        assert_eq!(1000, universe_with_one_thousand_boids.get_boids().len());
    }

    #[test]
    #[should_panic]
    fn panics_when_trying_to_create_sized_grid_with_no_boids() {
        universe::Builder::from_preset(universe::Preset::Basic)
            .number_of_boids(0)
            .build();
    }

    #[test]
    fn builds_with_expected_boid_values() {
        let boids: Vec<Boid> = (0..3)
            .map(|n| {
                let x = n as f32;
                Boid {
                    position: Vec2(x, x),
                    velocity: Vec2(x, x),
                    acceleration: Vec2(x, x),
                }
            })
            .collect();
        let universe = universe::Builder::from_preset(universe::Preset::Basic)
            .number_of_boids(boids.len() as u32)
            .boid_factory(Box::new(TestBoidFactory {
                boids: boids.clone().into(),
            }))
            .build();
        println!("{:?}", universe.get_boids());
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
        let original_positions: Vec<Vec2> = universe
            .get_boids()
            .iter()
            .map(|boid| boid.position)
            .collect();
        universe.tick();
        let updated_positions: Vec<Vec2> = universe
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
    fn boids_next_to_each_other_are_attracted() {
        let b1 = Boid {
            position: Vec2(1.0, 1.0),
            velocity: Vec2(0.0, 0.0),
            acceleration: Vec2(0.0, 0.0),
        };
        let b2 = Boid {
            position: Vec2(2.0, 1.0),
            velocity: Vec2(0.0, 0.0),
            acceleration: Vec2(0.0, 0.0),
        };
        let mut universe = universe::Builder::from_preset(universe::Preset::Basic)
            .number_of_boids(2)
            .grid_size(10.0)
            .attraction_weighting(1)
            .alignment_weighting(0)
            .separation_weighting(0)
            .attraction_radius(1.0)
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b1, b2].into(),
            }))
            .build();
        universe.tick();
        let boids = universe.get_boids();
        let (x1, _) = boids.first().unwrap().xy();
        let (x2, _) = boids.last().unwrap().xy();
        assert!(x1 > 1.0);
        assert!(x2 < 2.0);

        let b3 = Boid {
            position: Vec2(6.0, 5.0),
            velocity: Vec2(0.0, 0.0),
            acceleration: Vec2(0.0, 0.0),
        };
        let b4 = Boid {
            position: Vec2(6.0, 4.0),
            velocity: Vec2(0.0, 0.0),
            acceleration: Vec2(0.0, 0.0),
        };
        let mut universe = universe::Builder::from_preset(universe::Preset::Basic)
            .number_of_boids(2)
            .grid_size(10.0)
            .attraction_weighting(1)
            .alignment_weighting(0)
            .separation_weighting(0)
            .attraction_radius(1.0)
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b3, b4].into(),
            }))
            .build();
        universe.tick();
        let boids = universe.get_boids();
        let (_, y1) = boids.first().unwrap().xy();
        let (_, y2) = boids.last().unwrap().xy();
        assert!(y1 < 5.0);
        assert!(y2 > 4.0);
    }

    #[test]
    fn boids_wrapping_are_attracted() {
        let b1 = Boid {
            position: Vec2(5.0, 9.0),
            velocity: Vec2(0.0, 0.0),
            acceleration: Vec2(0.0, 0.0),
        };
        let b2 = Boid {
            position: Vec2(5.0, 1.0),
            velocity: Vec2(0.0, 0.0),
            acceleration: Vec2(0.0, 0.0),
        };
        let mut universe = universe::Builder::from_preset(universe::Preset::Basic)
            .number_of_boids(2)
            .grid_size(10.0)
            .attraction_weighting(1)
            .alignment_weighting(0)
            .separation_weighting(0)
            .attraction_radius(2.0)
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b1, b2].into(),
            }))
            .build();
        universe.tick();
        let boids = universe.get_boids();
        let (_, y1) = boids.first().unwrap().xy();
        let (_, y2) = boids.last().unwrap().xy();
        assert!(y1 > 9.0);
        assert!(y2 < 1.0);

        let b3 = Boid {
            position: Vec2(1.0, 5.0),
            velocity: Vec2(0.0, 0.0),
            acceleration: Vec2(0.0, 0.0),
        };
        let b4 = Boid {
            position: Vec2(9.0, 5.0),
            velocity: Vec2(0.0, 0.0),
            acceleration: Vec2(0.0, 0.0),
        };
        let mut universe = universe::Builder::from_preset(universe::Preset::Basic)
            .number_of_boids(2)
            .grid_size(10.0)
            .attraction_weighting(1)
            .alignment_weighting(0)
            .separation_weighting(0)
            .attraction_radius(2.0)
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b3, b4].into(),
            }))
            .build();
        universe.tick();
        let boids = universe.get_boids();
        let (x1, _) = boids.first().unwrap().xy();
        let (x2, _) = boids.last().unwrap().xy();
        assert!(x1 < 1.0);
        assert!(x2 > 9.0);

        let b5 = Boid {
            position: Vec2(1.0, 9.0),
            velocity: Vec2(0.0, 0.0),
            acceleration: Vec2(0.0, 0.0),
        };
        let b6 = Boid {
            position: Vec2(9.0, 1.0),
            velocity: Vec2(0.0, 0.0),
            acceleration: Vec2(0.0, 0.0),
        };
        let mut universe = universe::Builder::from_preset(universe::Preset::Basic)
            .number_of_boids(2)
            .grid_size(10.0)
            .attraction_weighting(1)
            .alignment_weighting(0)
            .separation_weighting(0)
            .attraction_radius(3.0)
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b5, b6].into(),
            }))
            .build();
        universe.tick();
        let boids = universe.get_boids();
        let (x1, y1) = boids.first().unwrap().xy();
        let (x2, y2) = boids.last().unwrap().xy();
        assert!(x1 < 1.0 && y1 > 9.0);
        assert!(x2 > 9.0 && y2 < 1.0);
    }

    #[test]
    fn boids_next_to_each_other_are_aligned() {
        let b1 = Boid {
            position: Vec2(1.0, 1.0),
            velocity: Vec2(0.0, 0.0),
            acceleration: Vec2(0.0, 0.0),
        };
        let b2 = Boid {
            position: Vec2(2.0, 1.0),
            velocity: Vec2(1.0, 0.0),
            acceleration: Vec2(0.0, 0.0),
        };
        let mut universe = universe::Builder::from_preset(universe::Preset::Basic)
            .number_of_boids(2)
            .grid_size(10.0)
            .attraction_weighting(0)
            .alignment_weighting(1)
            .separation_weighting(0)
            .alignment_radius(1.0)
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b1, b2].into(),
            }))
            .build();
        universe.tick();
        let boids = universe.get_boids();
        let Vec2(x1, y1) = boids.first().unwrap().velocity;
        let Vec2(x2, y2) = boids.last().unwrap().velocity;
        assert!(x1 > 0.0 && y1 == 0.0);
        assert!(x2 < 1.0 && y2 == 0.0);

        let b3 = Boid {
            position: Vec2(1.0, 1.0),
            velocity: Vec2(0.0, 1.0),
            acceleration: Vec2(0.0, 0.0),
        };
        let b4 = Boid {
            position: Vec2(2.0, 1.0),
            velocity: Vec2(0.0, 0.0),
            acceleration: Vec2(0.0, 0.0),
        };
        let mut universe = universe::Builder::from_preset(universe::Preset::Basic)
            .number_of_boids(2)
            .grid_size(10.0)
            .attraction_weighting(0)
            .alignment_weighting(1)
            .separation_weighting(0)
            .alignment_radius(1.0)
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b3, b4].into(),
            }))
            .build();
        universe.tick();
        let boids = universe.get_boids();
        let Vec2(x1, y1) = boids.first().unwrap().velocity;
        let Vec2(x2, y2) = boids.last().unwrap().velocity;
        assert!(x1 == 0.0 && y1 < 1.0);
        assert!(x2 == 0.0 && y2 > 0.0);
    }

    #[test]
    fn boids_wrapping_are_aligned() {
        let b1 = Boid {
            position: Vec2(5.0, 9.0),
            velocity: Vec2(-2.0, 3.0),
            acceleration: Vec2(0.0, 0.0),
        };
        let b2 = Boid {
            position: Vec2(5.0, 1.0),
            velocity: Vec2(1.0, 0.5),
            acceleration: Vec2(0.0, 0.0),
        };
        let mut universe = universe::Builder::from_preset(universe::Preset::Basic)
            .number_of_boids(2)
            .grid_size(10.0)
            .attraction_weighting(0)
            .alignment_weighting(1)
            .separation_weighting(0)
            .alignment_radius(2.0)
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b1, b2].into(),
            }))
            .build();
        universe.tick();
        let boids = universe.get_boids();
        let Vec2(x1, y1) = boids.first().unwrap().velocity;
        let Vec2(x2, y2) = boids.last().unwrap().velocity;
        println!("({},{}) ({},{})", x1, y1, x2, y2);
        assert!(x1 > -2.0);
        assert!(y1 < 3.0);
        assert!(x2 < 1.0);
        assert!(y2 > 0.5);
    }

    #[test]
    fn boids_next_to_each_other_separate() {
        let b1 = Boid {
            position: Vec2(1.0, 1.0),
            velocity: Vec2(0.0, 0.0),
            acceleration: Vec2(0.0, 0.0),
        };
        let b2 = Boid {
            position: Vec2(2.0, 1.0),
            velocity: Vec2(0.0, 0.0),
            acceleration: Vec2(0.0, 0.0),
        };
        let mut universe = universe::Builder::from_preset(universe::Preset::Basic)
            .number_of_boids(2)
            .grid_size(10.0)
            .attraction_weighting(0)
            .alignment_weighting(0)
            .separation_weighting(1)
            .separation_radius(1.0)
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b1, b2].into(),
            }))
            .build();
        universe.tick();
        let boids = universe.get_boids();
        let (x1, _) = boids.first().unwrap().xy();
        let (x2, _) = boids.last().unwrap().xy();
        assert!(x1 < 1.0);
        assert!(x2 > 2.0);
    }
}
