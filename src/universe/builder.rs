use wasm_bindgen::prelude::*;

use crate::{
    boid::Boid,
    boid_factory::{BlueNoiseBoidFactory, BoidFactory},
    grid::{Grid, NaiveGrid, TiledGrid},
};

use super::Universe;

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
    maximum_velocity: Option<f32>,
    boid_factory: Box<dyn BoidFactory>,
    naive: bool,
    multithreaded: bool,
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
                .separation_radius(1.0)
                .maximum_velocity(1.0),
            Preset::Maruyama => Builder::default()
                .number_of_boids(100)
                .density(600.0)
                .noise_fraction(0.0)
                .attraction_weighting(4)
                .alignment_weighting(30)
                .separation_weighting(1)
                .attraction_radius(0.05)
                .alignment_radius(0.05)
                .separation_radius(0.01)
                .maximum_velocity(0.05),
            Preset::Zhang => Builder::default()
                .number_of_boids(100)
                .grid_size(100.0)
                .noise_fraction(0.05)
                .attraction_weighting(10)
                .alignment_weighting(1)
                .separation_weighting(400)
                .attraction_radius(1.0)
                .alignment_radius(1.0)
                .separation_radius(1.0)
                .maximum_velocity(1.0),
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

    pub fn maximum_velocity(mut self, magnitude: f32) -> Self {
        self.maximum_velocity = Some(magnitude);
        self
    }

    pub fn naive(mut self, naive: bool) -> Self {
        self.naive = naive;
        self
    }

    pub fn multithreaded(mut self, multithreaded: bool) -> Self {
        self.multithreaded = multithreaded;
        self
    }

    pub fn build(mut self) -> Universe {
        let number_of_boids = self
            .number_of_boids
            .expect("Missing field: number_of_boids");
        let attraction_weighting = self
            .attraction_weighting
            .expect("Must provide attraction_weighting");
        let alignment_weighting = self
            .alignment_weighting
            .expect("Must provide alignment_weighting");
        let separation_weighting = self
            .separation_weighting
            .expect("Must provide separation_weighting");
        let mut total_weighting = attraction_weighting + alignment_weighting + separation_weighting;
        if total_weighting == 0 {
            total_weighting = 1;
        }

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

        let mut grid: Box<dyn Grid<Boid>> = if self.naive {
            Box::new(NaiveGrid::new((number_of_boids as f32 / density).sqrt()))
        } else {
            Box::new(TiledGrid::new((number_of_boids as f32 / density).sqrt()))
        };
        grid.set_points(self.boid_factory.create_n(&*grid, number_of_boids));
        Universe {
            noise_rng: rand::rng(),
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
            maximum_velocity: self
                .maximum_velocity
                .expect("Must provide a maximum_velocity"),
            boid_factory: self.boid_factory,
            grid,
            multithreaded: self.multithreaded,
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
            maximum_velocity: None,
            naive: false,
            multithreaded: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Universe,
        boid::{Boid, Vec2},
        universe::{self, tests::TestBoidFactory},
    };

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
}
