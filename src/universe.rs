pub mod builder;

use crate::{
    boid::{Boid, Vec2},
    boid_factory::BoidFactory,
    grid::Grid,
};
use builder::{Builder, Preset};
use wasm_bindgen::prelude::*;

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
            let mut position = boid.position + velocity;

            let grid_size = self.grid.get_size();
            if position.0 < 0.0 {
                position.0 += grid_size;
            } else if position.0 > grid_size {
                position.0 -= grid_size;
            }

            if position.1 < 0.0 {
                position.1 += grid_size;
            } else if position.1 > grid_size {
                position.1 -= grid_size;
            }

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
        let (x1, y1) = starting.into();
        let (x2, y2) = other.into();

        let adjusted_axis = |n1: f32, n2: f32| {
            let grid_size = self.grid.get_size();
            if n2 - n1 > grid_size / 2.0 {
                return n2 - grid_size;
            } else if n2 - x1 < -grid_size / 2.0 {
                return n2 + grid_size;
            }
            n2
        };

        Vec2(adjusted_axis(x1, x2), adjusted_axis(y1, y2))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use crate::boid::Vec2;
    use crate::universe::BoidFactory;
    use crate::{boid::Boid, *};

    pub struct TestBoidFactory {
        pub boids: VecDeque<Boid>,
    }

    fn create_boid_with_position(position: Vec2) -> Boid {
        Boid {
            position,
            velocity: Vec2(0.0, 0.0),
            acceleration: Vec2(0.0, 0.0),
        }
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
    fn boids_next_to_each_other_are_attracted() {
        let universe_builder_closure = || {
            universe::Builder::from_preset(universe::Preset::Basic)
                .number_of_boids(2)
                .grid_size(10.0)
                .attraction_weighting(1)
                .alignment_weighting(0)
                .separation_weighting(0)
                .attraction_radius(1.0)
        };

        let b1 = create_boid_with_position(Vec2(1.0, 1.0));
        let b2 = create_boid_with_position(Vec2(2.0, 1.0));
        let mut universe = universe_builder_closure()
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b1, b2].into(),
            }))
            .build();
        universe.tick();
        let Vec2(x1, _) = universe.get_boids()[0].position;
        let Vec2(x2, _) = universe.get_boids()[1].position;
        assert!(x1 > 1.0);
        assert!(x2 < 2.0);

        let b3 = create_boid_with_position(Vec2(6.0, 5.0));
        let b4 = create_boid_with_position(Vec2(6.0, 4.0));
        let mut universe = universe_builder_closure()
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b3, b4].into(),
            }))
            .build();
        universe.tick();
        let Vec2(_, y1) = universe.get_boids()[0].position;
        let Vec2(_, y2) = universe.get_boids()[1].position;
        assert!(y1 < 5.0);
        assert!(y2 > 4.0);
    }

    #[test]
    fn boids_wrapping_are_attracted() {
        let universe_builder_closure = || {
            universe::Builder::from_preset(universe::Preset::Basic)
                .number_of_boids(2)
                .grid_size(10.0)
                .attraction_weighting(1)
                .alignment_weighting(0)
                .separation_weighting(0)
                .attraction_radius(2.0)
        };

        let b1 = create_boid_with_position(Vec2(5.0, 9.0));
        let b2 = create_boid_with_position(Vec2(5.0, 1.0));
        let mut universe = universe_builder_closure()
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b1, b2].into(),
            }))
            .build();
        universe.tick();
        let Vec2(_, y1) = universe.get_boids()[0].position;
        let Vec2(_, y2) = universe.get_boids()[1].position;
        assert!(y1 > 9.0);
        assert!(y2 < 1.0);

        let b3 = create_boid_with_position(Vec2(1.0, 5.0));
        let b4 = create_boid_with_position(Vec2(9.0, 5.0));
        let mut universe = universe_builder_closure()
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b3, b4].into(),
            }))
            .build();
        universe.tick();
        let Vec2(x1, _) = universe.get_boids()[0].position;
        let Vec2(x2, _) = universe.get_boids()[1].position;
        assert!(x1 < 1.0);
        assert!(x2 > 9.0);

        let b5 = create_boid_with_position(Vec2(1.0, 9.0));
        let b6 = create_boid_with_position(Vec2(9.0, 1.0));
        let mut universe = universe_builder_closure()
            .attraction_radius(3.0)
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b5, b6].into(),
            }))
            .build();
        universe.tick();
        let Vec2(x1, y1) = universe.get_boids()[0].position;
        let Vec2(x2, y2) = universe.get_boids()[1].position;
        assert!(x1 < 1.0 && y1 > 9.0);
        assert!(x2 > 9.0 && y2 < 1.0);
    }

    #[test]
    fn boids_next_to_each_other_are_aligned() {
        let universe_builder_closure = || {
            universe::Builder::from_preset(universe::Preset::Basic)
                .number_of_boids(2)
                .grid_size(10.0)
                .attraction_weighting(0)
                .alignment_weighting(1)
                .separation_weighting(0)
                .alignment_radius(1.0)
        };

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
        let mut universe = universe_builder_closure()
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b1, b2].into(),
            }))
            .build();
        universe.tick();
        let Vec2(x1, y1) = universe.get_boids()[0].velocity;
        let Vec2(x2, y2) = universe.get_boids()[1].velocity;
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
        let mut universe = universe_builder_closure()
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b3, b4].into(),
            }))
            .build();
        universe.tick();
        let Vec2(x1, y1) = universe.get_boids()[0].velocity;
        let Vec2(x2, y2) = universe.get_boids()[1].velocity;
        assert!(x1 == 0.0 && y1 < 1.0);
        assert!(x2 == 0.0 && y2 > 0.0);
    }

    #[test]
    fn boids_wrapping_are_aligned() {
        let universe_builder_closure = || {
            universe::Builder::from_preset(universe::Preset::Basic)
                .number_of_boids(2)
                .grid_size(10.0)
                .attraction_weighting(0)
                .alignment_weighting(1)
                .separation_weighting(0)
                .alignment_radius(2.0)
        };

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
        let mut universe = universe_builder_closure()
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b1, b2].into(),
            }))
            .build();
        universe.tick();
        let Vec2(x1, y1) = universe.get_boids()[0].velocity;
        let Vec2(x2, y2) = universe.get_boids()[1].velocity;
        assert!(x1 > -2.0 && y1 < 3.0);
        assert!(x2 < 1.0 && y2 > 0.5);
    }

    #[test]
    fn boids_next_to_each_other_separate() {
        let universe_builder_closure = || {
            universe::Builder::from_preset(universe::Preset::Basic)
                .number_of_boids(2)
                .grid_size(10.0)
                .attraction_weighting(0)
                .alignment_weighting(0)
                .separation_weighting(1)
                .separation_radius(1.0)
        };

        let b1 = create_boid_with_position(Vec2(1.0, 1.0));
        let b2 = create_boid_with_position(Vec2(2.0, 1.0));
        let mut universe = universe_builder_closure()
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b1, b2].into(),
            }))
            .build();
        universe.tick();
        let Vec2(x1, _) = universe.get_boids()[0].position;
        let Vec2(x2, _) = universe.get_boids()[1].position;
        assert!(x1 < 1.0);
        assert!(x2 > 2.0);
    }

    #[test]
    fn boid_wraps_around_grid_when_moving() {
        let universe_builder_closure = || {
            universe::Builder::from_preset(universe::Preset::Basic)
                .number_of_boids(1)
                .grid_size(10.0)
        };
        let b1 = Boid {
            position: Vec2(0.0, 5.0),
            velocity: Vec2(-1.0, 0.0),
            acceleration: Vec2(0.0, 0.0),
        };
        let mut universe = universe_builder_closure()
            .boid_factory(Box::new(TestBoidFactory {
                boids: vec![b1].into(),
            }))
            .build();
        universe.tick();
        let Vec2(x1, y1) = universe.get_boids()[0].position;
        assert!(x1 > 9.0);
        assert!(y1 == 5.0);
    }
}
