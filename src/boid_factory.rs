use crate::{
    bluenoise::{BlueNoise, Sample},
    boid::Boid,
    grid::{Grid, NaiveGrid, Point},
};

pub trait BoidFactory {
    fn create_n(&mut self, grid: &dyn Grid<Boid>, number_of_boids: u32) -> Vec<Boid>;
}

pub struct BlueNoiseBoidFactory {
    noise: BlueNoise,
}

impl BlueNoiseBoidFactory {
    pub fn new() -> Self {
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
