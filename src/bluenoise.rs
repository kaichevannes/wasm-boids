use std::f32::consts::PI;

use crate::grid::{Grid, Point};
use rand::prelude::*;

const NUMBER_OF_SAMPLES_UNTIL_REJECTION: u32 = 30;

#[derive(Debug, PartialEq, Clone)]
pub struct Sample(pub f32, pub f32);

impl Point for Sample {
    fn xy(&self) -> (f32, f32) {
        (self.0, self.1)
    }
    fn set_xy(&mut self, x: f32, y: f32) {
        self.0 = x;
        self.1 = y;
    }
}

/// [Fast Poisson Disk Sampling in Arbitrary Dimenisions, Bridson 2007](https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf)
pub struct BlueNoise {
    rng: ThreadRng,
}

impl BlueNoise {
    pub fn new() -> Self {
        Self { rng: rand::rng() }
    }

    pub fn generate<T>(
        &mut self,
        grid: &mut dyn Grid<T>,
        number_of_samples_to_generate: u32,
    ) -> Vec<Sample>
    where
        T: Point,
    {
        // Step 1, Bridson 2007
        // Existing list or random point to start
        let mut active_points: Vec<Sample> = grid
            .get_points()
            .iter()
            .map(|p| {
                let (x, y) = p.xy();
                Sample(x, y)
            })
            .collect();

        let mut samples_generated = Vec::<Sample>::new();
        if active_points.is_empty() {
            let initial_point = Sample(
                self.rng.random_range(0.0..=grid.get_size()),
                self.rng.random_range(0.0..=grid.get_size()),
            );
            active_points.push(initial_point.clone());
            samples_generated.push(initial_point);
        }

        // Step 2, Bridson 2007
        // The number of cells in a grid with n dimensions is r / sqrt(n).
        // We rearrange that to get radius = sqtr(n) * number_of_cells
        // let radius = number_of_samples_to_generate as f32 * f32::sqrt(2.0);
        let radius =
            (grid.get_size() * f32::sqrt(2.0)) / number_of_samples_to_generate.isqrt() as f32;
        'outer: while !active_points.is_empty() {
            if samples_generated.len() as u32 >= number_of_samples_to_generate {
                return samples_generated;
            }

            let idx = self.rng.random_range(0..active_points.len());
            let (active_x, active_y) = active_points.get(idx).unwrap().xy();

            for _ in 0..NUMBER_OF_SAMPLES_UNTIL_REJECTION {
                let r = self.rng.random_range(radius..=radius * 2.0);
                let angle: f32 = self.rng.random_range(0.0..2.0 * PI);
                // Convert to cartesian coordinates from polar coordinates.
                // The point must be constrained within the size of the grid.
                let candidate_point = Sample(
                    (active_x + r * angle.cos()).clamp(0.0, grid.get_size()),
                    (active_y + r * angle.sin()).clamp(0.0, grid.get_size()),
                );
                // Don't add a point that already exists, happens on the clamped values.
                if samples_generated.contains(&candidate_point) {
                    continue;
                }
                if grid.neighbors(&candidate_point, radius).is_empty() {
                    active_points.push(candidate_point.clone());
                    samples_generated.push(candidate_point);
                    continue 'outer;
                }
            }

            active_points.remove(idx);
        }

        panic!(
            "Couldn't generate requested number of points. Something probably went wrong in the grid_size logic."
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{bluenoise::*, grid::NaiveGrid};

    #[test]
    fn generates_numbers() {
        let mut noise = BlueNoise::new();
        let mut grid = NaiveGrid::<Sample>::new(100.0);
        assert_eq!(1, noise.generate(&mut grid, 1).len());
        assert_eq!(2, noise.generate(&mut grid, 2).len());
        assert_eq!(10, noise.generate(&mut grid, 10).len());
        assert_eq!(100, noise.generate(&mut grid, 100).len());
        assert_eq!(1000, noise.generate(&mut grid, 1000).len());
    }

    #[test]
    fn generated_numbers_are_different() {
        let mut noise = BlueNoise::new();
        let mut grid = NaiveGrid::<Sample>::new(100.0);
        let numbers = noise.generate(&mut grid, 100);
        assert!(numbers.iter().enumerate().all(|(i, n)| {
            numbers.iter().enumerate().all(|(j, m)| {
                let (x1, y1) = n.xy();
                let (x2, y2) = m.xy();
                i == j || (x1, y1) != (x2, y2)
            })
        }));
    }
}
