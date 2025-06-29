use std::collections::HashMap;

pub trait Point {
    fn xy(&self) -> (f32, f32);
    fn set_xy(&mut self, x: f32, y: f32);
}

pub trait Grid<T>: Send + Sync
where
    T: Point,
{
    fn prepare(&mut self, max_radius: f32);
    fn insert(&mut self, point: T);
    fn neighbors(&self, point: &dyn Point, radius: f32) -> Vec<&T>;
    fn get_points(&self) -> &[T];
    fn set_points(&mut self, points: Vec<T>);
    fn get_size(&self) -> f32;
    fn resize(&mut self, size: f32);
}

pub struct NaiveGrid<T>
where
    T: Point + Send + Sync,
{
    points: Vec<T>,
    /// The width/height of the square grid.
    size: f32,
}

impl<T> NaiveGrid<T>
where
    T: Point + Send + Sync,
{
    pub fn new(size: f32) -> Self {
        NaiveGrid {
            points: vec![],
            size,
        }
    }
}

impl<T> Grid<T> for NaiveGrid<T>
where
    T: Point + Clone + Send + Sync + 'static,
{
    fn prepare(&mut self, _max_radius: f32) {}

    fn insert(&mut self, point: T) {
        let (x, y) = point.xy();
        if x < 0.0 || y < 0.0 || x > self.size || y > self.size {
            panic!("Cannot insert ({x},{y}) into grid with size {}", self.size);
        }
        self.points.push(point);
    }

    fn neighbors(&self, point: &dyn Point, radius: f32) -> Vec<&T> {
        let (ax, ay) = point.xy();
        let mut found_self = false;
        self.points
            .iter()
            .filter(|b| {
                if b.xy() == (ax, ay) && !found_self {
                    found_self = true;
                    return false;
                }
                let (bx, by) = b.xy();
                let dx = f32::min((bx - ax).abs(), self.size - (bx - ax).abs());
                let dy = f32::min((by - ay).abs(), self.size - (by - ay).abs());

                dx.powi(2) + dy.powi(2) <= radius.powi(2)
            })
            .collect()
    }

    fn get_points(&self) -> &[T] {
        &self.points
    }

    fn set_points(&mut self, points: Vec<T>) {
        self.points = points;
    }

    fn get_size(&self) -> f32 {
        self.size
    }

    fn resize(&mut self, size: f32) {
        let resize_factor = size / self.size;
        self.points.iter_mut().for_each(|p| {
            let (x, y) = p.xy();
            p.set_xy(x * resize_factor, y * resize_factor);
        });
        self.size = size;
    }
}

pub struct TiledGrid<T>
where
    T: Point + Send + Sync,
{
    points: Vec<T>,
    hash: HashMap<(u32, u32), Vec<usize>>,
    size: f32,
    tile_size: f32,
}

impl<T> TiledGrid<T>
where
    T: Point + Send + Sync,
{
    pub fn new(size: f32) -> Self {
        Self {
            points: vec![],
            hash: HashMap::new(),
            size,
            tile_size: 0.0,
        }
    }

    fn tile_coords(&self, point: &dyn Point) -> (u32, u32) {
        let (x, y) = point.xy();
        let tile_x = (x / self.tile_size).floor() as u32;
        let tile_y = (y / self.tile_size).floor() as u32;
        (tile_x, tile_y)
    }
}

impl<T> Grid<T> for TiledGrid<T>
where
    T: Point + Clone + Send + Sync + 'static,
{
    fn prepare(&mut self, max_radius: f32) {
        if max_radius > self.tile_size {
            self.tile_size = max_radius;
            self.set_points(self.points.to_vec());
        }
    }

    fn insert(&mut self, point: T) {
        let (x, y) = point.xy();
        if x < 0.0 || y < 0.0 || x > self.size || y > self.size {
            panic!("Cannot insert ({x},{y}) into grid with size {}", self.size);
        }

        if self.tile_size > 0.0 {
            self.hash
                .entry(self.tile_coords(&point))
                .or_default()
                .push(self.points.len());
        }

        self.points.push(point);
    }

    fn neighbors(&self, point: &dyn Point, radius: f32) -> Vec<&T> {
        let (x, y) = self.tile_coords(point);
        let mut tiled_points = vec![];
        let mut seen = vec![];

        let tiles_per_axis = (self.size / self.tile_size).ceil().max(1.0) as i32;
        for dx in -1..=1 {
            for dy in -1..=1 {
                let raw_other_tile_x = x as i32 + dx;
                let other_tile_x = (raw_other_tile_x + tiles_per_axis) % tiles_per_axis;
                let raw_other_tile_y = y as i32 + dy;
                let other_tile_y = (raw_other_tile_y + tiles_per_axis) % tiles_per_axis;

                // Final column can be narrower than self.tile_size so we need to also check 1 left
                // of it.
                let mut tiles_to_check = vec![(other_tile_x as u32, other_tile_y as u32)];
                if other_tile_x >= tiles_per_axis - 1 {
                    tiles_to_check.push(((other_tile_x - 1).max(0) as u32, other_tile_y as u32));
                    tiles_to_check.push((0, other_tile_y as u32));
                }

                for tile in tiles_to_check {
                    if let Some(indices) = self.hash.get(&tile) {
                        let (current_x, current_y) = tile;
                        // Sometimes if the radius is very large, a neighboring tiles can be calculated
                        // to be the same.
                        if !seen.contains(&(current_x, current_y)) {
                            for &idx in indices {
                                if let Some(point) = self.points.get(idx) {
                                    tiled_points.push(point);
                                }
                            }
                            seen.push((current_x, current_y));
                        }
                    }
                }
            }
        }

        let (ax, ay) = point.xy();
        let mut found_self = false;
        tiled_points
            .iter()
            .cloned()
            .filter(|b| {
                if b.xy() == (ax, ay) && !found_self {
                    found_self = true;
                    return false;
                }
                let (bx, by) = b.xy();
                let dx = f32::min((bx - ax).abs(), self.size - (bx - ax).abs());
                let dy = f32::min((by - ay).abs(), self.size - (by - ay).abs());

                dx.powi(2) + dy.powi(2) <= radius.powi(2)
            })
            .collect()
    }

    fn get_points(&self) -> &[T] {
        &self.points
    }

    fn set_points(&mut self, points: Vec<T>) {
        self.points.clear();
        self.hash.clear();
        for point in points {
            self.insert(point);
        }
    }

    fn get_size(&self) -> f32 {
        self.size
    }

    fn resize(&mut self, size: f32) {
        let resize_factor = size / self.size;
        self.points.iter_mut().for_each(|p| {
            let (x, y) = p.xy();
            p.set_xy(x * resize_factor, y * resize_factor);
        });
        self.size = size;
    }
}

#[cfg(test)]
mod tests {
    use crate::grid::*;
    use test_case::test_case;

    #[derive(Debug, PartialEq, Clone)]
    struct TestPoint(f32, f32);

    impl Point for TestPoint {
        fn xy(&self) -> (f32, f32) {
            (self.0, self.1)
        }
        fn set_xy(&mut self, x: f32, y: f32) {
            self.0 = x;
            self.1 = y;
        }
    }

    #[test_case(Box::new(NaiveGrid::new(100.0)) ; "NaiveGrid")]
    #[test_case(Box::new(TiledGrid::new(100.0)) ; "TiledGrid")]
    fn detects_neighbors(mut grid: Box<dyn Grid<TestPoint>>) {
        let p1 = TestPoint(0.0, 0.0);
        let p2 = TestPoint(1.0, 0.0);
        grid.insert(p1.clone());
        grid.insert(p2.clone());
        grid.prepare(1.0);
        assert_eq!(1, grid.neighbors(&p1, 1.0).len());
        assert_eq!(vec![&p2], grid.neighbors(&p1, 1.0));
        assert_eq!(vec![&p1], grid.neighbors(&p2, 1.0));

        let p3 = TestPoint(0.0, 1.0);
        let p4 = TestPoint(0.0, 2.0);
        grid.set_points(vec![p3.clone(), p4.clone()]);
        grid.prepare(1.3);
        assert_eq!(vec![&p4], grid.neighbors(&p3, 1.2));
        assert_eq!(vec![&p3], grid.neighbors(&p4, 1.3));

        let p5 = TestPoint(12.5, 12.5);
        let p6 = TestPoint(13.0, 13.0);
        grid.set_points(vec![p5.clone(), p6.clone()]);
        grid.prepare(1.2);
        assert_eq!(vec![&p6], grid.neighbors(&p5, 0.9));
        assert_eq!(vec![&p5], grid.neighbors(&p6, 1.2));

        let p7 = TestPoint(50.0, 50.0);
        grid.set_points(vec![
            p1.clone(),
            p2.clone(),
            p3.clone(),
            p4.clone(),
            p5.clone(),
            p6.clone(),
            p7.clone(),
        ]);
        grid.prepare(200.0);
        assert_eq!(
            1,
            grid.neighbors(&p1, 1.7)
                .iter()
                .filter(|&&n| n == &p2)
                .count()
        );
        assert_eq!(
            1,
            grid.neighbors(&p1, 1.05)
                .iter()
                .filter(|&&n| n == &p3)
                .count()
        );
        assert_eq!(vec![&p1, &p3, &p4], grid.neighbors(&p2, 3.0));
        assert!(grid.neighbors(&p7, 30.0).is_empty());
        assert_eq!(
            vec![&p1, &p2, &p3, &p4, &p6, &p7],
            grid.neighbors(&p5, 200.0)
        );
    }

    #[test_case(Box::new(NaiveGrid::new(10.0)) ; "NaiveGrid")]
    #[test_case(Box::new(TiledGrid::new(10.0)) ; "TiledGrid")]
    fn detects_neighbors_wrapping_around_the_grid(mut grid: Box<dyn Grid<TestPoint>>) {
        // Left/right wrapping
        let p1 = TestPoint(1.0, 5.0);
        let p2 = TestPoint(9.0, 5.0);
        grid.insert(p1.clone());
        grid.insert(p2.clone());
        grid.prepare(2.3);
        assert_eq!(vec![&p2], grid.neighbors(&p1, 2.1));
        assert_eq!(vec![&p1], grid.neighbors(&p2, 2.3));

        // Top/bottom wrapping
        let p3 = TestPoint(5.0, 9.0);
        let p4 = TestPoint(5.0, 1.0);
        grid.set_points(vec![p3.clone(), p4.clone()]);
        grid.prepare(2.7);
        assert_eq!(vec![&p4], grid.neighbors(&p3, 2.7));
        assert_eq!(vec![&p3], grid.neighbors(&p4, 2.0));
    }

    #[test_case(Box::new(NaiveGrid::new(10.0)) ; "NaiveGrid")]
    #[test_case(Box::new(TiledGrid::new(10.0)) ; "TiledGrid")]
    fn detects_neighbor_of_duplicated_point(mut grid: Box<dyn Grid<TestPoint>>) {
        let p1 = TestPoint(2.0, 2.0);
        let p2 = TestPoint(2.0, 2.0);
        grid.insert(p1.clone());
        grid.insert(p1.clone());
        grid.prepare(1.6);
        assert_eq!(vec![&p2], grid.neighbors(&p1, 1.2));
        assert_eq!(vec![&p1], grid.neighbors(&p2, 1.6));

        let p3 = TestPoint(0.0, 0.0);
        let p4 = TestPoint(0.0, 0.0);
        grid.set_points(vec![p3.clone(), p4.clone()]);
        grid.prepare(1.9);
        assert_eq!(vec![&p4], grid.neighbors(&p3, 1.9));
        assert_eq!(vec![&p3], grid.neighbors(&p4, 1.0));
    }

    #[test_case(Box::new(NaiveGrid::new(1.0)) ; "NaiveGrid")]
    #[test_case(Box::new(TiledGrid::new(1.0)) ; "TiledGrid")]
    #[should_panic]
    fn cannot_insert_point_outside_of_grid(mut grid: Box<dyn Grid<TestPoint>>) {
        grid.insert(TestPoint(2.0, 2.0));
    }

    #[test_case(Box::new(NaiveGrid::new(1.0)) ; "NaiveGrid")]
    #[test_case(Box::new(TiledGrid::new(1.0)) ; "TiledGrid")]
    #[should_panic]
    fn cannot_insert_negative_point(mut grid: Box<dyn Grid<TestPoint>>) {
        grid.insert(TestPoint(-1.0, -1.0));
    }

    #[test_case(Box::new(NaiveGrid::new(10.0)) ; "NaiveGrid")]
    #[test_case(Box::new(TiledGrid::new(10.0)) ; "TiledGrid")]
    fn resizing_maintains_point_positions_relative_to_size(mut grid: Box<dyn Grid<TestPoint>>) {
        grid.insert(TestPoint(0.0, 0.0));
        grid.insert(TestPoint(0.0, 10.0));
        grid.insert(TestPoint(10.0, 0.0));
        grid.insert(TestPoint(10.0, 10.0));
        grid.insert(TestPoint(5.0, 5.0));
        grid.insert(TestPoint(2.7, 3.2));
        grid.resize(100.0);
        assert_eq!(
            vec![
                TestPoint(0.0, 0.0),
                TestPoint(0.0, 100.0),
                TestPoint(100.0, 0.0),
                TestPoint(100.0, 100.0),
                TestPoint(50.0, 50.0),
                TestPoint(27.0, 32.0)
            ],
            grid.get_points()
        );
        grid.resize(50.0);
        assert_eq!(
            vec![
                TestPoint(0.0, 0.0),
                TestPoint(0.0, 50.0),
                TestPoint(50.0, 0.0),
                TestPoint(50.0, 50.0),
                TestPoint(25.0, 25.0),
                TestPoint(13.5, 16.0)
            ],
            grid.get_points()
        );
    }
}
