pub trait Point {
    fn xy(&self) -> (f32, f32);
    fn set_xy(&mut self, x: f32, y: f32);
}

pub trait Grid<T>
where
    T: Point,
{
    fn insert(&mut self, point: T);
    fn neighbors(&self, point: &dyn Point, radius: f32) -> Vec<&T>;
    fn get_points(&self) -> &[T];
    fn set_points(&mut self, points: Vec<T>);
    fn get_size(&self) -> f32;
    fn resize(&mut self, size: f32);
}

pub struct NaiveGrid<T>
where
    T: Point,
{
    points: Vec<T>,
    /// The width/height of the square grid.
    grid_size: f32,
}

impl<T> NaiveGrid<T>
where
    T: Point,
{
    pub fn new(grid_size: f32) -> Self {
        NaiveGrid {
            points: vec![],
            grid_size,
        }
    }
}

impl<T> Grid<T> for NaiveGrid<T>
where
    T: Point,
{
    fn insert(&mut self, point: T) {
        let (x, y) = point.xy();
        if x < 0.0 || y < 0.0 || x > self.grid_size || y > self.grid_size {
            panic!(
                "Cannot insert ({x},{y}) into grid with size {}",
                self.grid_size
            );
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
                let dx = f32::min((bx - ax).abs(), self.grid_size - (bx - ax).abs());
                let dy = f32::min((by - ay).abs(), self.grid_size - (by - ay).abs());

                dx.powi(2) + dy.powi(2) <= radius.powi(2)
            })
            .collect()
    }

    fn get_size(&self) -> f32 {
        self.grid_size
    }

    fn get_points(&self) -> &[T] {
        &self.points
    }

    fn set_points(&mut self, points: Vec<T>) {
        self.points = points;
    }

    fn resize(&mut self, size: f32) {
        let resize_factor = size / self.grid_size;
        self.points.iter_mut().for_each(|p| {
            let (x, y) = p.xy();
            p.set_xy(x * resize_factor, y * resize_factor);
        });
        self.grid_size = size;
    }
}

#[cfg(test)]
mod tests {
    use crate::grid::*;

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

    #[test]
    fn detects_neighbors() {
        let mut grid = NaiveGrid::new(100.0);
        let p1 = TestPoint(0.0, 0.0);
        let p2 = TestPoint(1.0, 0.0);
        grid.insert(p1.clone());
        grid.insert(p2.clone());
        assert_eq!(1, grid.neighbors(&p1, 1.0).len());
        assert_eq!(vec![&p2], grid.neighbors(&p1, 1.0));
        assert_eq!(vec![&p1], grid.neighbors(&p2, 1.0));

        let p3 = TestPoint(0.0, 1.0);
        let p4 = TestPoint(0.0, 2.0);
        grid.set_points(vec![p3.clone(), p4.clone()]);
        assert_eq!(vec![&p4], grid.neighbors(&p3, 1.0));
        assert_eq!(vec![&p3], grid.neighbors(&p4, 1.0));

        let p5 = TestPoint(12.5, 12.5);
        let p6 = TestPoint(13.0, 13.0);
        grid.set_points(vec![p5.clone(), p6.clone()]);
        assert_eq!(vec![&p6], grid.neighbors(&p5, 1.0));
        assert_eq!(vec![&p5], grid.neighbors(&p6, 1.0));

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
        assert_eq!(vec![&p2, &p3], grid.neighbors(&p1, 1.0));
        assert_eq!(vec![&p1, &p3, &p4], grid.neighbors(&p2, 3.0));
        assert!(grid.neighbors(&p7, 30.0).is_empty());
        assert_eq!(
            vec![&p1, &p2, &p3, &p4, &p6, &p7],
            grid.neighbors(&p5, 200.0)
        );
    }

    #[test]
    fn detects_neighbors_wrapping_around_the_grid() {
        let mut grid = NaiveGrid::new(10.0);
        // Left/right wrapping
        let p1 = TestPoint(1.0, 5.0);
        let p2 = TestPoint(9.0, 5.0);
        grid.insert(p1.clone());
        grid.insert(p2.clone());
        assert_eq!(vec![&p2], grid.neighbors(&p1, 2.0));
        assert_eq!(vec![&p1], grid.neighbors(&p2, 2.0));

        // Top/bottom wrapping
        let p3 = TestPoint(5.0, 9.0);
        let p4 = TestPoint(5.0, 1.0);
        grid.set_points(vec![p3.clone(), p4.clone()]);
        assert_eq!(vec![&p4], grid.neighbors(&p3, 2.0));
        assert_eq!(vec![&p3], grid.neighbors(&p4, 2.0));
    }

    #[test]
    fn detects_neighbor_of_duplicated_point() {
        let mut grid = NaiveGrid::new(10.0);
        let p1 = TestPoint(2.0, 2.0);
        let p2 = TestPoint(2.0, 2.0);
        grid.insert(p1.clone());
        grid.insert(p1.clone());
        assert_eq!(vec![&p2], grid.neighbors(&p1, 1.0));
        assert_eq!(vec![&p1], grid.neighbors(&p2, 1.0));

        let p3 = TestPoint(0.0, 0.0);
        let p4 = TestPoint(0.0, 0.0);
        grid.set_points(vec![p3.clone(), p4.clone()]);
        assert_eq!(vec![&p4], grid.neighbors(&p3, 1.0));
        assert_eq!(vec![&p3], grid.neighbors(&p4, 1.0));
    }

    #[test]
    #[should_panic]
    fn cannot_insert_point_outside_of_grid() {
        let mut grid = NaiveGrid::new(1.0);
        grid.insert(TestPoint(2.0, 2.0));
    }

    #[test]
    #[should_panic]
    fn cannot_insert_negative_point() {
        let mut grid = NaiveGrid::new(1.0);
        grid.insert(TestPoint(-1.0, -1.0));
    }

    #[test]
    fn resizing_maintains_point_positions_relative_to_size() {
        let mut grid = NaiveGrid::new(10.0);
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
