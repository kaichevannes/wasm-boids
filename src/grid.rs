pub trait Point {
    fn xy(&self) -> (f32, f32);
}

pub trait Grid<'a, T>
where
    T: Point,
{
    fn insert<'b: 'a>(&mut self, point: &'b T);
    fn neighbors(&self, point: &dyn Point, radius: f32) -> Vec<&T>;
    fn get_points(&self) -> &[&T];
    fn set_points<'b: 'a>(&mut self, points: &[&'b T]);
    fn get_size(&self) -> f32;
}

pub struct NaiveGrid<'a, T>
where
    T: Point,
{
    points: Vec<&'a T>,
    /// The width/height of the square grid.
    grid_size: f32,
}

impl<'a, T> NaiveGrid<'a, T>
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

impl<'a, T> Grid<'a, T> for NaiveGrid<'a, T>
where
    T: Point,
{
    fn insert<'b: 'a>(&mut self, point: &'b T) {
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
            .copied()
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

    fn get_points(&self) -> &[&T] {
        &self.points
    }

    fn set_points<'b: 'a>(&mut self, points: &[&'b T]) {
        self.points.clear();
        points.iter().for_each(|p| self.insert(p));
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
    }

    #[test]
    fn detects_neighbors() {
        let mut grid = NaiveGrid::new(100.0);
        let p1 = TestPoint(0.0, 0.0);
        let p2 = TestPoint(1.0, 0.0);
        grid.insert(&p1);
        grid.insert(&p2);
        assert_eq!(1, grid.neighbors(&p1, 1.0).len());
        assert_eq!(vec![&p2], grid.neighbors(&p1, 1.0));
        assert_eq!(vec![&p1], grid.neighbors(&p2, 1.0));

        let p3 = TestPoint(0.0, 1.0);
        let p4 = TestPoint(0.0, 2.0);
        grid.set_points(&vec![&p3, &p4]);
        assert_eq!(vec![&p4], grid.neighbors(&p3, 1.0));
        assert_eq!(vec![&p3], grid.neighbors(&p4, 1.0));

        let p5 = TestPoint(12.5, 12.5);
        let p6 = TestPoint(13.0, 13.0);
        grid.set_points(&vec![&p5, &p6]);
        assert_eq!(vec![&p6], grid.neighbors(&p5, 1.0));
        assert_eq!(vec![&p5], grid.neighbors(&p6, 1.0));

        let p7 = TestPoint(50.0, 50.0);
        grid.set_points(&vec![&p1, &p2, &p3, &p4, &p5, &p6, &p7]);
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
        grid.insert(&p1);
        grid.insert(&p2);
        assert_eq!(vec![&p2], grid.neighbors(&p1, 2.0));
        assert_eq!(vec![&p1], grid.neighbors(&p2, 2.0));

        // Top/bottom wrapping
        let p3 = TestPoint(5.0, 9.0);
        let p4 = TestPoint(5.0, 1.0);
        grid.set_points(&vec![&p3, &p4]);
        assert_eq!(vec![&p4], grid.neighbors(&p3, 2.0));
        assert_eq!(vec![&p3], grid.neighbors(&p4, 2.0));
    }

    #[test]
    fn detects_neighbor_of_duplicated_point() {
        let mut grid = NaiveGrid::new(10.0);
        let p1 = TestPoint(2.0, 2.0);
        let p2 = TestPoint(2.0, 2.0);
        grid.insert(&p1);
        grid.insert(&p1);
        assert_eq!(vec![&p2], grid.neighbors(&p1, 1.0));
        assert_eq!(vec![&p1], grid.neighbors(&p2, 1.0));

        let p3 = TestPoint(0.0, 0.0);
        let p4 = TestPoint(0.0, 0.0);
        grid.set_points(&vec![&p3, &p4]);
        assert_eq!(vec![&p4], grid.neighbors(&p3, 1.0));
        assert_eq!(vec![&p3], grid.neighbors(&p4, 1.0));
    }

    #[test]
    #[should_panic]
    fn cannot_insert_point_outside_of_grid() {
        let mut grid = NaiveGrid::new(1.0);
        grid.insert(&TestPoint(2.0, 2.0));
    }

    #[test]
    #[should_panic]
    fn cannot_insert_negative_point() {
        let mut grid = NaiveGrid::new(1.0);
        grid.insert(&TestPoint(-1.0, -1.0));
    }
}
