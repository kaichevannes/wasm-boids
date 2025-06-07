pub trait Positioned {
    fn position(&self) -> (f32, f32);
}

trait Grid<T>
where
    T: Positioned + Clone,
{
    fn new(grid_size: f32) -> Self;
    fn insert(&mut self, point: &T);
    fn neighbors(&mut self, point: &T, radius: f32) -> Vec<T>;
    fn set_points(&mut self, points: Vec<&T>);
    fn set_size(&mut self, size: f32);
}

pub struct NaiveGrid<T>
where
    T: Positioned + Clone,
{
    points: Vec<T>,
    /// The width/height of the square grid.
    grid_size: f32,
}

impl<T> Grid<T> for NaiveGrid<T>
where
    T: Positioned + Clone,
{
    fn new(grid_size: f32) -> Self {
        NaiveGrid {
            points: vec![],
            grid_size,
        }
    }

    fn insert(&mut self, point: &T) {
        self.points.push(point.clone());
    }

    fn neighbors(&mut self, point: &T, radius: f32) -> Vec<T> {
        // point (the function arg) is vector a, every other point in self.points is vector b
        // a is a neighbor of b if (bx - ax)^2 + (by - ay)^2 <= radius^2
        // i.e. the distance between them is less than the radius
        //
        // this works because sqrt((bx - ax)^2 + (by - ay)^2) <= radius all squared is the
        // expression above, saving a sqrt operation.
        let (ax, ay) = point.position();
        self.points
            .iter()
            .filter(|b| {
                if b.position() == (ax, ay) {
                    return false;
                }
                let (bx, by) = b.position();
                (bx - ax).powi(2) + (by - ay).powi(2) <= radius.powi(2)
            })
            .cloned()
            .collect()
    }

    fn set_points(&mut self, points: Vec<&T>) {
        self.points.clear();
        points.iter().for_each(|p| self.insert(p));
    }

    fn set_size(&mut self, size: f32) {
        self.grid_size = size;
    }
}

#[cfg(test)]
mod tests {
    use crate::grid::*;

    #[derive(Debug, PartialEq, Clone)]
    struct Point(f32, f32);

    impl Positioned for Point {
        fn position(&self) -> (f32, f32) {
            (self.0, self.1)
        }
    }

    #[test]
    fn detects_neighbors() {
        let mut grid = NaiveGrid::new(100.0);
        let p1 = Point(0.0, 0.0);
        let p2 = Point(1.0, 0.0);
        grid.insert(&p1);
        grid.insert(&p2);
        assert_eq!(1, grid.neighbors(&p1, 1.0).len());
        assert_eq!(&p2, grid.neighbors(&p1, 1.0).first().unwrap());
        assert_eq!(&p1, grid.neighbors(&p2, 1.0).first().unwrap());

        let p3 = Point(0.0, 1.0);
        let p4 = Point(0.0, 2.0);
        grid.set_points(vec![&p3, &p4]);
        assert_eq!(&p4, grid.neighbors(&p3, 1.0).first().unwrap());
        assert_eq!(&p3, grid.neighbors(&p4, 1.0).first().unwrap());

        let p5 = Point(12.5, 12.5);
        let p6 = Point(13.0, 13.0);
        grid.set_points(vec![&p5, &p6]);
        assert_eq!(&p6, grid.neighbors(&p5, 1.0).first().unwrap());
        assert_eq!(&p5, grid.neighbors(&p6, 1.0).first().unwrap());

        let p7 = Point(100.0, 100.0);
        grid.set_points(vec![&p1, &p2, &p3, &p4, &p5, &p6, &p7]);
        assert_eq!(vec![p2.clone(), p3.clone()], grid.neighbors(&p1, 1.0));
        assert_eq!(
            vec![p1.clone(), p3.clone(), p4.clone()],
            grid.neighbors(&p2, 3.0)
        );
        assert_eq!(vec![] as Vec<Point>, grid.neighbors(&p7, 50.0))
    }
}
