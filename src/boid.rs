use crate::grid::Point;

#[derive(Clone)]
pub struct Boid {
    pub position: (f32, f32),
    pub velocity: (f32, f32),
    pub acceleration: (f32, f32),
}

impl Point for Boid {
    fn xy(&self) -> (f32, f32) {
        self.position
    }
}
