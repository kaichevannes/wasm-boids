pub struct BlueNoise;

impl BlueNoise {
    pub fn new() -> Self {
        Self
    }
}

impl Iterator for BlueNoise {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        Some(0.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::bluenoise::*;

    #[test]
    fn generates_numbers() {
        let mut noise = BlueNoise::new();
        assert_eq!(1, noise.by_ref().take(1).collect::<Vec<_>>().len());
        assert_eq!(2, noise.by_ref().take(2).collect::<Vec<_>>().len());
        assert_eq!(10, noise.by_ref().take(10).collect::<Vec<_>>().len());
        assert_eq!(100, noise.by_ref().take(100).collect::<Vec<_>>().len());
        assert_eq!(1000, noise.by_ref().take(1000).collect::<Vec<_>>().len());
    }
}
