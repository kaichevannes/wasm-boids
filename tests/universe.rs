use wasm_boids::*;

#[test]
fn universe_has_expected_number_of_boids() {
    let universe_with_no_boids = universe::Builder::from_preset(universe::Preset::Basic)
        .boid_count(0)
        .build();
    assert!(universe_with_no_boids.get_boids().is_empty());

    let universe_with_single_boid = universe::Builder::from_preset(universe::Preset::Basic)
        .boid_count(1)
        .build();
    assert_eq!(1, universe_with_single_boid.get_boids().len());

    let universe_with_one_hundred_boids = universe::Builder::from_preset(universe::Preset::Basic)
        .boid_count(100)
        .build();
    assert_eq!(100, universe_with_one_hundred_boids.get_boids().len());
}

#[test]
fn universe_tick_moves_boids() {
    let mut universe = Universe::build_from_preset(universe::Preset::Basic);
    let original_positions: Vec<(f32, f32)> = universe
        .get_boids()
        .iter()
        .map(|boid| boid.position)
        .collect();
    universe.tick();
    let updated_positions: Vec<(f32, f32)> = universe
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
