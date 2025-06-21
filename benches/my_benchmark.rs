use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, PlotConfiguration};
use wasm_boids::universe::builder::{Builder, Preset};

enum Strategy {
    Naive,
    NaiveMultithreaded,
    Tiled,
    TiledMultithreaded,
}

impl std::fmt::Debug for Strategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Strategy::Naive => "naive",
            Strategy::NaiveMultithreaded => "naive_mt",
            Strategy::Tiled => "tiled",
            Strategy::TiledMultithreaded => "tiled_mt",
        })
    }
}

fn boids_per_thread(c: &mut Criterion) {
    let mut group = c.benchmark_group("boids_per_thread");

    let strategies = [Strategy::NaiveMultithreaded];

    let min_lengths = [
        1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 750,
        1000,
    ];

    for strategy in &strategies {
        for &min_length in &min_lengths {
            group.bench_with_input(
                BenchmarkId::new(format!("{:?}", strategy), min_length),
                &min_length,
                |b, &n| {
                    b.iter(|| {
                        let mut universe = Builder::from_preset(Preset::Basic)
                            .number_of_boids(1000)
                            .number_of_boids_per_thread(n)
                            .build();
                        for _ in 0..100 {
                            universe.tick();
                        }
                    });
                },
            );
        }
    }

    group.finish()
}

fn tick_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("tick_performance");

    group
        .plot_config(PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic));

    let strategies = [
        Strategy::Naive,
        Strategy::NaiveMultithreaded,
        Strategy::Tiled,
        Strategy::TiledMultithreaded,
    ];

    let boid_counts = [1, 10, 100, 1000, 10000];

    for strategy in &strategies {
        for &count in &boid_counts {
            group.throughput(criterion::Throughput::Elements(count));
            group.sample_size(10).bench_with_input(
                BenchmarkId::new(format!("{:?}", strategy), count),
                &count,
                |b, &n| {
                    b.iter(|| {
                        let universe_builder =
                            Builder::from_preset(Preset::Basic).number_of_boids(n as u32);
                        let mut universe = match strategy {
                            Strategy::Naive => {
                                universe_builder.multithreaded(false).naive(true).build()
                            }
                            Strategy::NaiveMultithreaded => {
                                universe_builder.multithreaded(true).naive(true).build()
                            }
                            Strategy::Tiled => {
                                universe_builder.multithreaded(false).naive(false).build()
                            }
                            Strategy::TiledMultithreaded => {
                                universe_builder.multithreaded(true).naive(false).build()
                            }
                        };
                        for _ in 0..1000 {
                            universe.tick();
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, boids_per_thread, tick_performance);
criterion_main!(benches);
