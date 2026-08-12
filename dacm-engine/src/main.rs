//! Binary front-end for the DACM snapping engine: self-test + throughput bench.

use std::env;
use std::time::Instant;

use dacm_engine::{snap_discrete_vertex, snap_once, SnapConfig, CATEGORICAL_GROUPS, FEATURE_DIM};

fn main() {
    let args: Vec<String> = env::args().collect();
    let bench = args
        .iter()
        .position(|a| a == "--bench")
        .and_then(|i| args.get(i + 1))
        .and_then(|n| n.parse::<usize>().ok())
        .unwrap_or(1_000_000);

    let orig = sample_orig();
    let mut adv = orig.clone();
    // Simulate a hostile gradient: blow out continuous cols, smear one-hots.
    for v in adv.iter_mut() {
        *v = (*v + 0.7).min(1.0);
    }

    let cfg = SnapConfig {
        epsilon: 0.15,
        ..SnapConfig::default()
    };

    println!("=== DACM self-test ===");
    let snapped = snap_once(&adv, &orig, cfg);
    println!("orig     : {:?}", &orig[..6]);
    println!("adv      : {:?}", &adv[..6]);
    println!("snapped  : {:?}", &snapped[..6]);
    println!("proto grp: {:?}", &snapped[4..7]);
    println!("svc  grp : {:?}", snap_discrete_vertex(&snapped[7..18]));
    println!("features : {FEATURE_DIM} | groups: {:?}", CATEGORICAL_GROUPS);

    println!("\n=== DACM benchmark ({bench} snap passes) ===");
    let start = Instant::now();
    let mut sink = 0u32;
    for _ in 0..bench {
        let out = snap_once(&adv, &orig, cfg);
        sink = sink.wrapping_add(out.iter().map(|x| x.to_bits()).sum::<u32>());
    }
    let elapsed = start.elapsed();
    let ns = elapsed.as_nanos() as f64 / bench as f64;
    println!("elapsed : {elapsed:?}");
    println!("per-pass: {:.1} ns", ns);
    println!("throughput: {:.1} M passes/s", bench as f64 / elapsed.as_secs_f64() / 1e6);
    println!("sink    : {sink}");
}

fn sample_orig() -> Vec<f32> {
    let mut v = vec![0.5f32; FEATURE_DIM];
    v[0] = 0.2;
    v[1] = 0.8;
    v[4] = 1.0;
    v[7] = 1.0;
    for i in (4..7).filter(|&i| i != 4) {
        v[i] = 0.0;
    }
    for i in (7..18).filter(|&i| i != 7) {
        v[i] = 0.0;
    }
    v
}
