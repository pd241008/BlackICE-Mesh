//! BlackICE-Mesh | dacm-engine
//!
//! High-performance, memory-safe implementation of **Discrete Adversarial
//! Constraint Mapping (DACM)**. Ported from the PyTorch `pgd_attack` logic
//! (`ml-optimizer/app/ml/attacks/pgd.py`) into zero-`unsafe` Rust.
//!
//! DACM reconciles two competing constraints when an adversarial gradient step
//! is applied to tabular telemetry:
//!
//! 1. **Continuous columns** are clamped to the normalized Min-Max domain
//!    `[0, 1]` and projected onto the `L_inf` ball of radius `epsilon`.
//! 2. **Categorical columns** (one-hot) are snapped onto the nearest valid
//!    vertex of the discrete simplex via Euclidean distance minimization —
//!    which, on a one-hot basis, reduces to an `argmax` — keeping the payload
//!    structurally parseable by real network parsers.
#![forbid(unsafe_code)]

use std::ops::Range;

/// Feature topology of the NSL-KDD tabular vector (18 features).
pub const FEATURE_DIM: usize = 18;
/// Indices of continuous columns.
pub const CONTINUOUS_COLS: Range<usize> = 0..4;
/// One-hot categorical groups: protocol (3) and service (11).
pub const CATEGORICAL_GROUPS: &[Range<usize>] = &[4..7, 7..18];

/// Tuning knobs for a single DACM snap pass.
#[derive(Debug, Clone, Copy)]
pub struct SnapConfig {
    /// Max `L_inf` perturbation budget.
    pub epsilon: f32,
    /// Min-Max lower bound for continuous columns.
    pub min: f32,
    /// Min-Max upper bound for continuous columns.
    pub max: f32,
}

impl Default for SnapConfig {
    fn default() -> Self {
        Self {
            epsilon: 0.1,
            min: 0.0,
            max: 1.0,
        }
    }
}

/// Euclidean nearest neighbor in one-hot space: `argmax` collapses to the
/// vertex carrying the largest magnitude.
#[inline]
pub fn snap_discrete_vertex(group: &[f32]) -> Vec<f32> {
    let n = group.len();
    let (mut best, mut best_idx) = (f32::MIN, 0);
    for (i, &v) in group.iter().enumerate() {
        if v > best {
            best = v;
            best_idx = i;
        }
    }
    let mut one_hot = vec![0.0f32; n];
    one_hot[best_idx] = 1.0;
    one_hot
}

/// Clamp a single continuous value into `[min, max]`.
#[inline]
pub fn clamp_min_max(v: f32, min: f32, max: f32) -> f32 {
    v.max(min).min(max)
}

/// Project `adv` onto the `L_inf` ball of radius `epsilon` centered on `orig`,
/// then clamp the result to the Min-Max domain. This mirrors the continuous
/// branch of `pgd_attack`.#[inline]
pub fn snap_continuous(adv: &[f32], orig: &[f32], cfg: SnapConfig) -> Vec<f32> {
    adv.iter()
        .zip(orig.iter())
        .map(|(&a, &o)| {
            let eta = (a - o).clamp(-cfg.epsilon, cfg.epsilon);
            clamp_min_max(o + eta, cfg.min, cfg.max)
        })
        .collect()
}

/// Apply one full DACM pass over a perturbed feature vector, snapping every
/// group back to its structurally valid domain. `adv` is expected to already
/// carry the model-gradient step; the engine only enforces constraints.
pub fn snap_once(adv: &[f32], orig: &[f32], cfg: SnapConfig) -> Vec<f32> {
    debug_assert_eq!(adv.len(), orig.len());
    let mut out = adv.to_vec();

    for i in CONTINUOUS_COLS {
        out[i] = snap_continuous(&adv[i..i + 1], &orig[i..i + 1], cfg)[0];
    }

    for group in CATEGORICAL_GROUPS {
        let snapped = snap_discrete_vertex(&adv[group.clone()]);
        for (slot, value) in out[group.clone()].iter_mut().zip(snapped) {
            *slot = value;
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_orig() -> Vec<f32> {
        // 4 continuous + 3 protocol + 11 service one-hot (valid vertices)
        let mut v = vec![0.5f32; FEATURE_DIM];
        v[0] = 0.2;
        v[1] = 0.8;
        v[4] = 1.0; // protocol = tcp
        v[7] = 1.0; // service = http
        for i in (4..7).filter(|&i| i != 4) {
            v[i] = 0.0;
        }
        for i in (7..18).filter(|&i| i != 7) {
            v[i] = 0.0;
        }
        v
    }

    #[test]
    fn continuous_stays_in_minmax_ball() {
        let orig = sample_orig();
        let mut adv = orig.clone();
        adv[0] = 10.0; // wild perturbation
        let cfg = SnapConfig { epsilon: 0.1, ..SnapConfig::default() };
        let snapped = snap_once(&adv, &orig, cfg);
        assert!((0.0..=1.0).contains(&snapped[0]));
        assert!((snapped[0] - orig[0]).abs() <= cfg.epsilon + 1e-6);
    }

    #[test]
    fn categorical_groups_are_valid_vertices() {
        let orig = sample_orig();
        let mut adv = orig.clone();
        adv[4..7].copy_from_slice(&[0.4, 0.9, 0.1]); // lean toward udp
        adv[7..18].copy_from_slice(&[0.1; 11]);
        let snapped = snap_once(&adv, &orig, SnapConfig::default());
        assert!(snapped[4..7].iter().sum::<f32>() - 1.0 < 1e-6);
        assert!(snapped[4..7].iter().all(|&v| v == 0.0 || v == 1.0));
        assert!(snapped[7..18].iter().all(|&v| v == 0.0 || v == 1.0));
    }

    #[test]
    fn identity_preserved_on_clean_input() {
        let orig = sample_orig();
        let snapped = snap_once(&orig, &orig, SnapConfig::default());
        assert_eq!(snapped, orig);
    }
}
