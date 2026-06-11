//! Opt-in decode phase timing.

use web_time::Instant;

use crate::types::{DecodeTelemetry, Measure};

/// Unix-epoch milliseconds (wall clock).
pub(crate) fn unix_time_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        js_sys::Date::now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
            * 1000.0
    }
}

struct MeasureCollector {
    epoch: Instant,
    measures: Vec<Measure>,
}

#[cfg(target_arch = "wasm32")]
mod storage {
    use super::{Measure, MeasureCollector};
    use std::sync::Mutex;
    use web_time::Instant;

    static COLLECTOR: Mutex<Option<MeasureCollector>> = Mutex::new(None);

    pub fn set(value: Option<MeasureCollector>) {
        if let Ok(mut collector) = COLLECTOR.lock() {
            *collector = value;
        }
    }

    pub fn epoch() -> Option<Instant> {
        COLLECTOR
            .lock()
            .ok()
            .and_then(|collector| collector.as_ref().map(|collector| collector.epoch))
    }

    pub fn push_measure(name: &'static str, start: Instant, epoch: Instant) {
        if let Ok(mut collector) = COLLECTOR.lock()
            && let Some(collector) = collector.as_mut()
        {
            collector.measures.push(Measure {
                name,
                start_ms: start.duration_since(epoch).as_secs_f64() * 1000.0,
                duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            });
        }
    }

    pub fn take() -> MeasureCollector {
        COLLECTOR
            .lock()
            .ok()
            .and_then(|mut collector| collector.take())
            .unwrap_or_else(|| MeasureCollector {
                epoch: Instant::now(),
                measures: Vec::new(),
            })
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod storage {
    use super::{Measure, MeasureCollector};
    use std::cell::RefCell;
    use web_time::Instant;

    thread_local! {
        static COLLECTOR: RefCell<Option<MeasureCollector>> = const { RefCell::new(None) };
    }

    pub fn set(value: Option<MeasureCollector>) {
        COLLECTOR.with(|collector| {
            *collector.borrow_mut() = value;
        });
    }

    pub fn epoch() -> Option<Instant> {
        COLLECTOR.with(|collector| {
            collector
                .borrow()
                .as_ref()
                .map(|collector| collector.epoch)
        })
    }

    pub fn push_measure(name: &'static str, start: Instant, epoch: Instant) {
        COLLECTOR.with(|collector| {
            let mut collector = collector.borrow_mut();
            let Some(collector) = collector.as_mut() else {
                return;
            };
            collector.measures.push(Measure {
                name,
                start_ms: start.duration_since(epoch).as_secs_f64() * 1000.0,
                duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            });
        });
    }

    pub fn take() -> MeasureCollector {
        COLLECTOR.with(|collector| {
            collector.borrow_mut().take().unwrap_or_else(|| MeasureCollector {
                epoch: Instant::now(),
                measures: Vec::new(),
            })
        })
    }
}

pub(crate) struct PhaseGuard {
    name: &'static str,
    start: Instant,
    epoch: Instant,
}

impl Drop for PhaseGuard {
    fn drop(&mut self) {
        storage::push_measure(self.name, self.start, self.epoch);
    }
}

pub(crate) fn enter_phase(name: &'static str) -> Option<PhaseGuard> {
    let epoch = storage::epoch()?;
    Some(PhaseGuard {
        name,
        start: Instant::now(),
        epoch,
    })
}

/// Runs `f` with a measure collector installed and returns the collected measures.
pub fn with_timing_subscriber<T, F>(f: F) -> (T, DecodeTelemetry)
where
    F: FnOnce() -> T,
{
    let rust_timebase = unix_time_ms();
    storage::set(Some(MeasureCollector {
        epoch: Instant::now(),
        measures: Vec::new(),
    }));

    let result = f();

    let collector = storage::take();
    storage::set(None);

    let telemetry = DecodeTelemetry {
        rust_timebase,
        total_ms: collector.epoch.elapsed().as_secs_f64() * 1000.0,
        measures: collector.measures,
    };

    (result, telemetry)
}

/// Consumer-facing telemetry after rebasing to an outer `<lang>_decode` origin.
#[derive(Debug, Clone, PartialEq)]
pub struct RebasingTelemetry {
    pub timebase: f64,
    pub total_ms: f64,
    pub measures: Vec<RebasedMeasure>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RebasedMeasure {
    pub name: String,
    pub start_ms: f64,
    pub duration_ms: f64,
}

/// Rebases Rust-internal measures to a binding wall-clock origin and prepends
/// `<lang>_decode` at `start_ms: 0`.
pub fn rebase_telemetry(
    native: &DecodeTelemetry,
    timebase: f64,
    outer_name: &str,
    outer_duration_ms: f64,
) -> RebasingTelemetry {
    let delta = (native.rust_timebase - timebase).max(0.0);
    let mut measures = Vec::with_capacity(native.measures.len() + 1);
    measures.push(RebasedMeasure {
        name: outer_name.to_string(),
        start_ms: 0.0,
        duration_ms: outer_duration_ms,
    });
    for measure in &native.measures {
        measures.push(RebasedMeasure {
            name: measure.name.to_string(),
            start_ms: (measure.start_ms + delta).max(0.0),
            duration_ms: measure.duration_ms,
        });
    }
    RebasingTelemetry {
        timebase,
        total_ms: outer_duration_ms,
        measures,
    }
}

/// Enters a phase measure when telemetry is enabled.
#[macro_export]
macro_rules! phase_guard {
    ($name:literal) => {{
        $crate::telemetry::enter_phase($name)
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collects_flat_measures_with_start_and_duration() {
        let (_result, telemetry) = with_timing_subscriber(|| {
            let _outer = crate::phase_guard!("outer");
            let _inner = crate::phase_guard!("inner");
        });

        assert!(telemetry.rust_timebase > 0.0);
        assert_eq!(telemetry.measures.len(), 2);
        assert_eq!(telemetry.measures[0].name, "inner");
        assert_eq!(telemetry.measures[1].name, "outer");
        assert!(telemetry.measures[1].duration_ms >= telemetry.measures[0].duration_ms);
        for measure in &telemetry.measures {
            assert!(measure.start_ms <= telemetry.total_ms);
        }
    }
}
