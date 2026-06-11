//! GPU runtime environment: adapter availability probed once per process.

use std::sync::OnceLock;

/// Snapshot of GPU runtime capabilities for kernel availability checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuEnvironment {
    /// Whether a wgpu adapter was found on this machine.
    pub device_available: bool,
}

impl GpuEnvironment {
    /// Probes the local GPU environment once and caches the result.
    pub fn current() -> Self {
        static ENV: OnceLock<GpuEnvironment> = OnceLock::new();
        *ENV.get_or_init(probe_environment)
    }
}

fn probe_environment() -> GpuEnvironment {
    #[cfg(feature = "gpu")]
    {
        let instance = wgpu::Instance::default();
        let adapters = instance.enumerate_adapters(wgpu::Backends::all());
        GpuEnvironment {
            device_available: !adapters.is_empty(),
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        GpuEnvironment {
            device_available: false,
        }
    }
}
