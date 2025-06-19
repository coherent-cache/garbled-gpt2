use tikv_jemalloc_ctl::{epoch, stats};

/// Memory tracking utilities using jemalloc stats
pub struct MemoryTracker {
    baseline_allocated: usize,
}

impl MemoryTracker {
    /// Create a new memory tracker and capture baseline memory usage
    pub fn new() -> Result<Self, String> {
        epoch::advance().map_err(|e| format!("Failed to advance epoch: {:?}", e))?;
        let baseline_allocated = stats::allocated::read()
            .map_err(|e| format!("Failed to read allocated stats: {:?}", e))?;
        Ok(Self { baseline_allocated })
    }

    /// Get current allocated memory in MB
    pub fn current_allocated_mb(&self) -> Result<f64, String> {
        epoch::advance().map_err(|e| format!("Failed to advance epoch: {:?}", e))?;
        let allocated = stats::allocated::read()
            .map_err(|e| format!("Failed to read allocated stats: {:?}", e))?;
        Ok(allocated as f64 / (1024.0 * 1024.0))
    }

    /// Get peak allocated memory since tracker creation in MB
    pub fn peak_allocated_mb(&self) -> Result<f64, String> {
        epoch::advance().map_err(|e| format!("Failed to advance epoch: {:?}", e))?;
        let allocated = stats::allocated::read()
            .map_err(|e| format!("Failed to read allocated stats: {:?}", e))?;
        let peak = allocated.max(self.baseline_allocated);
        Ok(peak as f64 / (1024.0 * 1024.0))
    }

    /// Get memory usage delta from baseline in MB
    pub fn allocated_delta_mb(&self) -> Result<f64, String> {
        epoch::advance().map_err(|e| format!("Failed to advance epoch: {:?}", e))?;
        let allocated = stats::allocated::read()
            .map_err(|e| format!("Failed to read allocated stats: {:?}", e))?;
        let delta = allocated.saturating_sub(self.baseline_allocated);
        Ok(delta as f64 / (1024.0 * 1024.0))
    }
}

/// Get current resident memory in MB (RSS)
pub fn resident_memory_mb() -> Result<f64, String> {
    epoch::advance().map_err(|e| format!("Failed to advance epoch: {:?}", e))?;
    let resident =
        stats::resident::read().map_err(|e| format!("Failed to read resident stats: {:?}", e))?;
    Ok(resident as f64 / (1024.0 * 1024.0))
}

/// Get current allocated memory in MB  
pub fn allocated_memory_mb() -> Result<f64, String> {
    epoch::advance().map_err(|e| format!("Failed to advance epoch: {:?}", e))?;
    let allocated =
        stats::allocated::read().map_err(|e| format!("Failed to read allocated stats: {:?}", e))?;
    Ok(allocated as f64 / (1024.0 * 1024.0))
}
