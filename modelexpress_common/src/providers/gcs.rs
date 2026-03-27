// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::cache::{ModelInfo, ProviderCache, directory_size};
use crate::models::ModelProvider;
use crate::providers::ModelProviderTrait;
use anyhow::{Context, Result};
use crc32c::Crc32cReader;
use futures::StreamExt;
use futures::stream;
use google_cloud_storage::client::{Storage, StorageControl};
use google_cloud_storage::model::{ListObjectsRequest, Object};
use google_cloud_storage::model_ext::ReadRange;
use std::ffi::OsStr;
use std::fs;
use std::future::Future;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;
use tokio::io::AsyncWriteExt;
use tracing::{info, warn};

const CACHE_ROOT_DIR_NAME: &str = "gcs";
const MAX_GCS_LIST_RESULTS: i32 = 1000;
const MAX_PARALLEL_DOWNLOADS: usize = 8;
const PROGRESS_LOG_INTERVAL_BYTES: u64 = 128 * 1024 * 1024;
const MODEL_MARKER_FILE_NAME: &str = ".mx-model";

fn ensure_crypto_provider() -> Result<()> {
    if rustls::crypto::CryptoProvider::get_default().is_some() {
        return Ok(());
    }

    match rustls::crypto::ring::default_provider().install_default() {
        Ok(()) => Ok(()),
        Err(_) if rustls::crypto::CryptoProvider::get_default().is_some() => Ok(()),
        Err(_) => anyhow::bail!("Failed to install rustls ring CryptoProvider for GCS"),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ModelName {
    bucket: BucketName,
    object_prefix: String,
}

#[derive(Debug, Clone, Copy)]
struct ModelDir<'a> {
    cache_dir: &'a Path,
    model_dir: &'a Path,
}

impl ModelName {
    fn parse(model_name: &str) -> Result<Self> {
        if model_name.is_empty() {
            anyhow::bail!("Model name must not be empty");
        }

        let Some(full_url) = model_name.strip_prefix("gs://") else {
            anyhow::bail!("GCS model name must be a full gs://<bucket>/<path> URL");
        };
        let (bucket_raw, object_prefix) = full_url
            .split_once('/')
            .ok_or_else(|| anyhow::anyhow!("GCS model URL must include bucket and object path"))?;
        if object_prefix.is_empty() {
            anyhow::bail!("GCS model URL must include a non-empty object path");
        }

        let bucket = BucketName::parse(bucket_raw)?;
        Self::new(bucket, object_prefix)
    }

    fn new(bucket: BucketName, object_prefix: &str) -> Result<Self> {
        let normalized_object_prefix = object_prefix.trim_end_matches('/');
        if normalized_object_prefix.is_empty() {
            anyhow::bail!("GCS model path must not be empty");
        }

        let mut components = Vec::new();
        for component in normalized_object_prefix.split('/') {
            if component.is_empty() {
                anyhow::bail!("GCS model path must not contain empty path segments");
            }
            if component == "." || component == ".." {
                anyhow::bail!("GCS model path must not contain '.' or '..' segments");
            }
            components.push(component);
        }

        if components.is_empty() {
            anyhow::bail!("GCS model path must not be empty");
        }

        Ok(Self {
            bucket,
            object_prefix: components.join("/"),
        })
    }

    fn model_dir(&self, cache_dir: &Path) -> PathBuf {
        let mut path = self.bucket_dir(cache_dir);
        for component in self.object_prefix.split('/') {
            path = path.join(component);
        }
        path
    }

    fn bucket_dir(&self, cache_dir: &Path) -> PathBuf {
        cache_dir
            .join(CACHE_ROOT_DIR_NAME)
            .join(self.bucket.as_str())
    }
}

impl std::fmt::Display for ModelName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gs://{}/{}", self.bucket, self.object_prefix)
    }
}

impl<'a> ModelDir<'a> {
    fn new(cache_dir: &'a Path, model_dir: &'a Path) -> Self {
        Self {
            cache_dir,
            model_dir,
        }
    }

    fn model_name(&self) -> Result<ModelName> {
        let gcs_root = self.cache_dir.join(CACHE_ROOT_DIR_NAME);
        let relative = self.model_dir.strip_prefix(&gcs_root).with_context(|| {
            format!(
                "GCS model directory '{}' is outside cache directory '{}'",
                self.model_dir.display(),
                gcs_root.display()
            )
        })?;
        let mut components = relative.components();
        let bucket_component = components.next().ok_or_else(|| {
            anyhow::anyhow!(
                "GCS model directory '{}' is missing bucket and object path",
                self.model_dir.display()
            )
        })?;
        let bucket_name = match bucket_component {
            Component::Normal(component) => component.to_str().ok_or_else(|| {
                anyhow::anyhow!(
                    "GCS bucket component in '{}' is not valid UTF-8",
                    self.model_dir.display()
                )
            })?,
            _ => {
                anyhow::bail!(
                    "GCS model directory '{}' has invalid bucket component",
                    self.model_dir.display()
                )
            }
        };
        let bucket = BucketName::parse(bucket_name)?;

        let mut object_prefix_components = Vec::new();
        for component in components {
            let component = match component {
                Component::Normal(component) => component.to_str().ok_or_else(|| {
                    anyhow::anyhow!(
                        "GCS object path component in '{}' is not valid UTF-8",
                        self.model_dir.display()
                    )
                })?,
                _ => {
                    anyhow::bail!(
                        "GCS model directory '{}' has invalid object path component",
                        self.model_dir.display()
                    )
                }
            };
            object_prefix_components.push(component);
        }

        if object_prefix_components.is_empty() {
            anyhow::bail!(
                "GCS model directory '{}' is missing object path",
                self.model_dir.display()
            );
        }

        ModelName::new(bucket, &object_prefix_components.join("/"))
    }

    fn write_model_marker(&self) -> Result<()> {
        fs::create_dir_all(self.model_dir).with_context(|| {
            format!(
                "Failed to create model directory '{}'",
                self.model_dir.display()
            )
        })?;

        let temp_path = self.model_dir.join(format!("{MODEL_MARKER_FILE_NAME}.tmp"));

        fs::write(&temp_path, [])
            .with_context(|| format!("Failed to write marker file '{}'", temp_path.display()))?;
        fs::rename(&temp_path, self.model_dir.join(MODEL_MARKER_FILE_NAME)).with_context(|| {
            format!(
                "Failed to move marker file '{}' into '{}'",
                temp_path.display(),
                self.model_dir.display()
            )
        })?;

        Ok(())
    }

    fn has_model_marker(&self) -> bool {
        self.model_dir.join(MODEL_MARKER_FILE_NAME).is_file()
    }

    fn find_ancestor_model(&self) -> Result<Option<ModelName>> {
        let model = self.model_name()?;
        let bucket_dir = model.bucket_dir(self.cache_dir);
        let mut current = self.model_dir.parent();
        while let Some(ancestor) = current {
            if !ancestor.starts_with(&bucket_dir) {
                break;
            }

            let ancestor_model_dir = ModelDir::new(self.cache_dir, ancestor);
            if ancestor_model_dir.has_model_marker() {
                return Ok(Some(ancestor_model_dir.model_name()?));
            }

            if ancestor == bucket_dir {
                break;
            }
            current = ancestor.parent();
        }

        Ok(None)
    }

    fn find_descendant_model(&self) -> Result<Option<ModelName>> {
        if !self.model_dir.is_dir() {
            return Ok(None);
        }

        for entry in fs::read_dir(self.model_dir)
            .with_context(|| format!("Failed to read directory '{}'", self.model_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            let child_model_dir = ModelDir::new(self.cache_dir, &path);
            if child_model_dir.has_model_marker() {
                return Ok(Some(child_model_dir.model_name()?));
            }

            if let Some(model) = (ModelDir::new(self.cache_dir, &path)).find_descendant_model()? {
                return Ok(Some(model));
            }
        }

        Ok(None)
    }

    fn find_overlapping_model(&self) -> Result<Option<ModelName>> {
        if let Some(overlap) = self.find_ancestor_model()? {
            return Ok(Some(overlap));
        }

        self.find_descendant_model()
    }

    fn is_removable(&self) -> Result<bool> {
        if self.has_model_marker() {
            self.model_name()?;
            return Ok(true);
        }

        if !self.model_dir.exists() {
            return Ok(false);
        }

        Ok(self.find_overlapping_model()?.is_none())
    }

    fn ensure_available(&self) -> Result<()> {
        let model = self.model_name()?;

        if let Some(overlap) = self.find_ancestor_model()? {
            anyhow::bail!(
                "GCS model '{}' overlaps cached ancestor model '{}'",
                model,
                overlap
            );
        }

        if let Some(overlap) = self.find_descendant_model()? {
            anyhow::bail!(
                "GCS model '{}' overlaps cached descendant model '{}'",
                model,
                overlap
            );
        }

        Ok(())
    }

    fn size(&self) -> Result<u64> {
        let mut size = 0u64;

        for entry in fs::read_dir(self.model_dir)
            .with_context(|| format!("Failed to read directory '{}'", self.model_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            let relative = path
                .strip_prefix(self.model_dir)
                .with_context(|| format!("Failed to make '{}' relative", path.display()))?;

            if path.is_file() {
                if relative == Path::new(MODEL_MARKER_FILE_NAME) {
                    continue;
                }
                size = size.saturating_add(fs::metadata(&path)?.len());
            } else if path.is_dir() {
                size = size.saturating_add(directory_size(&path)?);
            }
        }

        Ok(size)
    }

    fn has_any_files(&self) -> bool {
        if !self.model_dir.is_dir() {
            return false;
        }

        let mut pending = vec![self.model_dir.to_path_buf()];
        while let Some(current_dir) = pending.pop() {
            let Ok(entries) = fs::read_dir(&current_dir) else {
                continue;
            };

            for entry in entries.flatten() {
                let Ok(file_type) = entry.file_type() else {
                    continue;
                };
                let entry_path = entry.path();

                if file_type.is_dir() {
                    pending.push(entry_path);
                    continue;
                }

                if !file_type.is_file() {
                    continue;
                }

                let Ok(relative) = entry_path.strip_prefix(self.model_dir) else {
                    continue;
                };
                if relative == Path::new(MODEL_MARKER_FILE_NAME) {
                    continue;
                }

                return true;
            }
        }

        false
    }

    fn has_weight_files(&self) -> bool {
        if !self.model_dir.is_dir() {
            return false;
        }

        let mut pending = vec![self.model_dir.to_path_buf()];
        while let Some(current_dir) = pending.pop() {
            let Ok(entries) = fs::read_dir(&current_dir) else {
                continue;
            };

            for entry in entries.flatten() {
                let Ok(file_type) = entry.file_type() else {
                    continue;
                };
                let entry_path = entry.path();

                if file_type.is_dir() {
                    pending.push(entry_path);
                    continue;
                }

                if !file_type.is_file() {
                    continue;
                }

                let Ok(relative) = entry_path.strip_prefix(self.model_dir) else {
                    continue;
                };
                if relative.as_os_str().is_empty() || !relative.is_safe_relative() {
                    continue;
                }
                if relative.is_weight_file() {
                    return true;
                }
            }
        }

        false
    }

    fn has_cached_model(&self, ignore_weights: bool) -> bool {
        if !self.has_any_files() {
            return false;
        }

        ignore_weights || self.has_weight_files()
    }

    async fn with_prepared<T, F, Fut>(&self, op: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        if self.model_dir.exists() {
            if self.model_dir.is_dir() {
                fs::remove_dir_all(self.model_dir).with_context(|| {
                    format!(
                        "Failed to clear stale GCS download directory '{}'",
                        self.model_dir.display()
                    )
                })?;
            } else {
                fs::remove_file(self.model_dir).with_context(|| {
                    format!(
                        "Failed to clear stale GCS download file '{}'",
                        self.model_dir.display()
                    )
                })?;
            }
        }

        fs::create_dir_all(self.model_dir).with_context(|| {
            format!("Failed to create directory '{}'", self.model_dir.display())
        })?;

        op().await.map_err(|err| {
            if self.model_dir.exists()
                && let Err(cleanup_err) = fs::remove_dir_all(self.model_dir)
            {
                return anyhow::anyhow!(
                    "{}; also failed to remove partial download directory '{}': {}",
                    err,
                    self.model_dir.display(),
                    cleanup_err
                );
            }

            err
        })
    }
}

pub(crate) struct GcsProviderCache;

impl GcsProviderCache {
    fn collect_cached_models(
        cache_dir: &Path,
        current_dir: &Path,
        models: &mut Vec<ModelInfo>,
    ) -> Result<()> {
        let current_model_dir = ModelDir::new(cache_dir, current_dir);
        if current_model_dir.has_model_marker() {
            match current_model_dir.model_name() {
                Ok(model) => {
                    if !current_model_dir.has_any_files() {
                        warn!("Skipping empty GCS cache entry '{}'", current_dir.display());
                        return Ok(());
                    }

                    models.push(ModelInfo {
                        provider: ModelProvider::Gcs,
                        name: model.to_string(),
                        size: current_model_dir.size()?,
                        path: current_dir.to_path_buf(),
                    });
                    return Ok(());
                }
                Err(err) => {
                    warn!(
                        "Skipping invalid GCS cache entry '{}': {}",
                        current_dir.display(),
                        err
                    );
                }
            }
        }

        for entry in fs::read_dir(current_dir)
            .with_context(|| format!("Failed to read directory '{}'", current_dir.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                Self::collect_cached_models(cache_dir, &path, models)?;
            }
        }

        Ok(())
    }
}

impl ProviderCache for GcsProviderCache {
    fn clear_model(&self, cache_dir: &Path, model_name: &str) -> Result<()> {
        let model = ModelName::parse(model_name)?;
        let model_dir = model.model_dir(cache_dir);

        if !(ModelDir::new(cache_dir, &model_dir)).is_removable()? {
            info!(
                "Model not found in cache: {} ({:?})",
                model_name,
                ModelProvider::Gcs
            );
            return Ok(());
        }

        if model_dir.is_dir() {
            fs::remove_dir_all(&model_dir)
                .with_context(|| format!("Failed to remove model: {model_dir:?}"))?;
        } else {
            fs::remove_file(&model_dir)
                .with_context(|| format!("Failed to remove model: {model_dir:?}"))?;
        }

        info!("Cleared model: {} ({:?})", model_name, ModelProvider::Gcs);
        Ok(())
    }

    fn resolve_model_path(
        &self,
        cache_dir: &Path,
        model_name: &str,
        _revision: Option<&str>,
    ) -> Result<PathBuf> {
        Ok(ModelName::parse(model_name)?.model_dir(cache_dir))
    }

    fn list_models(&self, cache_dir: &Path) -> Result<Vec<ModelInfo>> {
        let mut models = Vec::new();
        let root = cache_dir.join(CACHE_ROOT_DIR_NAME);

        if !root.exists() {
            return Ok(models);
        }

        for bucket_entry in fs::read_dir(&root)? {
            let bucket_entry = bucket_entry?;
            let bucket_path = bucket_entry.path();
            if !bucket_path.is_dir() {
                continue;
            }

            let Some(bucket_name) = bucket_path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };

            if BucketName::parse(bucket_name).is_err() {
                warn!(
                    "Skipping invalid GCS bucket cache entry '{}'",
                    bucket_path.display()
                );
                continue;
            }

            Self::collect_cached_models(cache_dir, &bucket_path, &mut models)?;
        }

        Ok(models)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BucketName {
    normalized: String,
}

impl BucketName {
    fn parse(raw: &str) -> Result<Self> {
        use std::path::{Component, Path};

        let trimmed = raw.trim();
        if trimmed.is_empty() {
            anyhow::bail!("bucket name must not be empty");
        }

        let without_scheme = trimmed.strip_prefix("gs://").unwrap_or(trimmed);
        let without_trailing = without_scheme.trim_end_matches('/');
        if without_trailing.is_empty() {
            anyhow::bail!("trimmed bucket name must not be empty");
        }

        if without_trailing.contains('/') {
            anyhow::bail!("bucket name must contain only the bucket name (no object path)");
        }

        let mut components = Path::new(without_trailing).components();
        match (components.next(), components.next()) {
            (Some(Component::Normal(_)), None) => {}
            _ => anyhow::bail!("bucket name must be a single normal path segment"),
        }

        Ok(Self {
            normalized: without_trailing.to_string(),
        })
    }

    fn as_str(&self) -> &str {
        &self.normalized
    }

    fn resource_name(&self) -> String {
        format!("projects/_/buckets/{}", self.as_str())
    }
}

impl std::fmt::Display for BucketName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

struct DownloadContext {
    bucket: BucketName,
    model: ModelName,
    model_dir: PathBuf,
    ignore_weights: bool,
}

impl DownloadContext {
    fn new(model: ModelName, cache_dir: &Path, ignore_weights: bool) -> Result<DownloadContext> {
        let model_dir = model.model_dir(cache_dir);

        Ok(DownloadContext {
            bucket: model.bucket.clone(),
            model,
            model_dir,
            ignore_weights,
        })
    }
}

#[derive(Debug, Clone)]
struct DownloadTask {
    object_name: String,
    destination_path: PathBuf,
    expected_size: Option<u64>,
    generation: Option<i64>,
    expected_crc32c: Option<u32>,
}

impl DownloadTask {
    async fn verify_temp_file_crc32c(&self, temp_path: &Path) -> Result<()> {
        let Some(expected_crc32c) = self.expected_crc32c else {
            anyhow::bail!(
                "Cannot verify completed temp file '{}' for '{}': remote CRC32C is unavailable",
                temp_path.display(),
                self.object_name
            );
        };

        let verify_path = temp_path.to_path_buf();
        let actual_crc32c =
            tokio::task::spawn_blocking(move || verify_path.calculate_file_crc32c())
                .await
                .with_context(|| {
                    format!(
                        "CRC32C verification task panicked for '{}'",
                        temp_path.display()
                    )
                })??;

        if actual_crc32c != expected_crc32c {
            anyhow::bail!(
                "CRC32C mismatch for completed temp file '{}' for '{}': expected {:08x}, got {:08x}",
                temp_path.display(),
                self.object_name,
                expected_crc32c,
                actual_crc32c
            );
        }

        Ok(())
    }

    async fn promote_verified_temp_file(&self, temp_path: &Path) -> Result<()> {
        if let Err(err) = self.verify_temp_file_crc32c(temp_path).await {
            if let Err(remove_err) = tokio::fs::remove_file(temp_path).await {
                return Err(anyhow::anyhow!(
                    "{}; also failed to discard unverified temp file '{}': {}",
                    err,
                    temp_path.display(),
                    remove_err
                ));
            }
            return Err(err);
        }

        tokio::fs::rename(temp_path, &self.destination_path)
            .await
            .with_context(|| {
                format!(
                    "Failed to move '{}' to '{}'",
                    temp_path.display(),
                    self.destination_path.display()
                )
            })?;

        Ok(())
    }
}

struct DownloadProgress {
    total_files: usize,
    total_bytes: Option<u64>,
    completed_files: AtomicUsize,
    downloaded_bytes: AtomicU64,
    next_log_bytes: AtomicU64,
    started_at: Instant,
}

impl DownloadProgress {
    fn new(tasks: &[DownloadTask]) -> Self {
        let mut total_bytes = Some(0u64);
        for task in tasks {
            total_bytes = match (total_bytes, task.expected_size) {
                (Some(acc), Some(size)) => acc.checked_add(size),
                _ => None,
            };
        }

        Self {
            total_files: tasks.len(),
            total_bytes,
            completed_files: AtomicUsize::new(0),
            downloaded_bytes: AtomicU64::new(0),
            next_log_bytes: AtomicU64::new(PROGRESS_LOG_INTERVAL_BYTES),
            started_at: Instant::now(),
        }
    }

    fn log_start(&self) {
        let total_size = self
            .total_bytes
            .map(Self::format_bytes)
            .unwrap_or_else(|| "unknown".to_string());
        info!(
            "Starting GCS parallel download: {} files ({} total) with {} workers",
            self.total_files, total_size, MAX_PARALLEL_DOWNLOADS
        );
    }

    fn record_downloaded_bytes(&self, bytes: u64) {
        if bytes == 0 {
            return;
        }

        let previous = self.downloaded_bytes.fetch_add(bytes, Ordering::Relaxed);
        let current = previous.saturating_add(bytes);
        self.maybe_log_threshold_progress(current);
    }

    fn mark_file_completed(&self, object_name: &str) {
        let previous = self.completed_files.fetch_add(1, Ordering::Relaxed);
        let completed = previous.saturating_add(1);
        self.log_progress(&format!(
            "Completed '{}' ({}/{})",
            object_name, completed, self.total_files
        ));
    }

    fn log_finish(&self) {
        let elapsed = self.started_at.elapsed();
        self.log_progress(&format!(
            "Finished GCS download in {:.1}s",
            elapsed.as_secs_f64()
        ));
    }

    fn maybe_log_threshold_progress(&self, current_bytes: u64) {
        let mut threshold = self.next_log_bytes.load(Ordering::Relaxed);
        while current_bytes >= threshold {
            match self.next_log_bytes.compare_exchange(
                threshold,
                threshold.saturating_add(PROGRESS_LOG_INTERVAL_BYTES),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    self.log_progress("GCS download progress");
                    break;
                }
                Err(observed) => threshold = observed,
            }
        }
    }

    fn log_progress(&self, prefix: &str) {
        let completed = self.completed_files.load(Ordering::Relaxed);
        let downloaded = self.downloaded_bytes.load(Ordering::Relaxed);
        match self.total_bytes {
            Some(total) if total > 0 => {
                let percent = ((downloaded as f64) * 100.0 / (total as f64)).min(100.0);
                info!(
                    "{}: files {}/{}, bytes {}/{} ({:.1}%)",
                    prefix,
                    completed,
                    self.total_files,
                    Self::format_bytes(downloaded),
                    Self::format_bytes(total),
                    percent
                );
            }
            Some(_) | None => {
                info!(
                    "{}: files {}/{}, bytes {}",
                    prefix,
                    completed,
                    self.total_files,
                    Self::format_bytes(downloaded)
                );
            }
        }
    }

    fn format_bytes(bytes: u64) -> String {
        const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
        let mut value = bytes as f64;
        let mut unit_idx = 0usize;
        let last_unit_idx = UNITS.len().saturating_sub(1);
        while value >= 1024.0 && unit_idx < last_unit_idx {
            value /= 1024.0;
            unit_idx = unit_idx.saturating_add(1);
        }
        format!("{value:.1} {}", UNITS[unit_idx])
    }
}

fn has_empty_path_segment(path: &str) -> bool {
    path.split('/').any(str::is_empty)
}

trait PathExt {
    fn calculate_file_crc32c(&self) -> Result<u32>;
    fn is_safe_relative(&self) -> bool;
    fn is_weight_file(&self) -> bool;
    fn should_download(&self, ignore_weights: bool) -> bool;
    fn temp_download_path(&self) -> PathBuf;
}

impl PathExt for Path {
    fn calculate_file_crc32c(&self) -> Result<u32> {
        let file = fs::File::open(self).with_context(|| {
            format!(
                "Failed to open '{}' for CRC32C verification",
                self.display()
            )
        })?;
        let mut reader = Crc32cReader::new(file);
        io::copy(&mut reader, &mut io::sink()).with_context(|| {
            format!(
                "Failed to read '{}' for CRC32C verification",
                self.display()
            )
        })?;
        Ok(reader.crc32c())
    }

    fn is_safe_relative(&self) -> bool {
        if self.is_absolute() {
            return false;
        }

        self.components()
            .all(|component| matches!(component, Component::Normal(_)))
    }

    fn is_weight_file(&self) -> bool {
        GcsProvider::is_weight_file(self.to_string_lossy().as_ref())
    }

    fn should_download(&self, ignore_weights: bool) -> bool {
        !GcsProvider::is_ignored(self.to_string_lossy().as_ref())
            && !GcsProvider::is_image(self)
            && (!ignore_weights || !self.is_weight_file())
    }

    fn temp_download_path(&self) -> PathBuf {
        let mut temp_name = self
            .file_name()
            .map(|value| value.to_os_string())
            .unwrap_or_else(|| OsStr::new("download").to_os_string());
        temp_name.push(".tmp");
        self.with_file_name(temp_name)
    }
}

struct Downloader<'a> {
    storage: &'a Storage,
    control: &'a StorageControl,
    context: &'a DownloadContext,
    bucket_resource_name: String,
    list_prefix: String,
}

impl<'a> Downloader<'a> {
    fn new(
        storage: &'a Storage,
        control: &'a StorageControl,
        context: &'a DownloadContext,
    ) -> Self {
        Self {
            storage,
            control,
            context,
            bucket_resource_name: context.bucket.resource_name(),
            list_prefix: format!("{}/", context.model.object_prefix),
        }
    }

    fn build_list_request(&self, page_token: Option<String>) -> ListObjectsRequest {
        let mut request = ListObjectsRequest::new()
            .set_parent(self.bucket_resource_name.clone())
            .set_prefix(self.list_prefix.clone())
            .set_page_size(MAX_GCS_LIST_RESULTS);

        if let Some(page_token) = page_token {
            request = request.set_page_token(page_token);
        }

        request
    }

    fn ensure_parent_directory(path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create directory '{}'", parent.display()))?;
        }
        Ok(())
    }

    fn download_task_if_allowed(&self, object: &Object) -> Result<Option<DownloadTask>> {
        let object_name = &object.name;
        let Some(relative_key) = object_name
            .strip_prefix(&self.list_prefix)
            .filter(|suffix| !suffix.is_empty() && !suffix.ends_with('/'))
        else {
            return Ok(None);
        };
        if has_empty_path_segment(relative_key) {
            anyhow::bail!(
                "Unsafe object path '{relative_key}' for model '{}': empty path segments are not allowed",
                self.context.model
            );
        }
        let relative_path = Path::new(relative_key);
        if !relative_path.is_safe_relative() {
            anyhow::bail!(
                "Unsafe object path '{relative_key}' for model '{}'",
                self.context.model
            );
        }
        if relative_path == Path::new(MODEL_MARKER_FILE_NAME) {
            anyhow::bail!(
                "GCS model '{}' contains reserved marker path '{}'",
                self.context.model,
                relative_key
            );
        }

        if !relative_path.should_download(self.context.ignore_weights) {
            return Ok(None);
        }

        let destination_path = self.context.model_dir.join(relative_path);
        let expected_size = u64::try_from(object.size).ok();
        let generation = (object.generation > 0).then_some(object.generation);
        let expected_crc32c = object
            .checksums
            .as_ref()
            .and_then(|checksums| checksums.crc32c);
        Ok(Some(DownloadTask {
            object_name: object_name.to_string(),
            destination_path,
            expected_size,
            generation,
            expected_crc32c,
        }))
    }

    async fn collect_download_tasks(&self) -> Result<Vec<DownloadTask>> {
        let mut page_token: Option<String> = None;
        let mut tasks = Vec::new();

        loop {
            let response = self
                .control
                .list_objects()
                .with_request(self.build_list_request(page_token.clone()))
                .send()
                .await
                .with_context(|| {
                    format!(
                        "Failed to list objects in gs://{}/{}",
                        self.context.bucket, self.context.model.object_prefix
                    )
                })?;

            for object in response.objects {
                if let Some(task) = self.download_task_if_allowed(&object)? {
                    tasks.push(task);
                }
            }

            if response.next_page_token.is_empty() {
                break;
            }
            page_token = Some(response.next_page_token);
        }

        Ok(tasks)
    }

    async fn download_matching_objects(&self) -> Result<usize> {
        let tasks = self.collect_download_tasks().await?;
        self.download_tasks_in_parallel(tasks).await
    }

    async fn download_tasks_in_parallel(&self, tasks: Vec<DownloadTask>) -> Result<usize> {
        let task_count = tasks.len();
        if task_count == 0 {
            return Ok(0);
        }
        let progress = Arc::new(DownloadProgress::new(&tasks));
        progress.log_start();

        let download_stream = stream::iter(tasks.into_iter().map(|task| {
            let progress = Arc::clone(&progress);
            async move {
                self.download_task_once(&task, progress.as_ref()).await?;
                progress.mark_file_completed(&task.object_name);
                Ok::<(), anyhow::Error>(())
            }
        }))
        .buffer_unordered(MAX_PARALLEL_DOWNLOADS);

        tokio::pin!(download_stream);

        let mut downloaded_files = 0usize;
        while let Some(result) = download_stream.next().await {
            result?;
            downloaded_files = downloaded_files.saturating_add(1);
        }
        progress.log_finish();

        Ok(downloaded_files)
    }

    async fn download_task_once(
        &self,
        task: &DownloadTask,
        progress: &DownloadProgress,
    ) -> Result<()> {
        Self::ensure_parent_directory(&task.destination_path)?;

        let temp_path = task.destination_path.temp_download_path();
        let bucket = &self.context.bucket;
        let result: Result<()> = async {
            let mut resume_offset = match tokio::fs::metadata(&temp_path).await {
                Ok(metadata) => metadata.len(),
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => 0u64,
                Err(err) => {
                    return Err(anyhow::anyhow!(
                        "Failed to inspect '{}': {}",
                        temp_path.display(),
                        err
                    ));
                }
            };

            if let Some(expected_size) = task.expected_size {
                if resume_offset > expected_size {
                    tokio::fs::remove_file(&temp_path).await.with_context(|| {
                        format!(
                            "Failed to clear oversized temp file '{}'",
                            temp_path.display()
                        )
                    })?;
                    resume_offset = 0;
                } else if resume_offset == expected_size && resume_offset > 0 {
                    match task.promote_verified_temp_file(&temp_path).await {
                        Ok(()) => return Ok(()),
                        Err(err) => {
                            warn!("{err}");
                            resume_offset = 0;
                        }
                    }
                }
            }

            if resume_offset > 0 && task.generation.is_none() {
                warn!(
                    "Discarding partial temp file '{}' for '{}' because remote generation is unavailable",
                    temp_path.display(),
                    task.object_name
                );
                tokio::fs::remove_file(&temp_path).await.with_context(|| {
                    format!(
                        "Failed to discard temp file '{}' without a known remote generation",
                        temp_path.display()
                    )
                })?;
                resume_offset = 0;
            }

            let read_range = if resume_offset == 0 {
                ReadRange::all()
            } else {
                ReadRange::offset(resume_offset)
            };

            let mut request = self
                .storage
                .read_object(self.bucket_resource_name.clone(), task.object_name.clone())
                .set_read_range(read_range);
            if let Some(generation) = task.generation {
                request = request.set_generation(generation);
            }

            let mut stream = request
                .send()
                .await
                .with_context(|| {
                    format!(
                        "Failed to start download for gs://{bucket}/{}",
                        task.object_name
                    )
                })?;

            let mut file = if resume_offset == 0 {
                tokio::fs::File::create(&temp_path)
                    .await
                    .with_context(|| format!("Failed to create '{}'", temp_path.display()))?
            } else {
                tokio::fs::OpenOptions::new()
                    .append(true)
                    .open(&temp_path)
                    .await
                    .with_context(|| {
                        format!(
                            "Failed to open '{}' for resume at byte {}",
                            temp_path.display(),
                            resume_offset
                        )
                    })?
            };

            while let Some(chunk) = stream.next().await {
                let chunk = chunk.with_context(|| {
                    format!("Failed while streaming gs://{bucket}/{}", task.object_name)
                })?;
                file.write_all(&chunk)
                    .await
                    .with_context(|| format!("Failed to write '{}'", temp_path.display()))?;
                progress.record_downloaded_bytes(chunk.len() as u64);
            }

            file.flush()
                .await
                .with_context(|| format!("Failed to flush '{}'", temp_path.display()))?;
            drop(file);

            if let Some(expected_size) = task.expected_size {
                let final_size = tokio::fs::metadata(&temp_path)
                    .await
                    .with_context(|| {
                        format!(
                            "Failed to inspect completed temp file '{}'",
                            temp_path.display()
                        )
                    })?
                    .len();
                if final_size != expected_size {
                    anyhow::bail!(
                        "Incomplete download for '{}': expected {} bytes, got {} bytes",
                        task.object_name,
                        expected_size,
                        final_size
                    );
                }
            }

            task.promote_verified_temp_file(&temp_path).await?;

            Ok(())
        }
        .await;

        result
    }
}

pub struct GcsProvider;

#[async_trait::async_trait]
impl ModelProviderTrait for GcsProvider {
    async fn download_model(
        &self,
        model_name: &str,
        cache_dir: Option<PathBuf>,
        ignore_weights: bool,
    ) -> Result<PathBuf> {
        let cache_dir = cache_dir
            .ok_or_else(|| anyhow::anyhow!("GCS download requires cache_dir to be provided"))?;
        fs::create_dir_all(&cache_dir).with_context(|| {
            format!("Failed to create cache directory: {}", cache_dir.display())
        })?;

        let model = ModelName::parse(model_name)?;

        let model_dir = model.model_dir(&cache_dir);
        let current_model_dir = ModelDir::new(&cache_dir, &model_dir);
        if current_model_dir.has_model_marker()
            && current_model_dir.has_cached_model(ignore_weights)
        {
            info!(
                "Using cached GCS model '{}' from '{}'",
                model,
                model_dir.display()
            );
            return Ok(model_dir);
        }

        (ModelDir::new(&cache_dir, &model_dir)).ensure_available()?;

        let context = DownloadContext::new(model, &cache_dir, ignore_weights)?;

        let downloaded_files = (ModelDir::new(&cache_dir, &context.model_dir))
            .with_prepared(|| async {
                ensure_crypto_provider()?;
                let storage = Storage::builder()
                    .build()
                    .await
                    .context("Failed to initialize Google Cloud Storage data client")?;
                let control = StorageControl::builder()
                    .build()
                    .await
                    .context("Failed to initialize Google Cloud Storage control client")?;
                let downloader = Downloader::new(&storage, &control, &context);
                let downloaded_files = downloader.download_matching_objects().await?;

                if downloaded_files == 0 {
                    anyhow::bail!("No downloadable files found in {}", context.model);
                }

                Ok(downloaded_files)
            })
            .await?;

        info!(
            "Downloaded {} files for model '{}'",
            downloaded_files, context.model
        );

        (ModelDir::new(&cache_dir, &context.model_dir))
            .write_model_marker()
            .with_context(|| {
                format!(
                    "Failed to write marker for downloaded GCS model '{}'",
                    context.model
                )
            })?;

        Ok(context.model_dir)
    }

    async fn delete_model(&self, model_name: &str, cache_dir: PathBuf) -> Result<()> {
        let model = ModelName::parse(model_name)?;
        let model_dir = model.model_dir(&cache_dir);

        if !(ModelDir::new(&cache_dir, &model_dir)).is_removable()? {
            info!(
                "GCS model '{}' not found in cache, skipping delete",
                model_name
            );
            return Ok(());
        }

        if model_dir.is_dir() {
            fs::remove_dir_all(&model_dir).with_context(|| {
                format!(
                    "Failed to remove cached GCS model directory '{}'",
                    model_dir.display()
                )
            })?;
        } else {
            fs::remove_file(&model_dir).with_context(|| {
                format!(
                    "Failed to remove cached GCS model file '{}'",
                    model_dir.display()
                )
            })?;
        }

        info!(
            "Deleted cached GCS model '{}' from '{}'",
            model_name,
            model_dir.display()
        );
        Ok(())
    }

    async fn get_model_path(&self, model_name: &str, cache_dir: PathBuf) -> Result<PathBuf> {
        let model = ModelName::parse(model_name)?;
        let model_dir = model.model_dir(&cache_dir);
        if !(ModelDir::new(&cache_dir, &model_dir)).has_model_marker() {
            anyhow::bail!("GCS model '{model_name}' not found in cache");
        }

        if !(ModelDir::new(&cache_dir, &model_dir)).has_any_files() {
            anyhow::bail!("GCS '{model_name}' is empty");
        }

        Ok(model_dir)
    }

    fn canonical_model_name(&self, model_name: &str) -> Result<String> {
        Ok(ModelName::parse(model_name)?.to_string())
    }

    fn provider_name(&self) -> &'static str {
        "GCS"
    }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn expected_model_dir(cache_dir: &Path, model_name: &str) -> PathBuf {
        ModelName::parse(model_name)
            .expect("Expected model parsing")
            .model_dir(cache_dir)
    }

    fn write_cached_model(
        cache_dir: &Path,
        model_name: &str,
        relative_path: &str,
        contents: &[u8],
    ) {
        let model_dir = expected_model_dir(cache_dir, model_name);
        let payload_path = model_dir.join(relative_path);
        if let Some(parent) = payload_path.parent() {
            fs::create_dir_all(parent).expect("Failed to create model payload directory");
        }
        fs::write(&payload_path, contents).expect("Failed to write model payload");
        (ModelDir::new(cache_dir, &model_dir))
            .write_model_marker()
            .expect("Failed to write model marker");
    }

    fn write_incomplete_cached_model(
        cache_dir: &Path,
        model_name: &str,
        relative_path: &str,
        contents: &[u8],
    ) {
        let model_dir = expected_model_dir(cache_dir, model_name);
        let payload_path = model_dir.join(relative_path);
        if let Some(parent) = payload_path.parent() {
            fs::create_dir_all(parent).expect("Failed to create model payload directory");
        }
        fs::write(&payload_path, contents).expect("Failed to write model payload");
    }

    #[test]
    fn test_provider_name() {
        let provider = GcsProvider;
        assert_eq!(provider.provider_name(), "GCS");
    }

    #[test]
    fn test_model_name_display_full_url_is_stable() {
        assert_eq!(
            ModelName::parse("gs://testbucket/dev/bake/qwen/rev123")
                .expect("Expected model name")
                .to_string(),
            "gs://testbucket/dev/bake/qwen/rev123"
        );
    }

    #[test]
    fn test_model_name_display_trailing_slash_normalizes() {
        assert_eq!(
            ModelName::parse("gs://testbucket/dev/bake/qwen/rev123/")
                .expect("Expected model name")
                .to_string(),
            "gs://testbucket/dev/bake/qwen/rev123"
        );
    }

    #[test]
    fn test_model_marker_uses_mx_model_filename() {
        assert_eq!(Path::new(".mx-model"), Path::new(MODEL_MARKER_FILE_NAME));
        assert_ne!(
            Path::new(".mx-model.json"),
            Path::new(MODEL_MARKER_FILE_NAME)
        );
        assert_ne!(
            Path::new(".modelexpress-gcs-model.json"),
            Path::new(MODEL_MARKER_FILE_NAME)
        );
    }

    #[test]
    fn test_write_model_marker_creates_empty_marker_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_name = "gs://testbucket/dev/bake/qwen/rev123";
        let model_dir = expected_model_dir(temp_dir.path(), model_name);
        (ModelDir::new(temp_dir.path(), &model_dir))
            .write_model_marker()
            .expect("Expected marker write");

        let marker_path = model_dir.join(MODEL_MARKER_FILE_NAME);
        assert!(marker_path.is_file());
        assert_eq!(
            fs::metadata(&marker_path)
                .expect("Expected marker metadata")
                .len(),
            0
        );
    }

    #[test]
    fn test_read_marked_model_derives_model_name_from_cache_path() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_name = "gs://testbucket/dev/bake/qwen/rev123";
        let model_dir = expected_model_dir(temp_dir.path(), model_name);
        fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        fs::write(model_dir.join(MODEL_MARKER_FILE_NAME), []).expect("Failed to write marker file");

        let model = ModelDir::new(temp_dir.path(), &model_dir);
        let model = model.model_name().expect("Expected marker read");
        assert_eq!(model.to_string(), "gs://testbucket/dev/bake/qwen/rev123");
    }

    #[test]
    fn test_gcs_model_name_full_url_derives_canonical_paths() {
        let model =
            ModelName::parse("gs://testbucket/dev/bake/qwen/rev123").expect("Expected parsing");
        assert_eq!(model.to_string(), "gs://testbucket/dev/bake/qwen/rev123");
        assert_eq!(
            model.model_dir(Path::new("/tmp/cache")),
            PathBuf::from("/tmp/cache/gcs/testbucket/dev/bake/qwen/rev123")
        );
    }

    #[test]
    fn test_gcs_model_name_parse_rejects_empty() {
        assert!(ModelName::parse("").is_err());
    }

    #[test]
    fn test_gcs_model_name_parse_full_url() {
        let parsed =
            ModelName::parse("gs://sourcebucket/dev/bake/qwen/rev123").expect("Expected parse");
        assert_eq!(parsed.to_string(), "gs://sourcebucket/dev/bake/qwen/rev123");
    }

    #[test]
    fn test_gcs_model_name_full_url_trailing_slash_normalizes() {
        let parsed =
            ModelName::parse("gs://sourcebucket/dev/bake/qwen/rev123/").expect("Expected parse");
        assert_eq!(parsed.to_string(), "gs://sourcebucket/dev/bake/qwen/rev123");
    }

    #[test]
    fn test_gcs_model_name_requires_path_for_full_url() {
        assert!(ModelName::parse("gs://bucket-only").is_err());
        assert!(ModelName::parse("gs://bucket-only/").is_err());
    }

    #[test]
    fn test_gcs_model_name_rejects_relative_path() {
        assert!(ModelName::parse("dev/bake/qwen/rev123").is_err());
    }

    #[test]
    fn test_gcs_bucket_name_parse_normalizes() {
        assert_eq!(
            BucketName::parse("gs://example-bucket/")
                .expect("Expected bucket normalization")
                .as_str(),
            "example-bucket"
        );
    }

    #[test]
    fn test_gcs_bucket_name_parse_rejects_path() {
        assert!(BucketName::parse("example-bucket/path").is_err());
    }

    #[test]
    fn test_gcs_bucket_name_parse_rejects_escape_segments() {
        assert!(BucketName::parse("..").is_err());
        assert!(BucketName::parse(".").is_err());
        assert!(BucketName::parse("gs://../").is_err());
    }

    #[tokio::test]
    async fn test_download_model_requires_explicit_cache_dir() {
        let provider = GcsProvider;
        let result = provider
            .download_model("gs://test-bucket/test/model/rev-1", None, false)
            .await;
        assert!(result.is_err());
        assert!(
            result
                .expect_err("Expected missing cache_dir error")
                .to_string()
                .contains("requires cache_dir")
        );
    }

    #[test]
    fn test_download_model_uses_cached_model_for_full_url() {
        let provider = GcsProvider;
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_name = "gs://test-bucket/org/model/rev-1";
        let model_dir = expected_model_dir(temp_dir.path(), model_name);
        fs::create_dir_all(model_dir.join("weights")).expect("Failed to create model cache dir");
        fs::write(model_dir.join("weights/model.bin"), b"weights")
            .expect("Failed to create cached weight file");
        (ModelDir::new(temp_dir.path(), &model_dir))
            .write_model_marker()
            .expect("Failed to write model marker");

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create runtime");
        let result = runtime
            .block_on(provider.download_model(
                model_name,
                Some(temp_dir.path().to_path_buf()),
                false,
            ))
            .expect("Expected cached model reuse");

        assert_eq!(result, model_dir);
    }

    #[test]
    fn test_download_model_rejects_relative_path() {
        let provider = GcsProvider;
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create runtime");
        let result = runtime.block_on(provider.download_model(
            "org/model/rev-2",
            Some(temp_dir.path().to_path_buf()),
            false,
        ));
        assert!(result.is_err());
        assert!(
            result
                .expect_err("Expected relative-path validation error")
                .to_string()
                .contains("full gs://<bucket>/<path> URL")
        );
    }

    #[test]
    fn test_is_safe_relative_path() {
        assert!(Path::new("weights/model.safetensors").is_safe_relative());
        assert!(!Path::new("../model.safetensors").is_safe_relative());
        assert!(!Path::new("/etc/passwd").is_safe_relative());
    }

    #[test]
    fn test_should_download_respects_ignore_weights_and_standard_ignored_files() {
        let should_download = |relative_path: &str, ignore_weights: bool| {
            Path::new(relative_path).should_download(ignore_weights)
        };

        assert!(!should_download("README.md", false));
        assert!(!should_download("nested/README.md", false));
        assert!(!should_download(".gitattributes", false));
        assert!(!should_download("image.png", false));
        assert!(!should_download("nested/image.webp", false));
        assert!(should_download("config.json", false));
        assert!(should_download("tokenizer.json", false));
        assert!(should_download("foo.mlir", false));
        assert!(should_download("compile_summary.json", false));
        assert!(should_download("manifest.json", false));
        assert!(should_download("nested/preset.json", false));
        assert!(should_download("nested/pipeline-config.json", false));
        assert!(should_download("nested/text_embeddings.npz", false));

        assert!(should_download("model.safetensors", false));
        assert!(!should_download("model.safetensors", true));
        assert!(should_download("weights/model.bin", false));
        assert!(!should_download("weights/model.bin", true));
        assert!(should_download("weights/model.iop", false));
        assert!(!should_download("weights/model.iop", true));
        assert!(should_download("weights/model.gas", false));
        assert!(!should_download("weights/model.gas", true));
        assert!(should_download("weights/model.h5", false));
        assert!(!should_download("weights/model.h5", true));
        assert!(should_download("weights/model.msgpack", false));
        assert!(!should_download("weights/model.msgpack", true));
        assert!(should_download("weights/model.ckpt.index", false));
        assert!(!should_download("weights/model.ckpt.index", true));
    }

    #[test]
    fn test_gcs_weight_file_detection_includes_generic_and_lpu_formats() {
        let is_weight_file = |relative_path: &str| Path::new(relative_path).is_weight_file();

        assert!(is_weight_file("weights/model.bin"));
        assert!(is_weight_file("weights/model.safetensors"));
        assert!(is_weight_file("weights/model.h5"));
        assert!(is_weight_file("weights/model.msgpack"));
        assert!(is_weight_file("weights/model.ckpt.index"));
        assert!(is_weight_file("weights/model.iop"));
        assert!(is_weight_file("weights/model.gas"));
        assert!(!is_weight_file("tokenizer.json"));
        assert!(!is_weight_file("compile_summary.json"));
        assert!(!is_weight_file("image.png"));
    }

    #[test]
    fn test_relative_gcs_key_safe_suffix_is_downloadable() {
        let relative_path = Path::new("tokenizer.json");
        assert!(relative_path.is_safe_relative());
        assert!(relative_path.should_download(false));
    }

    #[test]
    fn test_gcs_candidate_filter_ignores_non_file_candidates() {
        fn candidate<'a>(object_name: &'a str, list_prefix: &str) -> Option<&'a str> {
            object_name
                .strip_prefix(list_prefix)
                .filter(|suffix| !suffix.is_empty() && !suffix.ends_with('/'))
        }

        assert!(candidate("other/model/file.bin", "org/model/rev123/").is_none());
        assert!(candidate("org/model/rev123/", "org/model/rev123/").is_none());
        assert!(candidate("org/model/rev123/subdir/", "org/model/rev123/").is_none());
        assert_eq!(
            candidate("org/model/rev123/tokenizer.json", "org/model/rev123/"),
            Some("tokenizer.json")
        );
    }

    #[test]
    fn test_relative_gcs_key_rejects_unsafe_suffix() {
        assert!(!Path::new("../secret.bin").is_safe_relative());
    }

    #[test]
    fn test_relative_gcs_key_rejects_empty_path_segments() {
        assert!(has_empty_path_segment("a//b.bin"));
        assert!(has_empty_path_segment("/a/b.bin"));
        assert!(has_empty_path_segment("a/b//c.bin"));
        assert!(!has_empty_path_segment("a/b/c.bin"));
    }

    #[test]
    fn test_relative_path_must_be_non_empty_and_safe() {
        let is_valid_relative =
            |path: &Path| !path.as_os_str().is_empty() && path.is_safe_relative();

        assert!(is_valid_relative(Path::new("weights/model.bin")));
        assert!(!is_valid_relative(Path::new("")));
        assert!(!is_valid_relative(Path::new("/etc/passwd")));
        assert!(!is_valid_relative(Path::new("../escape")));
    }

    #[test]
    fn test_temp_download_path_is_unique_per_filename() {
        let bin = Path::new("/tmp/weights/model.bin");
        let mlir = Path::new("/tmp/weights/model.mlir");
        let no_ext = Path::new("/tmp/weights/model");

        assert_eq!(
            bin.temp_download_path(),
            PathBuf::from("/tmp/weights/model.bin.tmp")
        );
        assert_eq!(
            mlir.temp_download_path(),
            PathBuf::from("/tmp/weights/model.mlir.tmp")
        );
        assert_eq!(
            no_ext.temp_download_path(),
            PathBuf::from("/tmp/weights/model.tmp")
        );
    }

    #[tokio::test]
    async fn test_promote_verified_temp_file_accepts_matching_crc32c() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let temp_path = temp_dir.path().join("weights.bin.tmp");
        let destination_path = temp_dir.path().join("weights.bin");
        let payload = b"weights";
        fs::write(&temp_path, payload).expect("Failed to write temp file");

        let task = DownloadTask {
            object_name: "org/model/rev/weights.bin".to_string(),
            destination_path: destination_path.clone(),
            expected_size: Some(payload.len() as u64),
            generation: Some(42),
            expected_crc32c: Some(crc32c::crc32c(payload)),
        };

        task.promote_verified_temp_file(&temp_path)
            .await
            .expect("Expected checksum verification");

        assert!(!temp_path.exists());
        assert_eq!(
            fs::read(&destination_path).expect("Expected promoted destination file"),
            payload
        );
    }

    #[tokio::test]
    async fn test_promote_verified_temp_file_rejects_crc32c_mismatch() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let temp_path = temp_dir.path().join("weights.bin.tmp");
        let destination_path = temp_dir.path().join("weights.bin");
        fs::write(&temp_path, b"weights").expect("Failed to write temp file");

        let task = DownloadTask {
            object_name: "org/model/rev/weights.bin".to_string(),
            destination_path: destination_path.clone(),
            expected_size: Some(7),
            generation: Some(42),
            expected_crc32c: Some(crc32c::crc32c(b"different")),
        };

        let err = task
            .promote_verified_temp_file(&temp_path)
            .await
            .expect_err("Expected checksum mismatch");

        assert!(
            err.to_string().contains("CRC32C mismatch"),
            "Unexpected error: {err}"
        );
        assert!(!temp_path.exists());
        assert!(!destination_path.exists());
    }

    #[tokio::test]
    async fn test_promote_verified_temp_file_rejects_missing_crc32c() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let temp_path = temp_dir.path().join("weights.bin.tmp");
        let destination_path = temp_dir.path().join("weights.bin");
        fs::write(&temp_path, b"weights").expect("Failed to write temp file");

        let task = DownloadTask {
            object_name: "org/model/rev/weights.bin".to_string(),
            destination_path: destination_path.clone(),
            expected_size: Some(7),
            generation: Some(42),
            expected_crc32c: None,
        };

        let err = task
            .promote_verified_temp_file(&temp_path)
            .await
            .expect_err("Expected missing checksum verification failure");

        assert!(
            err.to_string().contains("remote CRC32C is unavailable"),
            "Unexpected error: {err}"
        );
        assert!(!temp_path.exists());
        assert!(!destination_path.exists());
    }

    #[test]
    fn test_has_weight_files_detects_nested_weights() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let root = temp_dir.path();
        fs::create_dir_all(root.join("nested/deep")).expect("Failed to create nested dirs");
        fs::write(root.join("nested/deep/model.bin"), b"weights")
            .expect("Failed to create weight file");

        assert!((ModelDir::new(root, root)).has_weight_files());
    }

    #[test]
    fn test_has_weight_files_ignores_non_weight_files() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let root = temp_dir.path();
        fs::create_dir_all(root.join("nested")).expect("Failed to create nested dirs");
        fs::write(root.join("nested/tokenizer.json"), b"{}")
            .expect("Failed to create non-weight file");

        assert!(!(ModelDir::new(root, root)).has_weight_files());
    }

    #[test]
    fn test_has_cached_model_requires_weights_unless_ignored() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let root = temp_dir.path();
        fs::create_dir_all(root).expect("Failed to create root dir");
        fs::write(root.join("tokenizer.json"), b"{}").expect("Failed to create non-weight file");

        let model_dir = ModelDir::new(root, root);
        assert!(!model_dir.has_cached_model(false));
        assert!(model_dir.has_cached_model(true));
    }

    #[test]
    fn test_has_any_files_ignores_root_marker_file() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let root = temp_dir.path();
        fs::create_dir_all(root).expect("Failed to create root dir");
        fs::write(root.join(MODEL_MARKER_FILE_NAME), b"{}").expect("Failed to create marker file");

        assert!(!(ModelDir::new(root, root)).has_any_files());
    }

    #[test]
    fn test_list_models_discovers_recursive_sibling_roots() {
        let cache = GcsProviderCache;
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        write_cached_model(
            temp_dir.path(),
            "gs://bucket/foo/bar/baz",
            "tokenizer.json",
            b"{}",
        );
        write_cached_model(
            temp_dir.path(),
            "gs://bucket/foo/bar/buz",
            "weights/model.bin",
            b"abcd",
        );

        let mut models = cache
            .list_models(temp_dir.path())
            .expect("Expected model listing");
        models.sort_by(|left, right| left.name.cmp(&right.name));

        assert_eq!(models.len(), 2);
        assert_eq!(models[0].name, "gs://bucket/foo/bar/baz");
        assert_eq!(models[0].size, 2);
        assert_eq!(models[1].name, "gs://bucket/foo/bar/buz");
        assert_eq!(models[1].size, 4);
    }

    #[test]
    fn test_clear_model_leaves_descendant_when_ancestor_not_cached() {
        let cache = GcsProviderCache;
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let ancestor_name = "gs://bucket/foo/bar";
        let descendant_name = "gs://bucket/foo/bar/baz";
        write_cached_model(temp_dir.path(), descendant_name, "tokenizer.json", b"{}");

        cache
            .clear_model(temp_dir.path(), ancestor_name)
            .expect("Expected clear to succeed");

        assert!(expected_model_dir(temp_dir.path(), descendant_name).exists());
    }

    #[test]
    fn test_clear_model_removes_incomplete_model_dir_without_marker() {
        let cache = GcsProviderCache;
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_name = "gs://bucket/foo/bar";
        let model_dir = expected_model_dir(temp_dir.path(), model_name);
        write_incomplete_cached_model(temp_dir.path(), model_name, "tokenizer.json", b"{}");

        cache
            .clear_model(temp_dir.path(), model_name)
            .expect("Expected clear to succeed");

        assert!(!model_dir.exists());
    }

    #[test]
    fn test_clear_model_keeps_existing_ancestor_for_descendant_path() {
        let cache = GcsProviderCache;
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let ancestor_name = "gs://bucket/foo/bar";
        let descendant_name = "gs://bucket/foo/bar/baz";
        let ancestor_dir = expected_model_dir(temp_dir.path(), ancestor_name);
        let descendant_dir = expected_model_dir(temp_dir.path(), descendant_name);
        write_cached_model(temp_dir.path(), ancestor_name, "tokenizer.json", b"{}");
        fs::create_dir_all(&descendant_dir).expect("Failed to create descendant dir");
        fs::write(descendant_dir.join("partial.bin"), b"partial")
            .expect("Failed to create descendant payload");

        cache
            .clear_model(temp_dir.path(), descendant_name)
            .expect("Expected clear to succeed");

        assert!(ancestor_dir.exists());
        assert!(ancestor_dir.join("tokenizer.json").exists());
        assert!(descendant_dir.exists());
    }

    #[test]
    fn test_ensure_model_dir_available_rejects_ancestor_descendant_overlap() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        write_cached_model(
            temp_dir.path(),
            "gs://bucket/foo/bar/baz",
            "tokenizer.json",
            b"{}",
        );

        let model = ModelName::parse("gs://bucket/foo/bar").expect("Expected model parse");
        let model_dir = model.model_dir(temp_dir.path());

        let err = (ModelDir::new(temp_dir.path(), &model_dir))
            .ensure_available()
            .expect_err("Expected overlap rejection");

        assert!(err.to_string().contains("overlaps cached descendant model"));
    }

    #[tokio::test]
    async fn test_get_model_path_returns_not_found_for_missing_cache() {
        let provider = GcsProvider;
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let result = provider
            .get_model_path(
                "gs://test-bucket/org/model/rev-3",
                temp_dir.path().to_path_buf(),
            )
            .await;
        assert!(result.is_err());
        assert!(
            result
                .expect_err("Expected not found error")
                .to_string()
                .contains("not found in cache")
        );
    }

    #[tokio::test]
    async fn test_get_model_path_rejects_empty_model_dir() {
        let provider = GcsProvider;
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_name = "gs://test-bucket/org/model/rev-4";
        let model_dir = expected_model_dir(temp_dir.path(), model_name);
        fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        (ModelDir::new(temp_dir.path(), &model_dir))
            .write_model_marker()
            .expect("Failed to write model marker");

        let result = provider
            .get_model_path(model_name, temp_dir.path().to_path_buf())
            .await;
        assert!(result.is_err());
        assert!(
            result
                .expect_err("Expected empty model dir error")
                .to_string()
                .contains("is empty")
        );
    }

    #[tokio::test]
    async fn test_get_model_path_returns_existing_model_dir() {
        let provider = GcsProvider;
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let model_name = "gs://test-bucket/org/model/rev-5";
        let model_dir = expected_model_dir(temp_dir.path(), model_name);
        fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        fs::write(model_dir.join("tokenizer.json"), b"{}").expect("Failed to create model file");
        (ModelDir::new(temp_dir.path(), &model_dir))
            .write_model_marker()
            .expect("Failed to write model marker");

        let result = provider
            .get_model_path(model_name, temp_dir.path().to_path_buf())
            .await
            .expect("Expected model path");

        assert_eq!(result, model_dir);
    }

    #[tokio::test]
    async fn test_delete_model_trait_method_removes_cached_model() {
        let provider = GcsProvider;
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let model_name = "gs://test-bucket/org/model/rev-1";
        let model_dir = expected_model_dir(temp_dir.path(), model_name);
        fs::create_dir_all(&model_dir).expect("Failed to create model dir");
        fs::write(model_dir.join("tokenizer.json"), b"{}").expect("Failed to create model file");
        (ModelDir::new(temp_dir.path(), &model_dir))
            .write_model_marker()
            .expect("Failed to write model marker");

        provider
            .delete_model(model_name, temp_dir.path().to_path_buf())
            .await
            .expect("Expected successful delete");
        assert!(!model_dir.exists());
    }

    #[tokio::test]
    async fn test_delete_model_trait_method_removes_incomplete_model_dir_without_marker() {
        let provider = GcsProvider;
        let temp_dir = TempDir::new().expect("Failed to create temp dir");

        let model_name = "gs://test-bucket/org/model/rev-1";
        let model_dir = expected_model_dir(temp_dir.path(), model_name);
        write_incomplete_cached_model(temp_dir.path(), model_name, "tokenizer.json", b"{}");

        provider
            .delete_model(model_name, temp_dir.path().to_path_buf())
            .await
            .expect("Expected successful delete");
        assert!(!model_dir.exists());
    }

    #[tokio::test]
    async fn test_with_prepared_model_dir_removes_directory_on_error() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let download_dir = temp_dir
            .path()
            .join("gcs")
            .join("partials")
            .join("partial-1");
        fs::create_dir_all(&download_dir).expect("Failed to create download dir");

        let result: Result<()> = (ModelDir::new(temp_dir.path(), &download_dir))
            .with_prepared(|| async {
                fs::write(download_dir.join("partial.bin"), b"data")
                    .expect("Failed to write partial file");
                Err(anyhow::anyhow!("boom"))
            })
            .await;

        assert!(result.is_err());
        assert!(!download_dir.exists());
    }

    #[tokio::test]
    async fn test_with_prepared_model_dir_keeps_directory_on_success() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let download_dir = temp_dir
            .path()
            .join("gcs")
            .join("partials")
            .join("partial-2");
        fs::create_dir_all(&download_dir).expect("Failed to create download dir");
        let stale_file = download_dir.join("stale.bin");
        fs::write(&stale_file, b"stale").expect("Failed to write stale file");

        let result = (ModelDir::new(temp_dir.path(), &download_dir))
            .with_prepared(|| async {
                assert!(download_dir.exists());
                assert!(!stale_file.exists());
                Ok::<usize, anyhow::Error>(7)
            })
            .await
            .expect("Expected success");

        assert_eq!(result, 7);
        assert!(download_dir.exists());
    }

    #[tokio::test]
    async fn test_with_prepared_model_dir_replaces_stale_file_path() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let stale_file_path = temp_dir.path().join("stale-file-path");
        fs::write(&stale_file_path, b"stale").expect("Failed to create stale file");

        let result = (ModelDir::new(temp_dir.path(), &stale_file_path))
            .with_prepared(|| async {
                assert!(stale_file_path.is_dir());
                Ok::<usize, anyhow::Error>(11)
            })
            .await
            .expect("Expected success");

        assert_eq!(result, 11);
        assert!(stale_file_path.is_dir());
    }
}
