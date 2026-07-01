"""Flood-event extraction, quality control, and visualization utilities."""

from .flood_segment_extractor import FloodSegmentExtractor, identify_segments_vectorized
from .flood_segment_processor import FloodSegmentProcessor, ProcessingConfig
from .flood_segment_integrator import FloodSegmentIntegrator, IntegratorConfig
from .visualize_events import (
    visualize_flood_events,
    visualize_sensors,
    visualize_segments_by_sensor,
    visualize_removed_segments,
    visualize_interpolation_comparison,
)
from .utils import (
    load_segments,
    get_sensor_segments,
    get_segment,
    list_sensors,
    get_segment_stats,
    get_sensor_stats,
    get_summary_table,
    filter_by_duration,
    visualize_segments,
    DEFAULT_SEGMENTS_PATH,
)

__all__ = [
    "FloodSegmentExtractor",
    "identify_segments_vectorized",
    "FloodSegmentProcessor",
    "ProcessingConfig",
    "FloodSegmentIntegrator",
    "IntegratorConfig",
    "visualize_flood_events",
    "visualize_sensors",
    "visualize_segments_by_sensor",
    "visualize_removed_segments",
    "visualize_interpolation_comparison",
    "load_segments",
    "get_sensor_segments",
    "get_segment",
    "list_sensors",
    "get_segment_stats",
    "get_sensor_stats",
    "get_summary_table",
    "filter_by_duration",
    "visualize_segments",
    "DEFAULT_SEGMENTS_PATH",
]
