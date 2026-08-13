"""
Copyright (c) 2026, John Meshreki
All rights reserved.

john.meshreki@gmail.com

-----------------------------------------------------------------------------
Typed configuration models for config-driven FPM experiments.

This module defines the schema for a single experiment configuration file.
The experiment JSON is intentionally split into three sections:

- dataset:
    Describes where the acquisition lives and how files are named/read.
- setup:
    Describes the physical microscope and LED-array geometry.
- reconstruction:
    Describes algorithmic parameters for the reconstruction run.
- crop_mode:
    Describes how to resolve the active crop from the crop grid whether it is specified directly or derived from a list.

Pydantic is used so configuration mistakes fail early and clearly, before
long image loading or reconstruction begins.

Typical usage:
    from pathlib import Path
    from fpm.config_models import load_experiment_config

    cfg = load_experiment_config(Path("configs/experiment_usaf_crop82.json"))
    print(cfg.dataset.input_root)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DatasetConfig(BaseModel):
    """
    Dataset-specific configuration.

    Attributes
    ----------
    sample_name:
        Human-readable sample name.
    camera_name:
        Camera name or identifier string.    
    input_root:
        Root directory containing pre-cropped crop folders.
        This is used when ``input_mode="load_cropped_dirs"``.
    file_pattern:
        Filename pattern with placeholders ``{x}`` and ``{y}``.
    coords_file:
        Path to a JSON file containing LED coordinate pairs
        of the form ``[[x, y], ...]``.
    nused:
        Number of reordered images/LEDs to use.
    save_root:
        Root folder for saving output images and metadata.
    input_mode:
        Canonical input loading mode. Supported values are:
        - ``"load_cropped_dirs"``
        - ``"load_full_frames"``
    full_frame_input_root:
        Directory containing the original full-frame EXR images.
        This is used when ``input_mode="load_full_frames"``.

    Notes
    -----
    Legacy values for ``input_mode`` are accepted and normalized:
    - ``"cropped_dirs"`` -> ``"load_cropped_dirs"``
    - ``"full_frame"`` -> ``"load_full_frames"``
    - ``"full_frame_single_crop"`` -> ``"load_full_frames"``

    The old legacy field ``crop_dir`` is also accepted and ignored, because
    crop selection is now handled by ``load_crops``.
    """

    sample_name: str
    camera_name: str
    input_root: str
    file_pattern: str
    coords_file: str
    nused: int = 140
    save_root: str = "output"
    input_mode: Literal["load_cropped_dirs", "load_full_frames"] = "load_cropped_dirs"
    full_frame_input_root: str | None = None


    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_dataset_fields(cls, data):
        """
        Normalize legacy dataset fields before validation.

        Parameters
        ----------
        data:
            Raw input dictionary for the dataset section.

        Returns
        -------
        dict | object
            Normalized dataset dictionary.

        Notes
        -----
        This method converts old ``input_mode`` names to the new canonical
        names and drops the deprecated ``crop_dir`` field if it is present.
        """
        if not isinstance(data, dict):
            return data

        old_to_new_input_mode = {
            "cropped_dirs": "load_cropped_dirs",
            "full_frame": "load_full_frames",
            "full_frame_single_crop": "load_full_frames",
        }

        if "input_mode" in data:
            data["input_mode"] = old_to_new_input_mode.get(
                data["input_mode"],
                data["input_mode"],
            )

        # Legacy field kept in older JSON files. It is no longer used.
        data.pop("crop_dir", None)

        return data


class SetupConfig(BaseModel):
    """
    Physical microscope and illumination setup configuration.

    Attributes
    ----------
    wavelength_um:
        Illumination wavelength in micrometers.
    objective_na:
        Objective numerical aperture.
    magnification:
        Effective microscope magnification.
    camera_pixel_um:
        Camera pixel pitch in micrometers.
    patch_size:
        Reconstruction patch size ``[rows, cols]``.
    image_center:
        Reference image center ``[row, col]``.
    led_distance_um:
        Distance from the LED plane to the object plane in micrometers.
    led_coord_unit_um:
        Physical size represented by one coordinate unit in the LED coords file.
        Example: if the coords file is in pixels and each pixel is 63 um, set
        this to 63.0.
    led_coord_center:
        Center of the LED-coordinate system as ``[x_center, y_center]`` in the
        same coordinate units used by the coords file.
    led_coords_convention:
        Convention used in the coords file. Supported values:
        - ``"xy"``
        - ``"rc"``
    led_axis_signs:
        Sign convention applied after centering the LED coordinates.
        Example ``[-1, -1]`` means:
        - physical x = -(coord_x - x_center) * led_coord_unit_um
        - physical y = -(coord_y - y_center) * led_coord_unit_um
    z0:
        Defocus/propagation distance used when building H0.
    """

    wavelength_um: float
    objective_na: float
    magnification: float
    camera_pixel_um: float
    patch_size: list[int] = Field(min_length=2, max_length=2)
    image_center: list[int] = Field(min_length=2, max_length=2)

    led_distance_um: float
    led_coord_unit_um: float | None = None
    led_coord_center: list[float] | None = Field(default=None, min_length=2, max_length=2)
    led_coords_convention: Literal["xy", "rc"] = "xy"
    led_axis_signs: list[int] = Field(default=[-1, -1], min_length=2, max_length=2)

    #Keeping legacy fields for backward compatibility (for now).
    led_spacing_um: float | None = None
    led_diameter: int | None = None
    led_center_v: int | None = None
    led_center_h: int | None = None
    led_grid_shape: list[int] | None = None
    led_coord_pixel_um: float | None = None
    led_coord_center_px: list[float] | None = None

    z0: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_led_geometry_fields(cls, data):
        """
        Normalize legacy LED-geometry fields into the canonical coords-driven form.

        Legacy mappings supported
        -------------------------
        - led_coord_pixel_um -> led_coord_unit_um
        - led_coord_center_px -> led_coord_center

        Notes
        -----
        This preserves backward compatibility while the configs are being migrated.
        """
        if not isinstance(data, dict):
            return data

        if "led_coord_unit_um" not in data and "led_coord_pixel_um" in data:
            data["led_coord_unit_um"] = data["led_coord_pixel_um"]

        if "led_coord_center" not in data and "led_coord_center_px" in data:
            data["led_coord_center"] = data["led_coord_center_px"]

        return data
    
    @model_validator(mode="after")
    def validate_canonical_led_geometry_fields(self):
        """
        Ensure the canonical coords-driven LED geometry fields are available
        after legacy normalization.
        """
        if self.led_coord_unit_um is None:
            raise ValueError(
                "setup.led_coord_unit_um is required. "
                "You may provide it directly, or use legacy setup.led_coord_pixel_um."
            )

        if self.led_coord_center is None:
            raise ValueError(
                "setup.led_coord_center is required. "
                "You may provide it directly, or use legacy setup.led_coord_center_px."
            )

        return self


class ReconstructionConfig(BaseModel):
    """
    Reconstruction algorithm configuration.

    Attributes
    ----------
    tol:
        Convergence tolerance.
    max_iter:
        Maximum number of iterations.
    min_iter:
        Minimum number of iterations before monotonic stopping.
    monotone:
        Whether to stop when the error increases.
    step_size:
        Gradient-descent step size.
    op_alpha:
        Regularization/denominator offset for object update.
    op_beta:
        Regularization/denominator offset for pupil update.
    display:
        Display mode string passed through to the reconstruction code.
    mode:
        Reconstruction mode string, e.g. ``"fourier"``.
    use_default_p0:
        Whether to use the default pupil initialization.
    optimize_p_flags:
        List of pupil-optimization flags to sweep.
    pupil_scaling_factors:
        List of pupil scaling factors to sweep.
    save_iter_result:
        Kept for compatibility with the original script.
    poscalibrate:
        Kept for compatibility with the original script.
    calibrate_tol:
        Kept for compatibility with the original script.
    """

    tol: float = 0.01
    max_iter: int = 10
    min_iter: int = 1
    monotone: bool = True
    step_size: float = 0.01
    op_alpha: float = 1.0
    op_beta: float = 1e3
    display: str = "full"
    mode: Literal["fourier", "real"] = "fourier"
    use_default_p0: bool = True
    optimize_p_flags: list[bool] = Field(default_factory=lambda: [True])
    pupil_scaling_factors: list[float] = Field(default_factory=lambda: [1.0])
    save_iter_result: int = 1
    poscalibrate: int = 0
    calibrate_tol: float = 1e-1

class CropsConfig(BaseModel):
    """
    Crop configuration for the current run.

    Modes
    -----
    load:
        Use already existing crop directories such as crop_82.
    generate:
        Generate crop descriptors on the fly from explicit origins or
        from a grid/range definition.

    Selection modes
    ---------------
    single:
        Process one crop.
    list:
        Process an explicit list of crops.
    range:
        Process a contiguous range of crops.
    """

    mode: Literal["load", "generate"]
    selection: Literal["single", "list", "range"]

    # load mode
    crop_dir: str | None = None
    crop_dirs: list[str] = Field(default_factory=list)
    crop_range: list[int] | None = None

    # generate mode: explicit origins
    crop_origin_rc: list[int] | None = None
    crop_origins_rc: list[list[int]] = Field(default_factory=list)

    # generate mode: grid-based range support
    full_image_shape: list[int] | None = None
    crop_size: list[int] | None = None
    overlap: float | None = None
    max_crops: int | None = None
    indexing: Literal["row_major"] | None = "row_major"

    @model_validator(mode="after")
    def validate_crop_configuration(self):
        """
        Validate that the fields required by the selected crop mode are present.
        """
        if self.mode == "load":
            if self.selection == "single" and not self.crop_dir:
                raise ValueError("crops.crop_dir must be provided for mode='load', selection='single'.")
            if self.selection == "list" and not self.crop_dirs:
                raise ValueError("crops.crop_dirs must be provided for mode='load', selection='list'.")
            if self.selection == "range":
                if self.crop_range is None or len(self.crop_range) != 2:
                    raise ValueError("crops.crop_range must be [start, end] for mode='load', selection='range'.")

        if self.mode == "generate":
            if self.selection == "single":
                if self.crop_origin_rc is None or len(self.crop_origin_rc) != 2:
                    raise ValueError("crops.crop_origin_rc must be [row, col] for mode='generate', selection='single'.")
                if self.crop_size is None or len(self.crop_size) != 2:
                    raise ValueError("crops.crop_size must be provided for mode='generate', selection='single'.")
                if self.full_image_shape is None or len(self.full_image_shape) != 2:
                    raise ValueError("crops.full_image_shape must be provided for mode='generate', selection='single'.")

            if self.selection == "list":
                has_explicit_origins = bool(self.crop_origins_rc)
                has_grid_labels = bool(self.crop_dirs)

                if not has_explicit_origins and not has_grid_labels:
                    raise ValueError(
                        "For mode='generate', selection='list', provide either "
                        "crops.crop_origins_rc or crops.crop_dirs."
                    )

                if self.crop_size is None or len(self.crop_size) != 2:
                    raise ValueError("crops.crop_size must be provided for mode='generate', selection='list'.")

                if self.full_image_shape is None or len(self.full_image_shape) != 2:
                    raise ValueError("crops.full_image_shape must be provided for mode='generate', selection='list'.")

            if self.selection == "range":
                if self.crop_range is None or len(self.crop_range) != 2:
                    raise ValueError("crops.crop_range must be [start, end] for mode='generate', selection='range'.")
                if self.crop_size is None or len(self.crop_size) != 2:
                    raise ValueError("crops.crop_size must be provided for mode='generate', selection='range'.")
                if self.full_image_shape is None or len(self.full_image_shape) != 2:
                    raise ValueError("crops.full_image_shape must be provided for mode='generate', selection='range'.")
                if self.overlap is None:
                    raise ValueError("crops.overlap must be provided for mode='generate', selection='range'.")

        return self



class ExperimentConfig(BaseModel):
    """
    Top-level experiment configuration.

    This model groups all sections needed to run one experiment:
    - dataset
    - setup
    - reconstruction
    - crops

    Notes
    -----
    This model accepts both:
    - legacy top-level ``crop_mode``
    - newer ``load_crops`` + ``generate_crop_grid``

    Legacy ``crop_mode`` is normalized automatically into the newer split
    representation before validation.
    """

    dataset: DatasetConfig
    setup: SetupConfig
    reconstruction: ReconstructionConfig
    crops: CropsConfig
    

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_crop_sections(cls, data):
        """
        Normalize legacy crop configuration into the newer unified ``crops`` section.
        """
        if not isinstance(data, dict):
            return data

        if "crops" in data:
            return data

        if "crop_mode" in data:
            crop_mode = data.pop("crop_mode")
            data["crops"] = {
                "mode": "load",
                "selection": crop_mode.get("mode", "single"),
                "crop_dir": crop_mode.get("single_crop_dir"),
                "crop_dirs": crop_mode.get("crop_dirs", []),
                "crop_range": crop_mode.get("crop_range"),
                "full_image_shape": crop_mode.get("full_image_shape"),
                "crop_size": crop_mode.get("crop_size"),
                "overlap": crop_mode.get("overlap"),
                "max_crops": crop_mode.get("max_crops", 150),
                "indexing": crop_mode.get("crop_indexing", "row_major"),
            }
            return data

        if "load_crops" in data or "generate_crop_grid" in data:
            load_crops = data.pop("load_crops", {})
            generate_crop_grid = data.pop("generate_crop_grid", {})

            data["crops"] = {
                "mode": "load",
                "selection": load_crops.get("mode", "single"),
                "crop_dir": load_crops.get("crop_dir"),
                "crop_dirs": load_crops.get("crop_dirs", []),
                "crop_range": load_crops.get("crop_range"),
                "full_image_shape": generate_crop_grid.get("full_image_shape"),
                "crop_size": generate_crop_grid.get("crop_size"),
                "overlap": generate_crop_grid.get("overlap"),
                "max_crops": generate_crop_grid.get("max_crops", 150),
                "indexing": generate_crop_grid.get("indexing", "row_major"),
            }

        return data


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """
    Load and validate an experiment JSON file.

    Parameters
    ----------
    path:
        Path to the experiment JSON file.

    Returns
    -------
    ExperimentConfig
        Validated and normalized experiment configuration.

    Notes
    -----
    This function accepts both legacy and newer config naming conventions.
    The returned object always uses the canonical newer internal structure.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return ExperimentConfig.model_validate(raw)