"""
Copyright (c) 2026, John Meshreki
All rights reserved.

john.meshreki@gmail.com

-----------------------------------------------------------------------------
Input/output helpers for the FPM experiment runner.

This module provides utilities for:
- loading coordinate JSON files
- loading EXR images efficiently
- building the input image stack from a coordinate list

The EXR loader reads only the green channel, which matches the current
single-channel reconstruction workflow and avoids unnecessary decoding
work for the unused RGB channels.

Design notes
------------
This module deliberately keeps image decoding separate from the optical
geometry and reconstruction logic so that future changes—such as caching,
multi-worker loading, or alternative file formats—can be added without
touching the reconstruction code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import OpenEXR
import Imath
import torch

from .config_models import DatasetConfig


def load_coords(coords_file: str | Path) -> torch.Tensor:
    """
    Load LED coordinates from a JSON file.

    Parameters
    ----------
    coords_file:
        Path to a JSON file containing a list of [x, y] LED coordinates.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 2) with integer coordinates.
    """
    coords_file = Path(coords_file)
    with coords_file.open("r", encoding="utf-8") as f:
        coords = json.load(f)
    return torch.tensor(coords, dtype=torch.int64)


def load_exr_green(
    file_path: str | Path,
    device: str | torch.device = "cpu",
    allow_single_channel_gray: bool = True,
) -> torch.Tensor:
    """
    Load the green channel from an EXR image.

    This function first tries to read the ``G`` channel. If the EXR file does
    not contain a green channel and contains only one channel, it can
    optionally read that single available channel as a grayscale image.

    Parameters
    ----------
    file_path : str or Path
        Path to the EXR file.
    device : str or torch.device, optional
        Output device. Use ``"cpu"`` for loading many images and move the final
        stacked tensor to GPU in one shot.
    allow_single_channel_gray : bool, optional
        If True and the EXR file contains exactly one channel, read that
        channel as a grayscale image when ``G`` is not available.
        If False, the function raises an error when ``G`` is missing.

    Returns
    -------
    torch.Tensor
        Image tensor of shape ``(rows, cols)``, dtype ``float32``.

    Raises
    ------
    ValueError
        If the EXR file does not contain a green channel and grayscale fallback
        is disabled, or if the channel layout is unsupported.
    KeyError
        If the requested channel cannot be found.
    """
    # Convert the input path to string because OpenEXR expects a string path.
    file_path = str(file_path)

    # Open the EXR file and read image dimensions.
    exr_file = OpenEXR.InputFile(file_path)
    dw = exr_file.header()["dataWindow"]
    h = dw.max.y - dw.min.y + 1
    w = dw.max.x - dw.min.x + 1

    # Read the channel list from the EXR header.
    channel_names = list(exr_file.header()["channels"].keys())

    # Define the EXR pixel type to read 32-bit floating-point data.
    float_pt = Imath.PixelType(Imath.PixelType.FLOAT)

    # Case 1:
    # Prefer the green channel if it exists.
    if "G" in channel_names:
        img_buffer = exr_file.channel("G", float_pt)

    # Case 2:
    # If there is no green channel but only one channel exists, optionally
    # read that channel as a grayscale image.
    elif allow_single_channel_gray and len(channel_names) == 1:
        gray_channel_name = channel_names[0]
        img_buffer = exr_file.channel(gray_channel_name, float_pt)

    # Case 3:
    # Unsupported layout for this function.
    else:
        raise ValueError(
            f"Could not load green channel from EXR file: {file_path}. "
            f"Available channels: {channel_names}. "
            f"If this is a single-channel EXR, enable "
            f"'allow_single_channel_gray=True'."
        )

    # Convert the raw EXR buffer into a NumPy array and reshape it into 2D.
    img = np.frombuffer(img_buffer, dtype=np.float32).reshape(h, w)

    # Copy the NumPy array before converting to torch to ensure safe memory
    # ownership, then move it to the requested device.
    return torch.from_numpy(img.copy()).to(device, non_blocking=True)

def subtract_black_level(
    img: torch.Tensor,
    camera: str | None = None,
    black_level: float | None = None,
    clip_negative: bool = True,
) -> torch.Tensor:
    """
    Subtract the camera black level from an image.

    This function removes a constant black-level offset from an image tensor.
    If `black_level` is not provided, the function can choose a default value
    based on the camera name. For example, passing `camera="orcaflash"`
    uses a black level of 100.

    Parameters
    ----------
    img : torch.Tensor
        Input image tensor. The image can be 2D or higher-dimensional.
        The subtraction is applied element-wise.
    camera : str or None, optional
        Name of the camera used to acquire the image. If set to `"orcaflash"`
        and `black_level` is not provided, the default black level is set to 100.
        The camera name comparison is case-insensitive.
    black_level : float or None, optional
        Black-level value to subtract from the image. If provided, this value
        overrides any camera-specific default.
    clip_negative : bool, optional
        If True, negative values after black-level subtraction are clipped to 0.
        This is usually recommended for image intensity data.

    Returns
    -------
    torch.Tensor
        Image after black-level subtraction.

    Raises
    ------
    ValueError
        If neither `black_level` nor a supported `camera` name is provided.
    TypeError
        If `img` is not a torch.Tensor.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError(f"`img` must be a torch.Tensor, but got {type(img).__name__}.")

    if black_level is None:
        if camera is None:
            raise ValueError(
                "Please provide either `black_level` or a supported `camera` name."
            )

        camera_key = camera.strip().lower()

        if camera_key == "orcaflash":
            black_level = 100.0
        else:
            raise ValueError(
                f"Unsupported camera '{camera}'. "
                "Currently supported cameras: 'orcaflash'."
            )

    # Convert black level to the same dtype and device as the input image.
    black_level_tensor = torch.tensor(
        black_level,
        dtype=img.dtype,
        device=img.device,
    )

    # Subtract the black-level offset from every pixel.
    corrected_img = img - black_level_tensor

    # Intensity values should usually remain non-negative after correction.
    if clip_negative:
        corrected_img = torch.clamp(corrected_img, min=0)

    return corrected_img

def load_crop_dir_stack(
    dataset_cfg: DatasetConfig,
    coords: torch.Tensor,
    crop_dir: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:    
    """
    Load the full image stack and background vector.

    Parameters
    ----------
    dataset_cfg:
        Dataset configuration section.
    coords:
        Tensor of LED coordinates of shape (N, 2).
    crop_dir:
        Directory containing the crop images.
    device:
        Target device for the final stacked image tensor.

    Returns
    -------
    Iall:
        Tensor of shape (rows, cols, Nimg).
    Ibk:
        Background tensor of shape (Nimg,).
    """
    nimg = coords.shape[0]
    imgs: list[torch.Tensor] = []
    ibk = torch.zeros(nimg, dtype=torch.float32)

    crop_path = Path(dataset_cfg.input_root) / crop_dir

    for m in range(nimg):
        xval = int(coords[m, 0].item())
        yval = int(coords[m, 1].item())

        filename = dataset_cfg.file_pattern.format(x=xval, y=yval)
        full_path = crop_path / filename

        img = load_exr_green(full_path, device="cpu")

        # Remove black level from the image before stacking
        img = subtract_black_level(img, camera=dataset_cfg.camera_name)

        imgs.append(img)

        bk1 = 0.0
        bk2 = 0.0
        ibk[m] = (bk1 + bk2) / 2.0
        if ibk[m] > 300 and m > 0:
            ibk[m] = ibk[m - 1]

    iall = torch.stack(imgs, dim=2).to(device, non_blocking=True)
    ibk = ibk.to(device, non_blocking=True)
    return iall, ibk

def crop_tensor(
    image: torch.Tensor,
    origin_rc: tuple[int, int],
    crop_size: list[int],
) -> torch.Tensor:
    """
    Extract a crop from a 2D image tensor.

    Parameters
    ----------
    image:
        Full-frame image tensor of shape ``(rows, cols)``.
    origin_rc:
        Top-left crop origin as ``(row0, col0)`` in zero-based indexing.
    crop_size:
        Crop size as ``[rows, cols]``.

    Returns
    -------
    torch.Tensor
        Cropped tensor of shape ``(crop_rows, crop_cols)``.

    Raises
    ------
    ValueError
        If the requested crop lies outside the image bounds.
    """
    row0, col0 = origin_rc
    crop_rows, crop_cols = crop_size
    row1 = row0 + crop_rows
    col1 = col0 + crop_cols

    if row0 < 0 or col0 < 0 or row1 > image.shape[0] or col1 > image.shape[1]:
        raise ValueError(
            f"Requested crop {(row0, col0, row1, col1)} is outside image shape {tuple(image.shape)}"
        )

    return image[row0:row1, col0:col1]



def load_crop_stack_from_full_frames(
    dataset_cfg: DatasetConfig,
    coords: torch.Tensor,
    crop_desc: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load the active crop on the fly from a stack of full-frame EXR images.

    Parameters
    ----------
    dataset_cfg:
        Dataset configuration. The field ``dataset_cfg.input_root in load_full_frames mode`` must be
        set when this function is used.
    coords:
        Tensor of LED coordinates of shape ``(N, 2)``.
    crop_desc:
        Resolved crop descriptor for the active crop. It must contain:
        - ``origin_rc``: top-left crop origin as ``(row0, col0)``
        - ``crop_size``: crop size as ``[rows, cols]``
    device:
        Target device for the final stacked crop tensor.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple ``(Iall, Ibk)`` where:
        - ``Iall`` has shape ``(crop_rows, crop_cols, Nimg)``
        - ``Ibk`` has shape ``(Nimg,)``

    Notes
    -----
    This function reads each full-frame EXR image from disk, extracts the
    active crop defined by ``crop_desc``, stores the cropped images on CPU,
    and transfers the final stacked tensor to the target device in one shot.

    The function is crop-driven, not single-crop-specific: the active crop is
    determined entirely by ``crop_desc`` and therefore works with single-crop,
    crop-list, and crop-range execution modes.
    """

    nimg = coords.shape[0]
    imgs: list[torch.Tensor] = []
    ibk = torch.zeros(nimg, dtype=torch.float32)

    full_root = Path(dataset_cfg.input_root)
    origin_rc = tuple(crop_desc["origin_rc"])
    crop_size = crop_desc["crop_size"]

    for m in range(nimg):
        xval = int(coords[m, 0].item())
        yval = int(coords[m, 1].item())

        filename = dataset_cfg.file_pattern.format(x=xval, y=yval)
        full_path = full_root / filename

        full_img = load_exr_green(full_path, device="cpu")

        # Remove black level from the image before stacking
        full_img = subtract_black_level(full_img, camera=dataset_cfg.camera_name)

        crop_img = crop_tensor(full_img, origin_rc=origin_rc, crop_size=crop_size)
        imgs.append(crop_img)

        bk1 = 0.0
        bk2 = 0.0
        ibk[m] = (bk1 + bk2) / 2.0
        if ibk[m] > 300 and m > 0:
            ibk[m] = ibk[m - 1]

    iall = torch.stack(imgs, dim=2).to(device, non_blocking=True)
    ibk = ibk.to(device, non_blocking=True)
    return iall, ibk


def load_active_crop_stack(
    dataset_cfg: DatasetConfig,
    coords: torch.Tensor,
    crop_desc: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load the active crop stack using the configured dataset input mode.

    Parameters
    ----------
    dataset_cfg:
        Dataset configuration.
    coords:
        Tensor of LED coordinates of shape ``(N, 2)``.
    crop_desc:
        Resolved crop descriptor for the active crop.
    device:
        Target device for the final stacked tensor.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        A tuple ``(Iall, Ibk)`` for the active crop.

    Raises
    ------
    ValueError
        If the input mode is unsupported.
    """
    if dataset_cfg.input_mode == "load_cropped_dirs":
        return load_crop_dir_stack(
            dataset_cfg=dataset_cfg,
            coords=coords,
            crop_dir=crop_desc["crop_dir"],
            device=device,
        )

    if dataset_cfg.input_mode == "load_full_frames":
        return load_crop_stack_from_full_frames(
            dataset_cfg=dataset_cfg,
            coords=coords,
            crop_desc=crop_desc,
            device=device,
        )

    raise ValueError(f"Unsupported dataset input_mode: {dataset_cfg.input_mode}")