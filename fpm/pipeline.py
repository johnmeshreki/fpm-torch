"""
Copyright (c) 2026, John Meshreki
All rights reserved.

john.meshreki@gmail.com

------------------------------------------------------------------------------
Top-level FPM experiment pipeline.
This module defines the main function `run_experiment` that executes 
a full FPM reconstruction for one active crop, given the experiment configuration and crop descriptor.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config_models import ExperimentConfig
from .geometry import compute_setup_tensors
from .io import load_coords, load_active_crop_stack
from .logger import FPMLoggerObject
from fpm.reconstruction import AlterMin
from fpm.utils import (
    F,
    angle_to_uint16,
    padarray,
    save_u16_png,
    to_uint16_minmax,
    save_png_matlab_imwrite_like,
    save_exr_gray_as_rgb,
)


def run_experiment(
    cfg: ExperimentConfig,
    device: torch.device,
    run_dir: str | Path,
    crop_desc: dict,
) -> None:
    """
    Run a full FPM reconstruction experiment for one active crop.

    Parameters
    ----------
    cfg:
        Validated experiment configuration.
    device:
        PyTorch device used for tensor storage and reconstruction.
    run_dir:
        Timestamped output directory created at the start of the run.
    crop_desc:
        Resolved crop descriptor for the active crop. This dictionary must
        contain:
        - ``crop_dir``: crop directory name
        - ``crop_index``: integer crop index
        - ``origin_rc``: top-left crop origin in the full image
        - ``crop_size``: crop size as ``[rows, cols]``

    Notes
    -----
    In this step, the crop descriptor is passed through mainly for
    bookkeeping and future geometry updates. The reconstruction math is
    intentionally left unchanged.
    """
    run_dir = Path(run_dir)

    FPMLoggerObject.log.info("Loading coordinates from: %s", cfg.dataset.coords_file)
    coords = load_coords(cfg.dataset.coords_file)
    nimg = coords.shape[0]

    FPMLoggerObject.log.info("Sample: %s", cfg.dataset.sample_name)
    FPMLoggerObject.log.info("Camera: %s", cfg.dataset.camera_name)
    FPMLoggerObject.log.info("Number of images from coords: %s", nimg)
    FPMLoggerObject.log.info("Input root: %s", cfg.dataset.input_root)
    active_crop_dir = crop_desc["crop_dir"]
    FPMLoggerObject.log.info("Crop directory: %s", active_crop_dir)

    iall, ibk = load_active_crop_stack(
    dataset_cfg=cfg.dataset,
    coords=coords,
    crop_desc=crop_desc,
    device=device,
    )
    FPMLoggerObject.log.info("Finished loading images")
    FPMLoggerObject.log.info("Loaded stack shape: %s", tuple(iall.shape))
    FPMLoggerObject.log.info("Background vector shape: %s", tuple(ibk.shape))

    npatch = crop_desc["crop_size"]
    imea = iall

    FPMLoggerObject.log.info("Patch size from crop descriptor: %s", npatch)
    FPMLoggerObject.log.info("Measured image tensor shape: %s", tuple(imea.shape))

    setup = compute_setup_tensors(
        cfg.setup,
        crop_desc=crop_desc,
        full_image_shape=cfg.crops.full_image_shape,
        coords=coords,
    )

    if setup["idx_u_used"].numel() != nimg:
        raise ValueError("idx_u_used length does not match number of images")

    if setup["idx_v_used"].numel() != nimg:
        raise ValueError("idx_v_used length does not match number of images")

    if setup["illumination_na_used"].numel() != nimg:
        raise ValueError("illumination_na_used length does not match number of images")

    FPMLoggerObject.log.info(
    "Crop-aware image center (um): row=%s, col=%s",
    setup["img_center"][0],
    setup["img_center"][1],
    )
    FPMLoggerObject.log.info(
    "Full image shape used for geometry: %s",
    cfg.crops.full_image_shape,
    )
    FPMLoggerObject.log.info(
        "Crop origin used for geometry: row=%s, col=%s",
        crop_desc["origin_rc"][0],
        crop_desc["origin_rc"][1],
    )

    FPMLoggerObject.log.info("lambda: %s", setup["lambda_"])
    FPMLoggerObject.log.info("NA: %s", setup["na"])
    FPMLoggerObject.log.info("mag: %s", setup["mag"])
    FPMLoggerObject.log.info("NBF: %s", setup["NBF"])
    FPMLoggerObject.log.info("um_p: %s", setup["um_p"])
    FPMLoggerObject.log.info("dx0_p: %s", setup["dx0_p"])
    FPMLoggerObject.log.info("synthetic NA is %s", setup["um_p"] * setup["lambda_"])
    FPMLoggerObject.log.info("N_obj: %s", setup["N_obj"])

    numlit = 1

    idx_u_used = setup["idx_u_used"]
    idx_v_used = setup["idx_v_used"]
    illumination_na_used = setup["illumination_na_used"]

    if (
    idx_u_used.numel() != nimg
    or idx_v_used.numel() != nimg
    or illumination_na_used.numel() != nimg
    ):
        raise ValueError(
            "Used LED geometry size mismatch: "
            f"len(idx_u_used)={idx_u_used.numel()}, "
            f"len(idx_v_used)={idx_v_used.numel()}, "
            f"len(illumination_na_used)={illumination_na_used.numel()}, "
            f"nimg={nimg}"
        )
    
    FPMLoggerObject.log.info("Used LED count from geometry: %s", setup["nled_used"])

    idx_led = torch.argsort(illumination_na_used, stable=True)

    nsh_lit = idx_u_used.reshape(1, nimg)
    nsv_lit = idx_v_used.reshape(1, nimg)

    ns = torch.zeros((numlit, nimg, 2), dtype=torch.int64)
    ns[:, :, 0] = nsv_lit
    ns[:, :, 1] = nsh_lit

    FPMLoggerObject.log.info("Constructed Ns tensor with shape: %s", tuple(ns.shape))

    if idx_led.numel() != imea.shape[2]:
        raise ValueError(
            f"LED sorting index length mismatch: len(idx_led)={idx_led.numel()} "
            f"but imea has {imea.shape[2]} images."
        )
    imea_reorder = imea[:, :, idx_led]
    ibk_reorder = ibk[idx_led]

    ithresh_reorder = imea_reorder.clone()
    for m in range(nimg):
        itmp = ithresh_reorder[:, :, m] - ibk_reorder[m]
        itmp[itmp < 0] = 0
        ithresh_reorder[:, :, m] = itmp

    ns_reorder = ns[:, idx_led, :]

    nused = cfg.dataset.nused
    idx_used = torch.arange(nused)

    I = ithresh_reorder[:, :, idx_used]
    Ns2 = ns_reorder[:, idx_used, :]

    FPMLoggerObject.log.info("Using first %s reordered images/LEDs", nused)
    FPMLoggerObject.log.info("I shape: %s", tuple(I.shape))
    FPMLoggerObject.log.info("Ns2 shape: %s", tuple(Ns2.shape))

    P0 = setup["w_NA"]
    Ps = setup["w_NA"]

    O0 = F(torch.sqrt(ithresh_reorder[:, :, 0]))
    pad_y = (setup["N_obj"][0] - npatch[0]) // 2
    pad_x = (setup["N_obj"][1] - npatch[1]) // 2
    O0 = padarray(O0, setup["N_obj"])

    FPMLoggerObject.log.info("Initialized O0 with shape: %s", tuple(O0.shape))
    FPMLoggerObject.log.info("Initialized P0 with shape: %s", tuple(P0.shape))
    FPMLoggerObject.log.info("Pad values used for O0: pad_y=%s, pad_x=%s", pad_y, pad_x)

    for pupil_scale_fac in cfg.reconstruction.pupil_scaling_factors:
        for optimize_P in cfg.reconstruction.optimize_p_flags:
            optimize_str = "pupil_optimised" if optimize_P else "pupil_not_optimised"
            subfolder_name = f"pupil_scale_{pupil_scale_fac}_{optimize_str}"

            FPMLoggerObject.log.info("--- Running: %s ---", subfolder_name)

            O, P, dirac_cen, err_pc, c, Ns_cal = AlterMin(
                I,
                [setup["N_obj"][0], setup["N_obj"][1]],
                torch.round(Ns2),
                P0,
                O0,
                cfg.reconstruction.use_default_p0,
                pupil_scale_fac,
                optimize_P,
                cfg.reconstruction.mode,
                nimg,
                cfg.reconstruction.tol,
                cfg.reconstruction.max_iter,
                cfg.reconstruction.min_iter,
                cfg.reconstruction.monotone,
                cfg.reconstruction.step_size,
                cfg.reconstruction.op_alpha,
                cfg.reconstruction.op_beta,
                cfg.reconstruction.display,
                Ps,
                device,
            )

            np_iter = (
            f"Np_{npatch[0]}_{npatch[1]}_"
            f"minIter_{cfg.reconstruction.min_iter}_"
            f"maxIter_{cfg.reconstruction.max_iter}"
            )

            crop_output_dir = run_dir / crop_desc["crop_dir"]
            run_folder = crop_output_dir / subfolder_name / np_iter
            run_folder.mkdir(parents=True, exist_ok=True)

            FPMLoggerObject.log.info("Crop output directory: %s", crop_output_dir)
            FPMLoggerObject.log.info("Saving outputs to: %s", run_folder)

            angle_O = torch.angle(O).detach().cpu().numpy()
            abs_O = torch.abs(O).detach().cpu().numpy()
            angle_P = torch.angle(P).detach().cpu().numpy()
            abs_P = torch.abs(P).detach().cpu().numpy()

            plt.figure()
            plt.imshow(angle_O, cmap="coolwarm", vmin=-1.0, vmax=1.0)
            plt.title("angle(O) near 0 (±1.0 rad)")
            plt.colorbar()
            plt.show()

            plt.figure()
            plt.imshow(abs_O, cmap="gray")
            plt.title("abs (O)")
            plt.colorbar()
            plt.show()

            plt.figure()
            plt.plot(torch.arange(1, len(err_pc) + 1).cpu().numpy(), np.array(err_pc))
            plt.title("err/iter")
            plt.xlabel("iter")
            plt.ylabel("err")
            plt.show()

            plt.figure()
            plt.imshow(angle_P, cmap="coolwarm", vmin=-0.05, vmax=0.05)
            plt.title("angle(P) near 0 (±0.05 rad)")
            plt.colorbar()
            plt.show()

            abs_P = np.clip(abs_P, 0, 1)
            plt.figure()
            plt.imshow(abs_P, cmap="gray")
            plt.title("abs (P)")
            plt.colorbar()
            plt.show()

            plt.figure()
            plt.plot(np.arange(1, len(err_pc) + 1), np.log(np.array(err_pc)))
            plt.title("log(err)/iter")
            plt.xlabel("iter")
            plt.ylabel("log(err)")
            plt.show()

            # ------------------------------------------------------------------
            # 1) Keep current Python-specific 16-bit PNG outputs
            # ------------------------------------------------------------------
            angleO_u16 = angle_to_uint16(torch.angle(O))
            angleP_u16 = angle_to_uint16(torch.angle(P))
            absO_u16 = to_uint16_minmax(torch.abs(O))
            absP_u16 = to_uint16_minmax(torch.abs(P))

            save_u16_png(run_folder / "angle_O_u16.png", angleO_u16)
            save_u16_png(run_folder / "abs_O_u16.png", absO_u16)
            save_u16_png(run_folder / "angle_P_u16.png", angleP_u16)
            save_u16_png(run_folder / "abs_P_u16.png", absP_u16)

            # ------------------------------------------------------------------
            # 2) Save raw floating-point EXR outputs to match MATLAB intent
            #    MATLAB code replicates grayscale to RGB before EXR writing.
            # ------------------------------------------------------------------
            save_exr_gray_as_rgb(run_folder / f"angle_O_{np_iter}_image.exr", angle_O)
            save_exr_gray_as_rgb(run_folder / f"abs_O_{np_iter}_image.exr", abs_O)
            save_exr_gray_as_rgb(run_folder / f"angle_P_{np_iter}_image.exr", angle_P)
            save_exr_gray_as_rgb(run_folder / f"abs_P_{np_iter}_image.exr", abs_P)

            # ------------------------------------------------------------------
            # 3) Save MATLAB-like PNG outputs
            #    This mimics imwrite(double_image, "file.png"):
            #    assumes [0,1], clips outside, scales to 8-bit.
            # ------------------------------------------------------------------
            save_png_matlab_imwrite_like(run_folder / f"angle_O_{np_iter}_image.png", angle_O)
            save_png_matlab_imwrite_like(run_folder / f"abs_O_{np_iter}_image.png", abs_O)
            save_png_matlab_imwrite_like(run_folder / f"angle_P_{np_iter}_image.png", angle_P)
            save_png_matlab_imwrite_like(run_folder / f"abs_P_{np_iter}_image.png", abs_P)

            # # ------------------------------------------------------------------
            # # 4) Save exact numeric arrays for one-to-one debugging
            # # ------------------------------------------------------------------
            # save_npy(run_folder / f"angle_O_{np_iter}.npy", angle_O)
            # save_npy(run_folder / f"abs_O_{np_iter}.npy", abs_O)
            # save_npy(run_folder / f"angle_P_{np_iter}.npy", angle_P)
            # save_npy(run_folder / f"abs_P_{np_iter}.npy", abs_P)

            FPMLoggerObject.log.info("Saved 16-bit PNGs to: %s", run_folder)
            FPMLoggerObject.log.info("finished Altermin.m")
            FPMLoggerObject.log.info("Getting/saving low res intensities (skipped in this version)")
       
            # ------------------------------------------------------------------
            # 5) Save crop run metadata for future reference and blending
            crop_run_metadata = {
                "crop_dir": crop_desc["crop_dir"],
                "crop_index": crop_desc["crop_index"],
                "origin_rc": list(crop_desc["origin_rc"]),
                "crop_size": list(crop_desc["crop_size"]),
                "overlap": cfg.crops.overlap,
                "full_image_shape": list(cfg.crops.full_image_shape) if cfg.crops.full_image_shape is not None else None,
                "input_mode": cfg.dataset.input_mode,
                "reconstruction_mode": cfg.reconstruction.mode,
                "pupil_scale_fac": pupil_scale_fac,
                "optimize_P": bool(optimize_P),
                "np_iter": np_iter,
                "run_folder": str(run_folder),
                "object_shape": list(O.shape),
                "pupil_shape": list(P.shape),
                "nused": int(cfg.dataset.nused),
                "nled": int(setup["nled_used"]),
                "saved_files": {
                    "angle_O_u16_png": str(run_folder / "angle_O_u16.png"),
                    "abs_O_u16_png": str(run_folder / "abs_O_u16.png"),
                    "angle_P_u16_png": str(run_folder / "angle_P_u16.png"),
                    "abs_P_u16_png": str(run_folder / "abs_P_u16.png"),
                    "angle_O_png": str(run_folder / f"angle_O_{np_iter}_image.png"),
                    "abs_O_png": str(run_folder / f"abs_O_{np_iter}_image.png"),
                    "angle_P_png": str(run_folder / f"angle_P_{np_iter}_image.png"),
                    "abs_P_png": str(run_folder / f"abs_P_{np_iter}_image.png"),
                    "angle_O_exr": str(run_folder / f"angle_O_{np_iter}_image.exr"),
                    "abs_O_exr": str(run_folder / f"abs_O_{np_iter}_image.exr"),
                    "angle_P_exr": str(run_folder / f"angle_P_{np_iter}_image.exr"),
                    "abs_P_exr": str(run_folder / f"abs_P_{np_iter}_image.exr"),
                },
            }
            
            FPMLoggerObject.log.info("Saving crop run metadata to JSON for future reference and belding")
            FPMLoggerObject.save_json_config(
                crop_run_metadata,
                filename=f"{crop_desc['crop_dir']}_{subfolder_name}_{np_iter}_metadata.json",
            )

            FPMLoggerObject.log.info("processing completes")

            FPMLoggerObject.log.info(
                "This run used measured & estimated images of size %s x %s.",
                npatch[0],
                npatch[1],
            )
            FPMLoggerObject.log.info("--------------------")
            FPMLoggerObject.log.info("Parameters used for these computed images:")
            FPMLoggerObject.log.info("Np: %s x %s.", npatch[0], npatch[1])
            FPMLoggerObject.log.info("minIter: %s.", cfg.reconstruction.min_iter)
            FPMLoggerObject.log.info("maxIter: %s.", cfg.reconstruction.max_iter)