"""
Registration
============

Examples
--------

>>> task = registration_syn_quick(
...     dimensionality=3,
...     fixed_image="reference.nii",
...     moving_image="structural.nii",
... )
>>> task.cmdline    # doctest: +ELLIPSIS
'antsRegistration -d 3 -o [output,outputWarped.nii.gz,outputInverseWarped.nii.gz] ...'

>>> task = registration_syn_quick(
...     dimensionality=3,
...     fixed_image="reference.nii",
...     moving_image="structural.nii", 
...     transform_type="b",
...     gradient_step=0.2,
...     spline_distance=32,
... )
>>> task.cmdline    # doctest: +ELLIPSIS
'antsRegistration ... -t BSplineSyn[0.2,32,0,3] ...'

>>> task = registration_syn_quick(
...     dimensionality=3,
...     fixed_image="reference.nii",
...     moving_image="structural.nii",
...     fixed_mask="mask.nii",
...     random_seed=42,
... )
>>> task.cmdline    # doctest: +ELLIPSIS
'antsRegistration ... -x [mask.nii,NULL] ... --random-seed 42 ...'
"""

__all__ = ["Registration", "registration_syn", "registration_syn_quick"]

from functools import partial
from os import PathLike
from typing import Optional, Sequence

from attrs import NOTHING, define, field
from pydra.engine.specs import File, ShellOutSpec, ShellSpec, SpecInfo
from pydra.engine.task import ShellCommandTask


class Registration(ShellCommandTask):
    """Task definition for antsRegistration."""

    @define(kw_only=True)
    class InputSpec(ShellSpec):
        dimensionality: int = field(
            default=3, metadata={"help_string": "image dimensionality", "argstr": "-d", "allowed_values": {2, 3, 4}}
        )

        fixed_image: PathLike = field(metadata={"help_string": "fixed image", "mandatory": True})

        moving_image: PathLike = field(metadata={"help_string": "moving image", "mandatory": True})

        output_transform_prefix: str = field(
            default="output",
            metadata={
                "help_string": "output transform prefix",
                "formatter": lambda output_transform_prefix, warped_moving_image, warped_fixed_image: (
                    f"-o {output_transform_prefix}"
                    if not all([warped_moving_image, warped_fixed_image])
                    else f"-o [{output_transform_prefix},{warped_moving_image},{warped_fixed_image}]"
                ),
            },
        )

        warped_moving_image: str = field(
            metadata={
                "help_string": "warped moving image to fixed image space",
                "output_file_template": "{moving_image}_warped",
            }
        )

        warped_fixed_image: str = field(
            metadata={
                "help_string": "warped fixed image to moving image space",
                "output_file_template": "{fixed_image}_warped",
            }
        )

        write_composite_transform: bool = field(
            default=False,
            metadata={
                "help_string": "write composite transform",
                "formatter": lambda write_composite_transform: f"-a {write_composite_transform:d}",
            },
        )

        collapse_output_transforms: bool = field(
            default=True,
            metadata={
                "help_string": "collapse output transforms",
                "formatter": lambda collapse_output_transforms: f"-z {collapse_output_transforms:d}",
            },
        )

        initialize_transforms_per_stage: bool = field(
            default=False,
            metadata={
                "help_string": "initialize linear transforms from the previous stage",
                "formatter": lambda initialize_transforms_per_stage: f"-i {initialize_transforms_per_stage:d}",
            },
        )

        interpolation: str = field(
            default="Linear",
            metadata={
                "help_string": "type of interpolation",
                "allowed_values": {
                    "Linear",
                    "NearestNeighbor",
                    "Gaussian",
                    "BSpline",
                    "CosineWindowedSinc",
                    "WelchWindowedSinc",
                    "HammingWindowedSinc",
                    "LanczosWindowedSinc",
                },
                "formatter": lambda interpolation, sigma, alpha, order: (
                    "-n {}{}".format(
                        interpolation,
                        f"[{sigma},{alpha}]"
                        if interpolation == "Gaussian"
                        else f"[{order}]"
                        if interpolation == "BSpline"
                        else "",
                    )
                ),
            },
        )

        sigma: float = field(default=1.0, metadata={"help_string": "sigma parameter for interpolation"})

        alpha: float = field(default=1.0, metadata={"help_string": "alpha parameter for interpolation"})

        order: int = field(default=3, metadata={"help_string": "order parameter for interpolation"})

        masks_: str = field(
            metadata={
                "help_string": "masks parameter",
                "readonly": True,
                "formatter": lambda fixed_mask, moving_mask: (
                    f"-x [{fixed_mask or 'NULL'},{moving_mask or 'NULL'}]" if any([fixed_mask, moving_mask]) else ""
                ),
            }
        )

        fixed_mask: PathLike = field(metadata={"help_string": "mask applied to the fixed image"})

        moving_mask: PathLike = field(metadata={"help_string": "mask applied to the moving image"})

        initial_fixed_transforms: Sequence[PathLike] = field(
            metadata={
                "help_string": "initialize composite fixed transform with these transforms",
                "formatter": lambda initial_fixed_transforms, invert_fixed_transforms: (
                    ""
                    if not initial_fixed_transforms
                    else " ".join(f"-q {x}" for x in initial_fixed_transforms)
                    if not invert_fixed_transforms
                    else " ".join(f"-q [{x},{y:d}]" for x, y in zip(initial_fixed_transforms, invert_fixed_transforms))
                ),
            }
        )

        invert_fixed_transforms: Sequence[bool] = field(
            metadata={
                "help_string": "specify which initial fixed transforms to invert",
                "requires": {"initial_fixed_transforms"},
            }
        )

        initial_moving_transforms: Sequence[PathLike] = field(
            metadata={
                "help_string": "initialize composite moving transform with these transforms",
                "formatter": lambda initial_moving_transforms, invert_moving_transforms, fixed_image, moving_image: (
                    f"-r [{fixed_image}, {moving_image}, 1]"
                    if not initial_moving_transforms
                    else " ".join(f"-r {x}" for x in initial_moving_transforms)
                    if not invert_moving_transforms
                    else " ".join(
                        f"-r [{x},{y:d}]" for x, y in zip(initial_moving_transforms, invert_moving_transforms)
                    )
                ),
            }
        )

        invert_moving_transforms: Sequence[bool] = field(
            metadata={
                "help_string": "specify which initial moving transforms to invert",
                "requires": {"initial_moving_transforms"},
            }
        )

        enable_rigid_stage = field(default=True, metadata={"help_string": "enable rigid registration stage"})

        rigid_transform: str = field(
            default="Rigid",
            metadata={
                "help_string": "transform for rigid stage",
                "allowed_values": {"Rigid", "Translation"},
                "formatter": lambda enable_rigid_stage, rigid_transform, rigid_gradient_step: (
                    f"-t {rigid_transform}[{rigid_gradient_step}]" if enable_rigid_stage else ""
                ),
            },
        )

        rigid_gradient_step: bool = field(default=0.1, metadata={"help_string": "gradient step for rigid stage"})

        rigid_metric: str = field(
            metadata={
                "help_string": "rigid metric parameter",
                "allowed_values": {"CC", "MI", "Mattes", "MeanSquares", "Demons", "GC"},
                "formatter": lambda enable_rigid_stage, rigid_metric, fixed_image, moving_image, rigid_radius, rigid_num_bins, rigid_sampling_strategy, rigid_sampling_rate: (
                    "-m {}[{},{},1,{},{},{}]".format(
                        rigid_metric,
                        fixed_image,
                        moving_image,
                        rigid_num_bins if rigid_metric in {"MI", "Mattes"} else rigid_radius,
                        rigid_sampling_strategy,
                        rigid_sampling_rate,
                    )
                    if enable_rigid_stage
                    else ""
                ),
            }
        )

        rigid_radius: int = field(default=4, metadata={"help_string": "radius for rigid stage"})

        rigid_num_bins: int = field(default=32, metadata={"help_string": "number of bins for rigid stage"})

        rigid_sampling_strategy: str = field(
            default="Regular",
            metadata={
                "help_string": "sampling strategy for rigid stage",
                "allowed_values": {"None", "Regular", "Random"},
            },
        )

        rigid_sampling_rate: float = field(default=0.25, metadata={"help_string": "sampling rate for rigid stage"})

        rigid_convergence: Sequence[int] = field(
            default=(1000, 500, 250, 0),
            metadata={
                "help_string": "convergence for rigid stage",
                "formatter": lambda enable_rigid_stage, rigid_convergence, rigid_threshold, rigid_window_size: (
                    "-c [{},{},{}]".format(
                        "x".join(str(c) for c in rigid_convergence), rigid_threshold, rigid_window_size
                    )
                    if enable_rigid_stage
                    else ""
                ),
            },
        )

        rigid_threshold: float = field(default=1e-6, metadata={"help_string": "convergence threshold for rigid stage"})

        rigid_window_size: int = field(default=10, metadata={"help_string": "convergence window size for rigid stage"})

        rigid_shrink_factors: Sequence[int] = field(
            default=(8, 4, 2, 1),
            metadata={
                "help_string": "shrink factors for rigid stage",
                "formatter": lambda enable_rigid_stage, rigid_shrink_factors: (
                    "-f {}".format("x".join(str(f) for f in rigid_shrink_factors)) if enable_rigid_stage else ""
                ),
            },
        )

        rigid_smoothing_sigmas: Sequence[int] = field(
            default=(3, 2, 1, 0),
            metadata={
                "help_string": "smoothing sigmas for rigid stage",
                "formatter": lambda enable_rigid_stage, rigid_smoothing_sigmas, rigid_smoothing_units: (
                    "-s {}{}".format("x".join(str(s) for s in rigid_smoothing_sigmas), rigid_smoothing_units)
                    if enable_rigid_stage
                    else ""
                ),
            },
        )

        rigid_smoothing_units: str = field(
            default="vox",
            metadata={"help_string": "smoothing units for rigid stage", "allowed_values": {"vox", "mm"}},
        )

        enable_affine_stage: bool = field(default=True, metadata={"help_string": "enable affine registration stage"})

        affine_transform: str = field(
            default="Affine",
            metadata={
                "help_string": "affine transform",
                "allowed_values": {"Affine", "CompositeAffine", "Similarity"},
                "formatter": lambda enable_affine_stage, affine_transform, affine_gradient_step: (
                    f"-t {affine_transform}[{affine_gradient_step}]" if enable_affine_stage else ""
                ),
            },
        )

        affine_gradient_step: bool = field(default=0.1, metadata={"help_string": "gradient step for affine stage"})

        affine_metric: str = field(
            metadata={
                "help_string": "metric parameter for affine stage",
                "allowed_values": {"CC", "MI", "Mattes", "MeanSquares", "Demons", "GC"},
                "formatter": lambda enable_affine_stage, affine_metric, fixed_image, moving_image, affine_radius, affine_num_bins, affine_sampling_strategy, affine_sampling_rate: (
                    "-m {}[{},{},1,{},{},{}]".format(
                        affine_metric,
                        fixed_image,
                        moving_image,
                        affine_num_bins if affine_metric in {"MI", "Mattes"} else affine_radius,
                        affine_sampling_strategy,
                        affine_sampling_rate,
                    )
                    if enable_affine_stage
                    else ""
                ),
            }
        )

        affine_radius: int = field(default=4, metadata={"help_string": "radius for affine stage"})

        affine_num_bins: int = field(default=32, metadata={"help_string": "number of bins for affine stage"})

        affine_sampling_strategy: str = field(
            default="Regular",
            metadata={
                "help_string": "sampling strategy for affine stage",
                "allowed_values": {"None", "Regular", "Random"},
            },
        )

        affine_sampling_rate: float = field(default=0.25, metadata={"help_string": "sampling rate for affine stage"})

        affine_convergence: Sequence[int] = field(
            default=(1000, 500, 250, 0),
            metadata={
                "help_string": "convergence for affine stage",
                "formatter": lambda enable_affine_stage, affine_convergence, affine_threshold, affine_window_size: (
                    "-c [{},{},{}]".format(
                        "x".join(str(c) for c in affine_convergence), affine_threshold, affine_window_size
                    )
                    if enable_affine_stage
                    else ""
                ),
            },
        )

        affine_threshold: float = field(
            default=1e-6, metadata={"help_string": "convergence threshold for affine stage"}
        )

        affine_window_size: int = field(
            default=10, metadata={"help_string": "convergence window size for affine stage"}
        )

        affine_shrink_factors: Sequence[int] = field(
            default=(8, 4, 2, 1),
            metadata={
                "help_string": "shrink factors for affine stage",
                "formatter": lambda enable_affine_stage, affine_shrink_factors: (
                    "-f {}".format("x".join(str(f) for f in affine_shrink_factors)) if enable_affine_stage else ""
                ),
            },
        )

        affine_smoothing_sigmas: Sequence[int] = field(
            default=(3, 2, 1, 0),
            metadata={
                "help_string": "smoothing sigmas for affine stage",
                "formatter": lambda enable_affine_stage, affine_smoothing_sigmas, affine_smoothing_units: (
                    "-s {}{}".format("x".join(str(s) for s in affine_smoothing_sigmas), affine_smoothing_units)
                    if enable_affine_stage
                    else ""
                ),
            },
        )

        affine_smoothing_units: str = field(
            default="vox",
            metadata={"help_string": "smoothing units for affine stage", "allowed_values": {"vox", "mm"}},
        )

        enable_syn_stage: str = field(default=True, metadata={"help_string": "enable SyN registration stage"})

        syn_transform: str = field(
            default="Syn",
            metadata={
                "help_string": "SyN transform",
                "allowed_values": {"GaussianDisplacementField", "SyN", "BSplineSyN"},
                "formatter": lambda enable_syn_stage, syn_transform, syn_gradient_step, syn_flow_sigma, syn_total_sigma, syn_spline_distance, syn_spline_order: (
                    "-t {}[{}]".format(
                        syn_transform,
                        f"{syn_gradient_step},{syn_spline_distance},0,{syn_spline_order}"
                        if syn_transform == "BSplineSyn"
                        else f"{syn_gradient_step},{syn_flow_sigma},{syn_total_sigma}",
                    )
                    if enable_syn_stage
                    else ""
                ),
            },
        )

        syn_gradient_step: bool = field(default=0.1, metadata={"help_string": "gradient step for SyN stage"})

        syn_flow_sigma: float = field(default=3, metadata={"help_string": "sigma for flow field in SyN stage"})

        syn_total_sigma: float = field(default=0, metadata={"help_string": "sigma for total field in SyN stage"})

        syn_spline_distance: int = field(default=26, metadata={"help_string": "spline distance for SyN stage"})

        syn_spline_order: int = field(default=3, metadata={"help_string": "spline order for SyN stage"})

        syn_metric: str = field(
            default="MI",
            metadata={
                "help_string": "metric for SyN stage",
                "allowed_values": {"CC", "MI", "Mattes", "MeanSquares", "Demons", "GC"},
                "formatter": lambda enable_syn_stage, syn_metric, fixed_image, moving_image, syn_radius, syn_num_bins, syn_sampling_strategy, syn_sampling_rate: (
                    "-m {}[{},{},1,{},{},{}]".format(
                        syn_metric,
                        fixed_image,
                        moving_image,
                        syn_num_bins if syn_metric in {"MI", "Mattes"} else syn_radius,
                        syn_sampling_strategy,
                        syn_sampling_rate,
                    )
                    if enable_syn_stage
                    else ""
                ),
            },
        )

        syn_radius: int = field(default=4, metadata={"help_string": "radius for SyN stage"})

        syn_num_bins: int = field(default=32, metadata={"help_string": "number of bins for SyN stage"})

        syn_sampling_strategy: str = field(
            default="Regular",
            metadata={
                "help_string": "sampling strategy for SyN stage",
                "allowed_values": {"None", "Regular", "Random"},
            },
        )

        syn_sampling_rate: float = field(default=0.25, metadata={"help_string": "sampling rate for SyN stage"})

        syn_convergence: Sequence[int] = field(
            default=(100, 70, 50, 20),
            metadata={
                "help_string": "convergence for SyN stage",
                "formatter": lambda enable_syn_stage, syn_convergence, syn_threshold, syn_window_size: (
                    "-c [{},{},{}]".format("x".join(str(c) for c in syn_convergence), syn_threshold, syn_window_size)
                    if enable_syn_stage
                    else ""
                ),
            },
        )

        syn_threshold: float = field(default=1e-6, metadata={"help_string": "convergence threshold for SyN stage"})

        syn_window_size: int = field(default=10, metadata={"help_string": "convergence window size for SyN stage"})

        syn_shrink_factors: Sequence[int] = field(
            default=(8, 4, 2, 1),
            metadata={
                "help_string": "shrink factors for SyN stage",
                "formatter": lambda enable_syn_stage, syn_shrink_factors: (
                    "-f {}".format("x".join(str(f) for f in syn_shrink_factors)) if enable_syn_stage else ""
                ),
            },
        )

        syn_smoothing_sigmas: Sequence[int] = field(
            default=(3, 2, 1, 0),
            metadata={
                "help_string": "smoothing sigmas for SyN stage",
                "formatter": lambda enable_syn_stage, syn_smoothing_sigmas, syn_smoothing_units: (
                    "-s {}{}".format("x".join(str(s) for s in syn_smoothing_sigmas), syn_smoothing_units)
                    if enable_syn_stage
                    else ""
                ),
            },
        )

        syn_smoothing_units: str = field(
            default="vox",
            metadata={"help_string": "smoothing units for SyN stage", "allowed_values": {"vox", "mm"}},
        )

        use_histogram_matching: bool = field(
            default=False,
            metadata={
                "help_string": "use histogram matching",
                "formatter": lambda use_histogram_matching: f"-u {use_histogram_matching:d}",
            },
        )

        winsorize_image_intensities: bool = field(
            default=False,
            metadata={
                "help_string": "winsorize image intensities",
                "formatter": lambda winsorize_image_intensities, lower_quantile, upper_quantile: (
                    f"-w [{lower_quantile},{upper_quantile}]" if winsorize_image_intensities else ""
                ),
            },
        )

        lower_quantile: float = field(default=0.005, metadata={"help_string": "lower quantile"})

        upper_quantile: float = field(default=0.995, metadata={"help_string": "upper quantile"})

        precision: str = field(
            default="double",
            metadata={
                "help_string": "use float or double precision for internal computation",
                "allowed_values": {"float", "double"},
                "formatter": lambda precision: "--float {:d}".format(precision == "float"),
            },
        )

        random_seed: int = field(metadata={"help_string": "random seed", "argstr": "--random-seed"})

        verbose: bool = field(
            default=False,
            metadata={
                "help_string": "enable verbose output",
                "formatter": lambda verbose: f"--verbose {verbose:d}",
            },
        )

    input_spec = SpecInfo(name="Input", bases=(InputSpec,))

    @define(kw_only=True)
    class OutputSpec(ShellOutSpec):
        affine_transform: File = field(
            metadata={
                "help_string": "affine transform",
                "output_file_template": "{output_transform_prefix}0GenericAffine.mat",
            }
        )

        forward_warp_field: File = field(
            metadata={
                "help_string": "forward warp field",
                "output_file_template": "{output_transform_prefix}1Warp.nii.gz",
            }
        )

        inverse_warp_field: File = field(
            metadata={
                "help_string": "inverse warp field",
                "output_file_template": "{output_transform_prefix}1InverseWarp.nii.gz",
            }
        )

    output_spec = SpecInfo(name="Output", bases=(OutputSpec,))

    executable = "antsRegistration"


def registration_syn(
    dimensionality: int,
    fixed_image: PathLike,
    moving_image: PathLike,
    is_large_image: bool = True,
    output_prefix: str = "output",
    transform_type: str = "s",
    num_bins: int = 32,
    gradient_step: float = 0.1,
    radius: int = 4,
    spline_distance: int = 26,
    fixed_mask: Optional[PathLike] = None,
    moving_mask: Optional[PathLike] = None,
    precision: str = "double",
    use_histogram_matching: bool = False,
    reproducible: bool = False,
    random_seed: Optional[int] = None,
    verbose: bool = False,
    quick: bool = False,
    **kwargs,
) -> Registration:
    """Returns a task for SyN registration.

    This function instantiates a SyN registration task with parameters mimicking the `antsRegistrationSyn` scripts
    provided by ANTs.

    Parameters
    ----------
    dimensionality : {2, 3, 4}
        Image dimensionality.
    fixed_image : path_like
        Fixed image, also referred to as source image.
    moving_image : path_like
        Moving image, also referred to as target image.
    is_large_image : bool, default=True
        Whether registration is performed on images considered "large".
        ANTs considers input images to be large if any dimension is over 256.
    output_prefix : str, default="output"
        Prefix prepended to all output files.
    transform_type : {"t", "r", "a", "s", "sr", "so", "b", "br", "bo"}, default="s"
        Type of transform for the registration:
        * t: Translation only
        * r: Rigid only
        * a: Rigid + Affine
        * s: Rigid + Affine + SyN
        * sr: Rigid + SyN
        * so: SyN only
        * b: Rigid + Affine + BSplineSyn
        * br: Rigid + BSplineSyn
        * bo: BSplineSyn only
    num_bins : int, default=32
        Number of histogram bins for the MI metric in SyN stage.
    gradient_step : float, default=0.1
        Gradient step size for the CC metric in SyN stage.
    radius : int, default=4
        Radius for the CC metric used in SyN stage.
    spline_distance : int, default=26
        Spline distance for deformable B-splines in SyN stage.
    fixed_mask : path_like, optional
        Mask applied to the fixed image space.
    moving_mask : path_like, optional
        Mask applied to the moving image space.
    precision : {"float", "double"}, default="double"
        Precision used for internal computation.
    use_histogram_matching : bool, default=True
        Perform histogram matching prior to registration.
    reproducible : bool, default=False
        Use a reproducible set of parameters,
        i.e. `GC` metric for linear stages and `CC` for SyN.
        Random seed should be specified or else a fixed value of 1 is used.
    random_seed : int, optional
        Specify a custom random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose logging.
    quick : bool, default=False
        Use a set parameters for faster convergence.
    **kwargs : dict, optional
        Extra arguments passed to the task constructor.

    Returns
    -------
    Registration
        The configured registration task.

    See Also
    --------
    pydra.tasks.ants.registration.registration_syn_quick :
        Same as `registration_syn` with `quick` enabled.
    """
    return Registration(
        dimensionality=dimensionality,
        fixed_image=fixed_image,
        moving_image=moving_image,
        output_transform_prefix=output_prefix,
        warped_moving_image=f"{output_prefix}Warped.nii.gz",
        warped_fixed_image=f"{output_prefix}InverseWarped.nii.gz",
        collapse_output_transforms=True,
        fixed_mask=fixed_mask or NOTHING,
        moving_mask=moving_mask or NOTHING,
        enable_rigid_stage=transform_type not in {"bo", "so"},
        rigid_transform="Translation" if transform_type == "t" else "Rigid",
        rigid_metric="GC" if reproducible else "MI",
        rigid_radius=1,
        rigid_num_bins=32,
        rigid_convergence=(1000, 500, 250, 0 if quick else 100),
        rigid_shrink_factors=(12, 8, 4, 2) if is_large_image else (8, 4, 2, 1),
        rigid_smoothing_sigmas=(4, 3, 2, 1) if is_large_image else (3, 2, 1, 0),
        enable_affine_stage=transform_type in {"a", "b", "s"},
        affine_transform="Affine",
        affine_metric="GC" if reproducible else "MI",
        affine_radius=1,
        affine_num_bins=32,
        affine_convergence=(1000, 500, 250, 0 if quick else 100),
        affine_shrink_factors=(12, 8, 4, 2) if is_large_image else (8, 4, 2, 1),
        affine_smoothing_sigmas=(4, 3, 2, 1) if is_large_image else (3, 2, 1, 0),
        enable_syn_stage=transform_type[0] in {"b", "s"},
        syn_transform="BSplineSyn" if transform_type[0] == "b" else "Syn",
        syn_gradient_step=gradient_step,
        syn_spline_distance=spline_distance,
        syn_metric="CC" if reproducible else "MI",
        syn_radius=radius,
        syn_num_bins=num_bins,
        syn_convergence=(
            (100, 100, 70, 50, 0 if quick else 20) if is_large_image else (100, 70, 50, 0 if quick else 20)
        ),
        syn_shrink_factors=(10, 6, 4, 2, 1) if is_large_image else (8, 4, 2, 1),
        syn_smoothing_sigmas=(5, 3, 2, 1, 0) if is_large_image else (3, 2, 1, 0),
        use_histogram_matching=use_histogram_matching,
        precision=precision,
        random_seed=random_seed or (1 if reproducible else NOTHING),
        verbose=verbose,
        **kwargs,
    )


registration_syn_quick = partial(registration_syn, quick=True)
