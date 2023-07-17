"""
Registration
============

Examples
--------

>>> task = RegistrationSyNQuick(
...     dimensionality=3,
...     fixed_image="reference.nii",
...     moving_image="structural.nii",
... )
>>> task.cmdline    # doctest: +ELLIPSIS
'antsRegistrationSyNQuick.sh -d 3 -f reference.nii -m structural.nii ...'

>>> task = RegistrationSyNQuick(
...     dimensionality=3,
...     fixed_image="reference.nii",
...     moving_image="structural.nii", 
...     transform_type="b",
...     spline_distance=32,
...     gradient_step_size=0.2,
...     random_seed=42,
... )
>>> task.cmdline    # doctest: +ELLIPSIS
'antsRegistrationSyNQuick.sh ... -t b -s 32 -g 0.2 ... -e 42'

>>> task = RegistrationSyNQuick(
...     dimensionality=3,
...     fixed_image="reference.nii",
...     moving_image="structural.nii",
...     fixed_mask="mask.nii",
... )
>>> task.cmdline    # doctest: +ELLIPSIS
'antsRegistrationSyNQuick.sh ... -x mask.nii ...'
"""

__all__ = ["RegistrationSyNQuick"]

from os import PathLike
from typing import Sequence

from attrs import NOTHING, define, field
from pydra.engine.specs import File, ShellOutSpec, ShellSpec, SpecInfo
from pydra.engine.task import ShellCommandTask


class Registration(ShellCommandTask):
    """Task definition for antsRegistration."""

    @define(kw_only=True)
    class InputSpec(ShellSpec):
        dimensionality: int = field(
            metadata={"help_string": "force image dimensionality", "argstr": "-d", "allowed_values": {2, 3, 4}}
        )

        fixed_image: PathLike = field(metadata={"help_string": "fixed image", "mandatory": True})

        moving_image: PathLike = field(metadata={"help_string": "moving image", "mandatory": True})

        output_transform_prefix: str = field(
            default="output",
            metadata={
                "help_string": "output transform prefix",
                "formatter": lambda output_transform_prefix, warped_fixed_image, warped_moving_image: (
                    f"-o [{output_transform_prefix}, {warped_fixed_image}, {warped_moving_image}]"
                ),
            },
        )

        warped_fixed_image: str = field(
            metadata={
                "help_string": "warped fixed image to moving image space",
                "output_file_template": "{fixed_image}_warped",
            }
        )

        warped_moving_image: str = field(
            metadata={
                "help_string": "warped moving image to fixed image space",
                "output_file_template": "{moving_image}_warped",
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
            default=True,
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
                        f"[{sigma}, {alpha}]"
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

        enable_rigid_stage = field(default=True, metadata={"help_string": "enable rigid stage"})

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

        rigid_metric_: str = field(
            metadata={
                "help_string": "rigid metric parameter",
                "readonly": True,
                "formatter": lambda enable_rigid_stage, rigid_metric_name, fixed_image, moving_image, rigid_metric_parameter, rigid_sampling_strategy, rigid_sampling_rate: (
                    "-m {}[{}, {}, 1, {}, {}, {}]".format(
                        rigid_metric_name,
                        fixed_image,
                        moving_image,
                        rigid_metric_parameter,
                        rigid_sampling_strategy,
                        rigid_sampling_rate,
                    )
                    if enable_rigid_stage
                    else ""
                ),
            }
        )

        rigid_metric_name: str = field(
            default="MI",
            metadata={
                "help_string": "metric name for rigid stage",
                "allowed_values": {"CC", "MI", "Mattes", "MeanSquares", "Demons", "GC"},
            },
        )

        rigid_metric_parameter: float = field(default=32, metadata={"help_string": "metric parameter for rigid stage"})

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
                    "-c [{}, {}, {}]".format(
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

        affine_metric_: str = field(
            metadata={
                "help_string": "metric parameter for affine stage",
                "readonly": True,
                "formatter": lambda enable_affine_stage, affine_metric_name, fixed_image, moving_image, affine_metric_parameter, affine_sampling_strategy, affine_sampling_rate: (
                    "-m {}[{}, {}, 1, {}, {}, {}]".format(
                        affine_metric_name,
                        fixed_image,
                        moving_image,
                        affine_metric_parameter,
                        affine_sampling_strategy,
                        affine_sampling_rate,
                    )
                    if enable_affine_stage
                    else ""
                ),
            }
        )

        affine_metric_name: str = field(
            default="MI",
            metadata={
                "help_string": "metric name for affine stage",
                "allowed_values": {"CC", "MI", "Mattes", "MeanSquares", "Demons", "GC"},
            },
        )

        affine_metric_parameter: float = field(
            default=32, metadata={"help_string": "metric parameter for affine stage"}
        )

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
                    "-c [{}, {}, {}]".format(
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
                "allowed_values": {"BSpline", "SyN", "BSplineSyN"},
                "formatter": lambda enable_syn_stage, syn_transform, syn_gradient_step, syn_spline_distance, syn_flow_sigma, syn_total_sigma, syn_flow_spline_distance, syn_total_spline_distance, syn_spline_order: (
                    "-t {}[{}]".format(
                        syn_transform,
                        f"{syn_gradient_step}, {syn_spline_distance}"
                        if syn_transform == "BSpline"
                        else f"{syn_gradient_step}, {syn_flow_sigma}, {syn_total_sigma}"
                        if syn_transform == "Syn"
                        else f"{syn_gradient_step}, {syn_flow_spline_distance}, {syn_total_spline_distance}, {syn_spline_order}",
                    )
                    if enable_syn_stage
                    else ""
                ),
            },
        )

        syn_gradient_step: bool = field(default=0.1, metadata={"help_string": "gradient step for SyN stage"})

        syn_spline_distance: int = field(default=26, metadata={"help_string": "spline distance for SyN stage"})

        syn_flow_sigma: float = field(default=3, metadata={"help_string": "sigma for flow field in SyN stage"})

        syn_total_sigma: float = field(default=0, metadata={"help_string": "sigma for total field in SyN stage"})

        syn_flow_spline_distance: int = field(
            default=26, metadata={"help_string": "spline distance for flow field in SyN stage"}
        )

        syn_total_spline_distance: int = field(
            default=0, metadata={"help_string": "spline distance for total field in SyN stage"}
        )

        syn_spline_order: int = field(default=3, metadata={"help_string": "spline order for SyN stage"})

        syn_metric: str = field(
            default="MI",
            metadata={
                "help_string": "metric for SyN stage",
                "allowed_values": {"CC", "MI", "Mattes", "MeanSquares", "Demons", "GC"},
                "formatter": lambda enable_syn_stage, syn_metric, fixed_image, moving_image, syn_radius, syn_num_bins, syn_sampling_strategy, syn_sampling_rate: (
                    "-m {}[{}, {}, 1, {}, {}, {}]".format(
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
                    "-c [{}, {}, {}]".format("x".join(str(c) for c in syn_convergence), syn_threshold, syn_window_size)
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
                    f"-w [{lower_quantile}, {upper_quantile}]" if winsorize_image_intensities else ""
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

    executable = "antsRegistration"


class RegistrationSyNQuick(ShellCommandTask):
    """Task definition for antsRegistrationSyNQuick."""

    @define(kw_only=True)
    class InputSpec(ShellSpec):
        dimensionality: int = field(
            metadata={
                "help_string": "force image dimensionality (2 or 3)",
                "mandatory": True,
                "argstr": "-d",
                "allowed_values": {2, 3},
            }
        )

        fixed_image: PathLike = field(metadata={"help_string": "fixed image", "mandatory": True, "argstr": "-f"})

        moving_image: PathLike = field(metadata={"help_string": "moving image", "mandatory": True, "argstr": "-m"})

        output_prefix: str = field(
            default="output", metadata={"help_string": "prefix for output files", "argstr": "-o"}
        )

        num_threads: int = field(default=1, metadata={"help_string": "number of threads", "argstr": "-n"})

        initial_transforms: Sequence[PathLike] = field(metadata={"help_string": "initial transforms", "argstr": "-i"})

        transform_type: str = field(
            default="s",
            metadata={
                "help_string": "transform type",
                "argstr": "-t",
                "allowed_values": {
                    "t",  # translation
                    "r",  # rigid
                    "a",  # rigid + affine
                    "s",  # rigid + affine + syn
                    "sr",  # rigid + syn
                    "so",  # syn only
                    "b",  # rigid + affine + b-spline
                    "br",  # rigid + b-spline
                    "bo",  # b-spline only
                },
            },
        )

        num_histogram_bins: int = field(
            default=32,
            metadata={
                "help_string": "number of histogram bins in SyN stage",
                "formatter": lambda transform_type, num_histogram_bins: (
                    f"-r {num_histogram_bins}" if "s" in transform_type else ""
                ),
            },
        )

        spline_distance: float = field(
            default=26,
            metadata={
                "help_string": "spline distance for deformable B-spline SyN transform",
                "formatter": lambda transform_type, spline_distance: (
                    f"-s {spline_distance}" if "b" in transform_type else ""
                ),
            },
        )

        gradient_step_size: float = field(
            default=0.1,
            metadata={
                "help_string": "gradient step size for SyN and B-spline SyN",
                "formatter": lambda transform_type, gradient_step_size: (
                    f"-g {gradient_step_size}" if any(c in transform_type for c in ("s", "b")) else ""
                ),
            },
        )

        mask_parameters: str = field(
            metadata={
                "help_string": "mask parameters",
                "readonly": True,
                "formatter": lambda fixed_mask, moving_mask: (
                    "" if not fixed_mask else f"-x {fixed_mask},{moving_mask}" if moving_mask else f"-x {fixed_mask}"
                ),
            }
        )

        fixed_mask: PathLike = field(metadata={"help_string": "mask applied to the fixed image"})

        moving_mask: PathLike = field(
            metadata={"help_string": "mask applied to the moving image", "requires": {"fixed_mask"}}
        )

        precision: str = field(
            default="double",
            metadata={
                "help_string": "use float or double precision",
                "allowed_values": {"float", "double"},
                "formatter": lambda precision: f"-p {precision[0]}",
            },
        )

        use_histogram_matching: bool = field(
            default=False,
            metadata={
                "help_string": "use histogram matching",
                "formatter": lambda use_histogram_matching: f"-j {use_histogram_matching:d}",
            },
        )

        use_reproducible_mode: bool = field(
            default=False,
            metadata={
                "help_string": "use reproducible mode",
                "formatter": lambda use_reproducible_mode: f"-y {use_reproducible_mode:d}",
            },
        )

        random_seed: int = field(metadata={"help_string": "fix random seed", "argstr": "-e"})

    input_spec = SpecInfo(name="Input", bases=(InputSpec,))

    @define(kw_only=True)
    class OutputSpec(ShellOutSpec):
        warped_moving_image: File = field(
            metadata={
                "help_string": "moving image warped to fixed image space",
                "output_file_template": "{output_prefix}Warped.nii.gz",
            }
        )

        warped_fixed_image: File = field(
            metadata={
                "help_string": "fixed image warped to moving image space",
                "output_file_template": "{output_prefix}InverseWarped.nii.gz",
            }
        )

        affine_transform: File = field(
            metadata={
                "help_string": "affine transform",
                "output_file_template": "{output_prefix}0GenericAffine.mat",
            }
        )

        forward_warp_field: File = field(
            metadata={
                "help_string": "forward warp field",
                "callable": lambda transform_type, output_prefix: (
                    f"{output_prefix}1Warp.nii.gz" if transform_type not in ("t", "r", "a") else NOTHING
                ),
            }
        )

        inverse_warp_field: File = field(
            metadata={
                "help_string": "inverse warp field",
                "callable": lambda transform_type, output_prefix: (
                    f"{output_prefix}1InverseWarp.nii.gz" if transform_type not in ("t", "r", "a") else NOTHING
                ),
            }
        )

    output_spec = SpecInfo(name="Output", bases=(OutputSpec,))

    executable = "antsRegistrationSyNQuick.sh"
