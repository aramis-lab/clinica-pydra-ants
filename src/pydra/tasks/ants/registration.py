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

__all__ = ["Registration", "RegistrationSyNQuick"]

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
            metadata={
                "help_string": "force image dimensionality (2, 3 or 4)",
                "argstr": "-d",
                "allowed_values": {2, 3, 4},
            }
        )

        fixed_image: PathLike = field(metadata={"help_string": "fixed image", "mandatory": True})

        moving_image: PathLike = field(metadata={"help_string": "moving image", "mandatory": True})

        output_: str = field(
            metadata={
                "help_string": "output parameter",
                "argstr": "-o [{output_transform_prefix},{warped_moving_image},{warped_fixed_image}]",
                "readonly": True,
            }
        )

        output_transform_prefix: str = field(default="output", metadata={"help_string": "output transform prefix"})

        warped_moving_image: str = field(
            metadata={"help_string": "warped moving image", "output_file_template": "{moving_image}_warped"}
        )

        warped_fixed_image: str = field(
            metadata={"help_string": "warped fixed image", "output_file_template": "{fixed_image}_warped"}
        )

        @staticmethod
        def format_metric(
            metric_type,
            fixed_image,
            moving_image,
            metric_weight,
            radius,
            num_bins,
            sampling_strategy,
            sampling_percentage,
            use_gradient_filter,
        ) -> str:
            return "-m {0}[{1}]".format(
                metric_type,
                "{0},{1},{2},{3},{4},{5},{6}".format(
                    fixed_image,
                    moving_image,
                    metric_weight,
                    num_bins if metric_type in {"Mattes", "MI"} else radius,
                    sampling_strategy,
                    sampling_percentage,
                    use_gradient_filter,
                ),
            )

        metric_: str = field(metadata={"help_string": "metric parameter", "readonly": True, "formatter": format_metric})

        metric_type: str = field(
            metadata={
                "help_string": "metric type",
                "mandatory": True,
                "allowed_values": {"CC", "Demons", "GC", "Mattes", "MeanSquares", "MI"},
            }
        )

        metric_weight: float = field(default=1.0, metadata={"help_string": "metric weighting"})

        radius: int = field(default=4, metadata={"help_string": "radius"})

        num_bins: int = field(default=32, metadata={"help_string": "number of bins"})

        sampling_strategy: str = field(
            default="None",
            metadata={"help_string": "sampling strategy", "allowed_values": {"None", "Regular", "Random"}},
        )

        sampling_percentage: float = field(default=1.0, metadata={"help_string": "sampling percentage"})

        use_gradient_filter: bool = field(default=False, metadata={"help_string": "use gradient filter"})

        @staticmethod
        def format_transform(
            transform_type,
            gradient_step,
            mesh_size,
            update_field_variance,
            total_field_variance,
            update_field_mesh_size,
            total_field_mesh_size,
            spline_order,
            num_time_indices,
            update_field_time_variance,
            total_field_time_variance,
            velocity_field_mesh_size,
            num_timepoint_samples,
        ) -> str:
            # Common to all transform types.
            parameters = [gradient_step]

            if transform_type == "BSpline":
                parameters += [mesh_size]
            elif transform_type in ("GaussianDisplacementField", "Syn"):
                parameters += [update_field_variance, total_field_variance]
            elif transform_type in ("BSplineDisplacementField", "BSplineSyn"):
                parameters += [update_field_mesh_size, total_field_mesh_size, spline_order]
            elif transform_type == "TimeVaryingVelocityField":
                parameters += [
                    num_time_indices,
                    update_field_variance,
                    update_field_time_variance,
                    total_field_variance,
                    total_field_time_variance,
                ]
            elif transform_type == "TimeVaryingBSplineVelocityField":
                parameters += [velocity_field_mesh_size, num_timepoint_samples, spline_order]
            elif transform_type == "Exponential":
                parameters += [update_field_variance, total_field_variance, num_integration_steps]
            elif transform_type == "BSplineExponential":
                parameters += [update_field_mesh_size, total_field_mesh_size, num_integration_steps, spline_order]
            else:
                pass

            return f"-t {transform_type}[{','.join(parameters)}]"

        transform_: str = field(
            metadata={"help_string": "transform parameter", "readonly": True, "formatter": format_transform}
        )

        transform_type: str = field(
            metadata={
                "help_string": "transform type",
                "mandatory": True,
                "allowed_values": {
                    "Rigid",
                    "Affine",
                    "CompositeAffine",
                    "Similarity",
                    "Translation",
                    "BSpline",
                    "GaussianDisplacementField",
                    "BSplineDisplacementField",
                    "TimeVaryingVelocityField",
                    "TimeVaryingBSplineVelocityField",
                    "SyN",
                    "BSplineSyN",
                    "Exponential",
                    "BSplineExponential",
                },
            }
        )

        gradient_step: float = field(metadata={"help_string": "gradient step or learning rate", "mandatory": True})

        mesh_size: Sequence[int] = field(metadata={"help_string": "mesh size at base level"})

        update_field_variance: float = field(metadata={"help_string": "variance for update field"})

        total_field_variance: float = field(metadata={"help_string": "variance for total field"})

        update_field_mesh_size: Sequence[int] = field(
            metadata={"help_string": "mesh size at base level for update field"}
        )

        total_field_mesh_size: Sequence[int] = field(
            metadata={"help_string": "mesh size at base level for total field"}
        )

        spline_order: int = field(default=3, metadata={"help_string": "spline order"})

        num_time_indices: int = field(metadata={"help_string": "number of time indices"})

        update_field_time_variance: float = field(metadata={"help_string": "time variance for update field"})

        total_field_time_variance: float = field(metadata={"help_string": "time variance for total field"})

        velocity_field_mesh_size: Sequence[int] = field(
            metadata={"help_string": "mesh size at base level for velocity field"}
        )

        num_timepoint_samples: int = field(default=4, metadata={"help_string": "number of timepoint samples"})

        num_integration_steps: int = field(metadata={"help_string": "number of integration steps"})

        @staticmethod
        def format_convergence(num_iterations_per_level, convergence_threshold, convergence_window_size) -> str:
            return "-c [{0},{1},{2}]".format(
                "x".join(str(n) for n in num_iterations_per_level),
                convergence_threshold,
                convergence_window_size,
            )

        convergence_: str = field(
            metadata={"help_string": "convergence parameter", "readonly": True, "formatter": format_convergence}
        )

        num_iterations_per_level: Sequence[int] = field(
            metadata={"help_string": "number of iterations per level", "mandatory": True}
        )

        convergence_threshold: float = field(default=1e-6, metadata={"help_string": "convergence threshold"})

        convergence_window_size: float = field(default=10, metadata={"help_string": "convergence window size"})

        smoothing_sigmas: Sequence[int] = field(
            metadata={
                "help_string": "sigma of Gaussian smoothing for each level",
                "mandatory": True,
                "formatter": lambda smoothing_sigmas, smoothing_in_mm: (
                    "-s {0}{1}".format(
                        "x".join(str(s) for s in smoothing_sigmas),
                        "mm" if smoothing_in_mm else "vox",
                    )
                ),
            }
        )

        smoothing_in_mm: bool = field(
            default=False, metadata={"help_string": "specify smoothing in millimeters instead of voxels"}
        )

        shrink_factor_per_level: Sequence[int] = field(
            metadata={
                "help_string": "shrink factor for each level",
                "mandatory": True,
                "formatter": lambda shrink_factor_per_level: "-f {}".format(
                    "x".join(str(f) for f in shrink_factor_per_level)
                ),
            }
        )

        use_histogram_matching: bool = field(
            default=False,
            metadata={
                "help_string": "use histogram matching",
                "formatter": lambda use_histogram_matching: f"-j {use_histogram_matching:d}",
            },
        )

        winsorize_image_intensities: bool = field(
            metadata={
                "help_string": "winsorize image intensities",
                "formatter": lambda winsorize_image_intensities, lower_quantile, upper_quantile: (
                    f"-w [{lower_quantile},{upper_quantile}]" if winsorize_image_intensities else ""
                ),
            }
        )

        lower_quantile: float = field(default=0.0, metadata={"help_string": "lower quantile for winsorization"})

        upper_quantile: float = field(default=1.0, metadata={"help_string": "upper quantile for winsorization"})

        masks_: str = field(
            metadata={
                "help_string": "masks parameter",
                "readonly": True,
                "formatter": lambda fixed_mask, moving_mask: f"-x [{fixed_mask},{moving_mask}]",
            }
        )

        fixed_mask: PathLike = field(default="NA", metadata={"help_string": "mask applied to fixed image"})

        moving_mask: PathLike = field(default="NA", metadata={"help_string": "mask applied to moving image"})

        precision: str = field(
            default="double",
            metadata={
                "help_string": "use float or double precision",
                "allowed_values": {"float", "double"},
                "formatter": lambda precision: "--float" if precision == "float" else "",
            },
        )

        use_minc_format: bool = field(metadata={"help_string": "save transforms to MINC format", "argstr": "--minc"})

        random_seed: int = field(metadata={"help_string": "random seed", "argstr": "--random-seed"})

        verbose: bool = field(metadata={"help_string": "verbose output", "formatter": lambda verbose: f"{verbose:d}"})

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

        velocity_field: File = field(
            metadata={
                "help_string": "velocity field",
                "output_file_template": "{output_transform_prefix}1VelocityField.nii.gz",
            }
        )

    input_spec = SpecInfo(name="Input", bases=(InputSpec,))

    output_spec = SpecInfo(name="Output", bases=(OutputSpec,))

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
