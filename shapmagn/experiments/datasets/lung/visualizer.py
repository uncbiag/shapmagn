import os
import pyvista as pv
import subprocess
from shapmagn.utils.visualizer import visualize_source_flowed_target_overlap, default_plot, color_adaptive
PPI=2

def lung_plot(color="source"):
    def plot(plotter, cloud, visualfea, levels=10, **kwargs):

        if color == "source":
            cmap = "Reds"
            clim = [-0.6, 1.1]
        elif color == "target":
            cmap = "Blues"
            clim = [-0.6, 1.1]
        else:
            raise ValueError(f"Unknown color: {color}.")

        # Normalize to [0, 1]:
        visualfea = color_adaptive(visualfea)

        for k in range(levels):
            mask = (visualfea > (k / levels)) * (visualfea <= ((k + 1) / levels))

            plotter.add_mesh(
                pv.PolyData(cloud[mask, :]),
                scalars=visualfea[mask],
                point_size=15 * PPI * (((k + 1) + 0.5) / levels),
                render_points_as_spheres=True,
                lighting=True,
                cmap=cmap,
                clim=clim,
                style="points",
                #show_scalar_bar=False,
                ambient=0.5,
                **kwargs,
            )
    return plot












def lung_capture_plotter(camera_pos=None, add_bg_contrast=True):
    def save(record_path, stage_suffix, pair_name_list, shape_pair):
        source, flowed, target = shape_pair.source, shape_pair.flowed, shape_pair.target
        # due to the bug of vtk 9.0, at most around 200+ plots can be saved, so this function would be safe if calling less than 50 times
        stage_folder = os.path.join(record_path, stage_suffix)
        os.makedirs(stage_folder, exist_ok=True)
        for sp, fp, tp, sw, fw, tw, pair_name in zip(
            source.points,
            flowed.points,
            target.points,
            source.weights,
            flowed.weights,
            target.weights,
            pair_name_list,
        ):
            case_folder = os.path.join(record_path, pair_name)
            os.makedirs(case_folder, exist_ok=True)
            path = os.path.join(
                case_folder, "flowed_target" + "_" + stage_suffix + ".png"
            )

            visualize_source_flowed_target_overlap(
                sp,
                fp,
                tp,
                sw,
                fw,
                tw,
                "source",
                "flowed",
                "target",
                rgb_on=False,
                saving_capture_path=path,
                camera_pos=camera_pos,
                add_bg_contrast=add_bg_contrast,
                source_plot_func=lung_plot(color="source"),
                flowed_plot_func=lung_plot(color="source"),
                target_plot_func=lung_plot(color="target"),
                show=False,
                light_mode="none"
            )

            cp_command = "cp {} {}".format(
                path, os.path.join(stage_folder, pair_name + "_flowed_target.png")
            )
            subprocess.Popen(cp_command, stdout=subprocess.PIPE, shell=True)

    return save