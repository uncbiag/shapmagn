import os
import random
import numpy as np
import torch
import pyvista as pv
import subprocess
from shapmagn.utils.utils import add_zero_last_dim

pv.set_plot_theme("document")

PPI = 2

LIGHTING = "none"  # "none"  # "three lights", "light_kit"


def setup_lights(plotter, elev=75, azim=100):

    if LIGHTING == "none":
        # elev = 0, azim = 0 is the +x direction
        # elev = 0, azim = 90 is the +y direction
        # elev = 90, azim = 0 is the +z direction

        light = pv.Light()
        light.set_direction_angle(elev, azim)
        # light.set_headlight()
        plotter.add_light(light)


def format_input(input):
    dim = input.shape[-1]
    if dim == 2:
        input = add_zero_last_dim(input)
    if isinstance(input, torch.Tensor):
        input = input.squeeze().detach().cpu().numpy()
    return input


def color_adaptive(color, turn_on=True):
    if turn_on:
        if len(color) > 3:
            color = (color - color.min()) / (color.max() - color.min() + 1e-7)
    return color


def plot_lungs(plotter, cloud, radii, nradii=10, color="source", **kwargs):

    if color == "source":
        cmap = "Reds"
        clim = [-0.6, 1.1]
    elif color == "target":
        cmap = "Blues"
        clim = [-0.6, 1.1]
    else:
        raise ValueError(f"Unknown color: {color}.")

    # Normalize to [0, 1]:
    radii = color_adaptive(radii)

    for k in range(nradii):
        mask = (radii > (k / nradii)) * (radii <= ((k + 1) / nradii))

        plotter.add_mesh(
            pv.PolyData(cloud[mask, :]),
            scalars=radii[mask],
            point_size=15 * PPI * (((k + 1) + 0.5) / nradii),
            render_points_as_spheres=True,
            lighting=True,
            cmap=cmap,
            clim=clim,
            style="points",
            show_scalar_bar=False,
            ambient=0.5,
            **kwargs,
        )


def plot_ghost(plotter, obj):
    plotter.add_mesh(
        obj,
        color="gray",
        point_size=10,
        render_points_as_spheres=True,
        opacity=0.05,
        style="points",
        show_scalar_bar=True,
    )


def finalize_camera(p, camera_pos, show, saving_capture_path, saving_gif_path):
    if camera_pos is not None:
        p.camera_position = camera_pos
    if show:
        p.show(auto_close=False)
        if saving_capture_path:
            p.screenshot(saving_capture_path)

    elif saving_capture_path:
        p.show(screenshot=saving_capture_path)

    if saving_gif_path:
        p.open_gif(saving_gif_path)

        # Update camera and write a frame for each updated position
        nframe = 360
        for i in range(nframe):
            p.camera_position = [
                (
                    7 * np.cos(i * np.pi / 180.0),
                    7 * np.cos(i * np.pi / 180.0),
                    7 * np.sin(i * np.pi / 180.0),
                ),
                (0, 0, 0),
                (0, 1, 0),
            ]
            p.write_frame()
            p.render()

        # Close movie and delete object
    p.close()


def visualize_point_fea(
    points,
    fea,
    rgb_on=True,
    saving_gif_path=None,
    saving_capture_path=None,
    camera_pos=None,
    show=True,
    col_adaptive=True,
):
    points = format_input(points)
    fea = format_input(fea)
    p = pv.Plotter(window_size=[1920, 1280], off_screen=not show)
    p.add_mesh(
        pv.PolyData(points),
        scalars=color_adaptive(fea, col_adaptive),
        cmap="magma",
        point_size=10,
        render_points_as_spheres=True,
        rgb=rgb_on,
        opacity="linear",
        lighting=True,
        style="points",
        show_scalar_bar=True,
    )
    # p.show_grid()

    finalize_camera(p, camera_pos, show, saving_capture_path, saving_gif_path)

    return p


def visualize_point_fea_with_arrow(
    points,
    fea,
    vectors,
    rgb_on=True,
    saving_gif_path=None,
    saving_capture_path=None,
    camera_pos=None,
    show=True,
):
    points = format_input(points)
    fea = format_input(fea)
    vectors = format_input(vectors)
    p = pv.Plotter(window_size=[1920, 1280], off_screen=not show)
    # install pyvistaqt for background plotting that plots without pause the program
    # p = pyvistaqt.BackgroundPlotter(off_screen= not show)
    point_obj = pv.PolyData(points)
    point_obj.vectors = vectors
    p.add_mesh(
        point_obj.arrows, scalars="GlyphScale", lighting=True, stitle="Vector Magnitude"
    )
    p.add_mesh(
        point_obj,
        scalars=color_adaptive(fea),
        cmap="magma",
        point_size=10,
        render_points_as_spheres=True,
        rgb=rgb_on,
        opacity="linear",
        lighting=True,
        style="points",
        show_scalar_bar=True,
    )
    p.show_grid()

    finalize_camera(p, camera_pos, show, saving_capture_path, saving_gif_path)

    return p


def visualize_point_pair(
    points1,
    points2,
    feas1,
    feas2,
    title1,
    title2,
    rgb_on=True,
    point_size=[10, 10],
    saving_gif_path=None,
    saving_capture_path=None,
    camera_pos=None,
    show=True,
    col_adaptive=True,
):
    points1 = format_input(points1)
    points2 = format_input(points2)
    feas1 = format_input(feas1)
    feas2 = format_input(feas2)

    if isinstance(rgb_on, bool):
        rgb_on = [rgb_on] * 2

    p = pv.Plotter(
        window_size=[1920, 1280],
        notebook=0,
        shape=(1, 2),
        border=False,
        off_screen=not show,
    )
    p.subplot(0, 0)
    p.add_text(title1, font_size=18)
    p.add_mesh(
        pv.PolyData(points1),
        scalars=color_adaptive(feas1, col_adaptive),
        cmap="viridis",
        point_size=point_size[0],
        render_points_as_spheres=True,
        rgb=rgb_on[0],
        opacity="linear",
        lighting=True,
        style="points",
        show_scalar_bar=True,
    )
    p.subplot(0, 1)
    p.add_text(title2, font_size=18)
    p.add_mesh(
        pv.PolyData(points2),
        scalars=color_adaptive(feas2, col_adaptive),
        cmap="magma",
        point_size=point_size[1],
        render_points_as_spheres=True,
        rgb=rgb_on[1],
        opacity="linear",
        lighting=True,
        style="points",
        show_scalar_bar=True,
    )
    p.link_views()  # link all the views

    finalize_camera(p, camera_pos, show, saving_capture_path, saving_gif_path)

    return p


def visualize_point_overlap(
    points1,
    points2,
    feas1,
    feas2,
    title,
    point_size=10,
    rgb_on=True,
    color="source",
    light_params={},
    opacity=("linear", "linear"),
    saving_gif_path=None,
    saving_capture_path=None,
    camera_pos=None,
    show=True,
):
    # Format the source and target shapes:
    points1 = format_input(points1)
    points2 = format_input(points2)
    feas1 = format_input(feas1)
    feas2 = format_input(feas2)

    if isinstance(rgb_on, bool):
        rgb_on = [rgb_on] * 2

    # Create the window:
    p = pv.Plotter(
        window_size=[1920, 1920],
        off_screen=not show,
        lighting=LIGHTING,
    )

    setup_lights(p, **light_params)

    # install pyvistaqt for background plotting that plots without pause the program
    # p = pyvistaqt.BackgroundPlotter(off_screen= not show)

    p.add_text(title, font_size=18)

    plot_lungs(
        p,
        points1,
        color_adaptive(feas1),
        color=color,
        rgb=rgb_on[0],
        opacity=opacity[0],
    )

    p.add_mesh(
        pv.PolyData(points2),
        scalars=color_adaptive(feas2),
        point_size=point_size * PPI,
        render_points_as_spheres=True,
        lighting=True,
        cmap="Oranges",
        style="points",
        show_scalar_bar=False,
        ambient=0.5,
        rgb=rgb_on[1],
        opacity=opacity[1],
    )

    # p.show_grid()
    finalize_camera(p, camera_pos, show, saving_capture_path, saving_gif_path)

    return p


def visualize_full(
    source=None,
    flowed=None,
    target=None,
    flow=None,
    rgb_on=True,
    saving_gif_path=None,
    saving_capture_path=None,
    camera_pos=None,
    add_bg_contrast=True,
    show=True,
    light_params={},
):

    # Format the input data:
    for dic in [source, flowed, target]:
        if dic is not None:
            dic["points"] = format_input(dic["points"])
            dic["radii"] = format_input(dic["radii"])

    if flow is not None:
        flow = format_input(flow)

    if isinstance(rgb_on, bool):
        rgb_on = [rgb_on] * 3

    # PyVista window:
    if source is None:
        p = pv.Plotter(
            window_size=[2500 * PPI, 1024 * PPI],
            shape=(1, 3),
            border=False,
            off_screen=not show,
            lighting=LIGHTING,
        )
    else:
        p = pv.Plotter(
            window_size=[3000 * PPI, 1024 * PPI],
            shape=(1, 4),
            border=False,
            off_screen=not show,
            lighting=LIGHTING,
        )

    setup_lights(p, **light_params)

    plot_id = 0

    # Plot 1 ---------------------------------
    if source is not None:
        p.subplot(0, plot_id)
        plot_id += 1
        p.add_text(source["name"], font_size=18)

        plot_lungs(p, source["points"], source["radii"], color="source", rgb=rgb_on[0])

    # Plot 2 ---------------------------------
    p.subplot(0, plot_id)
    plot_id += 1

    p.add_text(flowed["name"], font_size=18)

    plot_lungs(p, flowed["points"], flowed["radii"], color="source", rgb=rgb_on[1])

    if source is not None:
        obj1 = pv.PolyData(source["points"])

        if flow is not None:
            npoints = flow.shape[0]
            flow_ = np.zeros_like(flow)
            index = list(range(0, npoints, 30))
            flow_[index, :] = flow[index]
            obj1.point_arrays["flow"] = flow_  # flow_
            geom = pv.Arrow(tip_radius=0.08, shaft_radius=0.035)
            arrows = obj1.glyph(orient="flow", geom=geom)
            p.add_mesh(arrows, color="black", opacity=0.3)

        if add_bg_contrast:
            plot_ghost(p, obj1)

    # Plot 3 ----------------------------------
    p.subplot(0, plot_id)
    plot_id += 1
    p.add_text(target["name"], font_size=18)

    if source is not None and add_bg_contrast:
        plot_ghost(p, obj1)

    plot_lungs(p, target["points"], target["radii"], color="target", rgb=rgb_on[2])

    # Plot 4: ----------------------------------
    p.subplot(0, plot_id)
    plot_id += 1

    plot_lungs(
        p,
        flowed["points"],
        flowed["radii"],
        color="source",
        rgb=rgb_on[1],
        opacity=0.75,
    )

    plot_lungs(
        p,
        target["points"],
        target["radii"],
        color="target",
        rgb=rgb_on[2],
        opacity=0.75,
    )

    p.add_text(flowed["name"] + "_overlap_" + target["name"], font_size=22)

    # Camera manipulation: -----------------------
    p.link_views()  # link all the views
    finalize_camera(p, camera_pos, show, saving_capture_path, saving_gif_path)

    return p


def visualize_point_pair_overlap(
    flowed_points,
    target_points,
    flowed_radii,
    target_radii,
    flowed_name,
    target_name,
    **kwargs,
):
    return visualize_full(
        source=None,
        flowed={
            "points": flowed_points,
            "radii": flowed_radii,
            "name": flowed_name,
        },
        target={
            "points": target_points,
            "radii": target_radii,
            "name": target_name,
        },
        **kwargs,
    )


def visualize_source_flowed_target_overlap(
    source_points,
    flowed_points,
    target_points,
    source_radii,
    flowed_radii,
    target_radii,
    source_name,
    flowed_name,
    target_name,
    **kwargs,
):
    return visualize_full(
        source={
            "points": source_points,
            "radii": source_radii,
            "name": source_name,
        },
        flowed={
            "points": flowed_points,
            "radii": flowed_radii,
            "name": flowed_name,
        },
        target={
            "points": target_points,
            "radii": target_radii,
            "name": target_name,
        },
        **kwargs,
    )


def visualize_multi_point(
    points_list,
    feas_list,
    titles_list,
    rgb_on=True,
    saving_gif_path=None,
    saving_capture_path=None,
    camera_pos=None,
    show=True,
):
    num_views = len(points_list)
    for i, points in enumerate(points_list):
        points_list[i] = format_input(points)
    for i, feas in enumerate(feas_list):
        feas_list[i] = format_input(feas)
    if isinstance(rgb_on, bool):
        rgb_on = [rgb_on] * num_views

    p = pv.Plotter(
        window_size=[1920, 1280],
        notebook=0,
        shape=(1, num_views),
        border=False,
        off_screen=not show,
    )
    for i in range(num_views):
        p.subplot(0, i)
        p.add_text(titles_list[i], font_size=18)
        p.add_mesh(
            pv.PolyData(points_list[i]),
            scalars=color_adaptive(feas_list[i]),
            cmap="magma",
            point_size=10,
            render_points_as_spheres=True,
            rgb=rgb_on[i],
            opacity="linear",
            lighting=True,
            style="points",
            show_scalar_bar=True,
        )
    p.link_views()  # link all the views
    # Set a camera position to all linked views
    if camera_pos is not None:
        p.camera_position = camera_pos

    if show:
        cm_position = p.show(auto_close=False)
        print(cm_position)
    if saving_capture_path:
        p.show(screenshot=saving_capture_path)

    if saving_gif_path:
        p.open_gif(saving_gif_path)

        # Update camera and write a frame for each updated position
        nframe = 360
        for i in range(nframe):
            p.camera_position = [
                (
                    7 * np.cos(i * np.pi / 180.0),
                    7 * np.cos(i * np.pi / 180.0),
                    7 * np.sin(i * np.pi / 180.0),
                ),
                (0, 0, 0),
                (0, 1, 0),
            ]
            p.write_frame()
            p.render()

        # Close movie and delete object
        p.close()
    return p


# def visualize_multi_point(points_list, feas_list, titles_list,rgb_on=True, saving_gif_path=None, saving_capture_path=None, camera_pos=None,show=True):
#     num_views = len(points_list)
#     for i,points in enumerate(points_list):
#         points_list[i] = format_input(points)
#     for i, feas in enumerate(feas_list):
#         feas_list[i] = format_input(feas)
#     if isinstance(rgb_on,bool):
#         rgb_on = [rgb_on]* num_views
#
#     p = pv.Plotter(window_size=[1920, 1280],notebook=0, shape=(1, num_views), border=False, off_screen= not show)
#     for i in range(num_views):
#         p.subplot(0, i)
#         p.add_text(titles_list[i], font_size=18)
#         p.add_mesh(pv.PolyData(points_list[i]),
#                          scalars=color_adaptive(feas_list[i]),
#                          cmap="magma", point_size=10,
#                          render_points_as_spheres=True,
#                          rgb=rgb_on[i],
#                          opacity="linear",
#                          lighting=True,
#                          style="points", show_scalar_bar=True)
#     p.link_views()  # link all the views
#     # Set a camera position to all linked views
#     if camera_pos is not None:
#         p.camera_position = camera_pos
#
#
#     if show:
#         cm_position = p.show(auto_close=False)
#     if saving_capture_path:
#         p.show(screenshot=saving_capture_path)
#
#     if saving_gif_path:
#         p.open_gif(saving_gif_path)
#
#         # Update camera and write a frame for each updated position
#         nframe = 360
#         for i in range(nframe):
#             p.camera_position = [
#                 (7 * np.cos(i * np.pi / 180.0), 7 * np.cos(i * np.pi / 180.0), 7 * np.sin(i * np.pi / 180.0)),
#                 (0, 0, 0),
#                 (0, 1, 0),
#             ]
#             p.write_frame()
#             p.render()
#
#         # Close movie and delete object
#         p.close()
#     return p


def capture_plotter(render_by_weight=False, camera_pos=None, add_bg_contrast=True):
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
            if render_by_weight:
                visualize_source_flowed_target_overlap(
                    sp,
                    fp,
                    tp,
                    sw,
                    fw,
                    tw,
                    title1="source",
                    title2="flowed",
                    title3="target",
                    rgb_on=False,
                    saving_capture_path=path,
                    camera_pos=camera_pos,
                    add_bg_contrast=add_bg_contrast,
                    show=False,
                )
            else:
                visualize_source_flowed_target_overlap(
                    sp,
                    fp,
                    tp,
                    sp,
                    sp,
                    tp,
                    title1="source",
                    title2="flowed",
                    title3="target",
                    rgb_on=True,
                    saving_capture_path=path,
                    camera_pos=camera_pos,
                    add_bg_contrast=add_bg_contrast,
                    show=False,
                )
            cp_command = "cp {} {}".format(
                path, os.path.join(stage_folder, pair_name + "_flowed_target.png")
            )
            subprocess.Popen(cp_command, stdout=subprocess.PIPE, shell=True)

    return save


def shape_capture_plotter(
    render_by_weight=False, camera_pos=None, add_bg_contrast=True
):
    def save(record_path, stage_suffix, shape_name_list, shape):
        # due to the bug of vtk 9.0, at most around 200+ plots can be saved, so this function would be safe if calling less than 50 times
        stage_folder = os.path.join(record_path, stage_suffix)
        os.makedirs(stage_folder, exist_ok=True)
        for sp, sw, shape_name in zip(shape.points, shape.weights, shape_name_list):
            case_folder = os.path.join(record_path, shape_name)
            os.makedirs(case_folder, exist_ok=True)
            path = os.path.join(case_folder, stage_suffix + ".png")
            if render_by_weight:
                visualize_point_fea(
                    sp,
                    sw,
                    rgb_on=False,
                    saving_capture_path=path,
                    camera_pos=camera_pos,
                    show=False,
                )
            else:
                visualize_point_fea(
                    sp,
                    sp,
                    rgb_on=True,
                    saving_capture_path=path,
                    camera_pos=camera_pos,
                    show=False,
                )

            cp_command = "cp {} {}".format(
                path, os.path.join(stage_folder, shape_name + ".png")
            )
            subprocess.Popen(cp_command, stdout=subprocess.PIPE, shell=True)

    return save
