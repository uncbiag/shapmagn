import os
import numpy as np
import torch
import pyvista as pv
import subprocess
from shapmagn.utils.utils import add_zero_last_dim
pv.set_plot_theme("document")
PPI = 2


def setup_lights(plotter, light_mode="light_kit", elev=75, azim=100):

    if light_mode == "none":
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


def default_plot(cmap="magma", rgb=True, point_size=15, render_points_as_spheres=True):
    def plot(plotter, cloud, visualfea, **kwargs):
        use_rgb = visualfea.shape[-1]==3 and rgb
        plotter.add_mesh(
            pv.PolyData(cloud),
            scalars=visualfea,
            lighting=True,
            render_points_as_spheres=render_points_as_spheres,
            rgb=use_rgb,
            point_size=point_size,
            cmap=cmap,
            style="points",
            **kwargs
        )
    return plot




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
        cur_camera_pos = p.show(auto_close=False)
        print(cur_camera_pos)
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
    opacity=('linear', 'linear'),
    source_plot_func=default_plot(cmap="magma"),
    target_plot_func=default_plot(cmap="viridis"),
    saving_gif_path=None,
    saving_capture_path=None,
    camera_pos=None,
    show=True,
    col_adaptive=True,
    light_mode="light_kit",
    light_params={}
):
    points1 = format_input(points1)
    points2 = format_input(points2)
    feas1 = format_input(feas1)
    feas2 = format_input(feas2)
    p = pv.Plotter(
        window_size=[1920, 1280],
        notebook=0,
        shape=(1, 2),
        border=False,
        off_screen=not show,
        lighting=light_mode,
    )

    setup_lights(p, light_mode, **light_params)
    p.subplot(0, 0)
    p.add_text(title1, font_size=18)
    source_plot_func(p, points1, color_adaptive(feas1, col_adaptive), opacity=opacity[0], show_scalar_bar=True)
    p.subplot(0, 1)
    p.add_text(title2, font_size=18)
    target_plot_func(p, points2, color_adaptive(feas2, col_adaptive), opacity=opacity[1], show_scalar_bar=True)
    p.link_views()  # link all the views

    finalize_camera(p, camera_pos, show, saving_capture_path, saving_gif_path)

    return p



def visualize_landmark_overlap(
    points1,
    points2,
    feas1,
    feas2,
    title,
    opacity=(1, 1),
    source_plot_func=default_plot(cmap="magma"),
    target_plot_func=default_plot(cmap="viridis"),
    saving_gif_path=None,
    saving_capture_path=None,
    camera_pos=None,
    show=True,
    col_adaptive=True,
    light_mode="light_kit",
    light_params={}
):
    # Format the source and target shapes:
    points1 = format_input(points1)
    points2 = format_input(points2)
    feas1 = format_input(feas1)
    feas2 = format_input(feas2)

    # Create the window:
    p = pv.Plotter(
        window_size=[1920, 1920],
        off_screen=not show,
        lighting=light_mode,
    )

    setup_lights(p,light_mode, **light_params)
    p.add_text(title, font_size=18)
    source_plot_func(p, points1,color_adaptive(feas1,col_adaptive), opacity=opacity[0],show_scalar_bar=True)
    target_plot_func(p, points2,feas2, opacity=opacity[1],show_scalar_bar=True, stitle="")
    # p.show_grid()
    finalize_camera(p, camera_pos, show, saving_capture_path, saving_gif_path)
    return p


def visualize_point_overlap(
    points1,
    points2,
    feas1,
    feas2,
    title,
    opacity=("linear", "linear"),
    saving_gif_path=None,
    saving_capture_path=None,
    camera_pos=None,
    show=True,
    source_plot_func=default_plot(cmap="magma"),
    target_plot_func=default_plot(cmap="viridis"),
    col_adaptive = True,
    light_mode="light_kit",
    light_params={}
):
    # Format the source and target shapes:
    points1 = format_input(points1)
    points2 = format_input(points2)
    feas1 = format_input(feas1)
    feas2 = format_input(feas2)

    # Create the window:
    p = pv.Plotter(
        window_size=[1920, 1920],
        off_screen=not show,
        lighting=light_mode,
    )

    setup_lights(p,light_mode, **light_params)
    p.add_text(title, font_size=18)
    source_plot_func(p, points1, color_adaptive(feas1,col_adaptive), opacity=opacity[0])
    target_plot_func(p, points2, color_adaptive(feas2,col_adaptive), opacity=opacity[1])
    # p.show_grid()
    finalize_camera(p, camera_pos, show, saving_capture_path, saving_gif_path)

    return p


def visualize_full(
    source=None,
    flowed=None,
    target=None,
    flow=None,
    saving_gif_path=None,
    saving_capture_path=None,
    camera_pos=None,
    add_bg_contrast=True,
    opacity= ("linear","linear","linear"),
    source_plot_func=default_plot(cmap="magma"),
    flowed_plot_func=default_plot(cmap="magma"),
    target_plot_func=default_plot(cmap="viridis"),
    col_adaptive =True,
    show=True,
    light_mode="light_kit",
    light_params={}
):

    # Format the input data:
    for dic in [source, flowed, target]:
        if dic is not None:
            dic["points"] = format_input(dic["points"])
            dic["visualfea"] = format_input(dic["visualfea"])

    if flow is not None:
        flow = format_input(flow)



    # PyVista window:
    if source is None:
        p = pv.Plotter(
            window_size=[2500 * PPI, 1024 * PPI],
            shape=(1, 3),
            border=False,
            off_screen=not show,
            lighting=light_mode,
        )
    else:
        p = pv.Plotter(
            window_size=[3000 * PPI, 1024 * PPI],
            shape=(1, 4),
            border=False,
            off_screen=not show,
            lighting=light_mode,
        )

    setup_lights(p,light_mode, **light_params)

    plot_id = 0

    # Plot 1 ---------------------------------
    if source is not None:
        p.subplot(0, plot_id)
        plot_id += 1
        p.add_text(source["name"], font_size=18)

        source_plot_func(p, source["points"], color_adaptive(source["visualfea"],col_adaptive),opacity=opacity[0])

    # Plot 2 ---------------------------------
    p.subplot(0, plot_id)
    plot_id += 1

    p.add_text(flowed["name"], font_size=18)

    flowed_plot_func(p, flowed["points"], color_adaptive(flowed["visualfea"],col_adaptive),opacity=opacity[1])

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

    target_plot_func(p, target["points"], color_adaptive(target["visualfea"],col_adaptive),opacity=opacity[1])

    # Plot 4: ----------------------------------
    p.subplot(0, plot_id)
    plot_id += 1

    flowed_plot_func(
        p,
        flowed["points"],
        color_adaptive(flowed["visualfea"],col_adaptive),
        opacity=0.75 if light_mode=="none" else "linear",
    )

    target_plot_func(
        p,
        target["points"],
        color_adaptive(target["visualfea"],col_adaptive),
        opacity=0.75 if light_mode=="none" else "linear",
    )

    #p.add_text(flowed["name"] + "_overlap_" + target["name"], font_size=22)
    p.add_text("overlap", font_size=22)

    # Camera manipulation: -----------------------
    p.link_views()  # link all the views
    finalize_camera(p, camera_pos, show, saving_capture_path, saving_gif_path)

    return p


def visualize_point_pair_overlap(
    pc1_points,
    pc2_points,
    pc1_visualfea,
    pc2_visualfea,
    pc1_name,
    pc2_name,
    pc1_plot_func=default_plot(cmap="magma"),
    pc2_plot_func=default_plot(cmap="viridis"),
    opacity=("linear","linear"),
    **kwargs,
):
    return visualize_full(
        source=None,
        flowed={
            "points": pc1_points,
            "visualfea": pc1_visualfea,
            "name": pc1_name,
        },
        target={
            "points": pc2_points,
            "visualfea": pc2_visualfea,
            "name": pc2_name,
        },
        flowed_plot_func= pc1_plot_func,
        target_plot_func= pc2_plot_func,
        opacity =("linear",opacity[0],opacity[1]),
        **kwargs,
    )


def visualize_source_flowed_target_overlap(
    source_points,
    flowed_points,
    target_points,
    source_visualfea,
    flowed_visualfea,
    target_visualfea,
    source_name,
    flowed_name,
    target_name,
    source_plot_func=default_plot(cmap="magma"),
    flowed_plot_func=default_plot(cmap="magma"),
    target_plot_func=default_plot(cmap="viridis"),
    opacity=("linear", "linear", "linear"),
    **kwargs,
):
    return visualize_full(
        source={
            "points": source_points,
            "visualfea": source_visualfea,
            "name": source_name,
        },
        flowed={
            "points": flowed_points,
            "visualfea": flowed_visualfea,
            "name": flowed_name,
        },
        target={
            "points": target_points,
            "visualfea": target_visualfea,
            "name": target_name,
        },
        source_plot_func=source_plot_func,
        flowed_plot_func=flowed_plot_func,
        target_plot_func=target_plot_func,
        opacity = opacity,
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
                    "source",
                    "flowed",
                    "target",
                    rgb_on=False,
                    saving_capture_path=path,
                    camera_pos=camera_pos,
                    add_bg_contrast=add_bg_contrast,
                    source_plot_func=default_plot(),
                    flowed_plot_func=default_plot(),
                    target_plot_func=default_plot(),
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
                    "source",
                    "flowed",
                    "target",
                    saving_capture_path=path,
                    camera_pos=camera_pos,
                    add_bg_contrast=add_bg_contrast,
                    source_plot_func=default_plot(rgb=True),
                    flowed_plot_func=default_plot(rgb=True),
                    target_plot_func=default_plot(rgb=True),
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
