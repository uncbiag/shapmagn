import numpy as np
import torch
import pyvista as pv

def visualize_point_fea(points, fea, rgb_on=True, saving_gif_path=None, saving_capture_path=None, show=True):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(fea, torch.Tensor):
        fea = fea.detach().cpu().numpy()
    p = pv.Plotter(off_screen= not show)
    # install pyvistaqt for background plotting that plots without pause the program
    # p = pyvistaqt.BackgroundPlotter(off_screen= not show)
    p.add_mesh(pv.PolyData(points),
                     scalars=fea,
                     cmap="magma", point_size=10,
                     render_points_as_spheres=True,
                     rgb=rgb_on,
                     opacity="linear",
                     lighting=True,
                     style="points", show_scalar_bar=True)
    p.show_grid()
    if show:
        p.show()
    elif saving_capture_path:
        p.show(screenshot=saving_capture_path)

    if saving_gif_path:
        p.open_gif(saving_gif_path)

        # Update camera and write a frame for each updated position
        nframe = 360
        for i in range(nframe):
            p.camera_position = [
                (5 * np.cos(i * np.pi / 180.0), 5 * np.cos(i * np.pi / 180.0), 5 * np.sin(i * np.pi / 180.0)),
                (0, 0, 0),
                (0, 1, 0),
            ]
            p.write_frame()
            p.render()

        # Close movie and delete object
        p.close()



def visualize_point_fea_with_arrow(points, fea, vectors, rgb_on=True, saving_gif_path=None, saving_capture_path=None, show=True):
    if isinstance(points, torch.Tensor):
        points = points.squeeze().detach().cpu().numpy()
    if isinstance(fea, torch.Tensor):
        fea = fea.squeeze().detach().cpu().numpy()
    if isinstance(vectors,torch.Tensor):
        vectors = vectors.squeeze().detach().cpu().numpy()
    p = pv.Plotter(off_screen= not show)
    # install pyvistaqt for background plotting that plots without pause the program
    # p = pyvistaqt.BackgroundPlotter(off_screen= not show)
    point_obj = pv.PolyData(points)
    point_obj.vectors = vectors
    p.add_mesh(point_obj.arrows, scalars='GlyphScale', lighting=False, stitle="Vector Magnitude")
    p.add_mesh(point_obj,
                     scalars=fea,
                     cmap="magma", point_size=10,
                     render_points_as_spheres=True,
                     rgb=rgb_on,
                     opacity="linear",
                     lighting=True,
                     style="points", show_scalar_bar=True)
    p.show_grid()
    if show:
        p.show()
    elif saving_capture_path:
        p.show(screenshot=saving_capture_path)

    if saving_gif_path:
        p.open_gif(saving_gif_path)

        # Update camera and write a frame for each updated position
        nframe = 360
        for i in range(nframe):
            p.camera_position = [
                (5 * np.cos(i * np.pi / 180.0), 5 * np.cos(i * np.pi / 180.0), 5 * np.sin(i * np.pi / 180.0)),
                (0, 0, 0),
                (0, 1, 0),
            ]
            p.write_frame()
            p.render()

        # Close movie and delete object
        p.close()


def visualize_point_pair(points1, points2, feas1, feas2, title1, title2, rgb_on=True, saving_gif_path=None, saving_capture_path=None, show=True):
    if isinstance(points1, torch.Tensor):
        points1 = points1.detach().cpu().numpy()
    if isinstance(points2, torch.Tensor):
        points2 = points2.detach().cpu().numpy()
    if isinstance(feas1, torch.Tensor):
        feas1 = feas1.detach().cpu().numpy()
    if isinstance(feas2, torch.Tensor):
        feas2 = feas2.detach().cpu().numpy()

    if isinstance(rgb_on,bool):
        rgb_on = [rgb_on]* 2

    p = pv.Plotter(notebook=0, shape=(1, 2), border=False,off_screen= not show)
    p.subplot(0, 0)
    p.add_text(title1, font_size=18)
    p.add_mesh(pv.PolyData(points1),
                     scalars=feas1,
                     cmap="magma", point_size=10,
                     render_points_as_spheres=True,
                     rgb=rgb_on[0],
                     opacity="linear",
                     lighting=True,
                     style="points", show_scalar_bar=True)
    p.subplot(0, 1)
    p.add_text(title2, font_size=18)
    p.add_mesh(pv.PolyData(points2),
                     scalars=feas2,
                     cmap="magma", point_size=10,
                     render_points_as_spheres=True,
                     rgb=rgb_on[1],
                     opacity="linear",
                     lighting=True,
                     style="points", show_scalar_bar=True)

    p.link_views()  # link all the views
    # Set a camera position to all linked views
    p.camera_position = [(-5, 5, -5), (0, 0, 0), (0, 1, 0)]

    if show:
        p.show()
    if saving_capture_path:
        p.show(screenshot=saving_capture_path)

    if saving_gif_path:
        p.open_gif(saving_gif_path)

        # Update camera and write a frame for each updated position
        nframe = 360
        for i in range(nframe):
            p.camera_position = [
                (5 * np.cos(i * np.pi / 180.0), 5 * np.cos(i * np.pi / 180.0), 5 * np.sin(i * np.pi / 180.0)),
                (0, 0, 0),
                (0, 1, 0),
            ]
            p.write_frame()
            p.render()

        # Close movie and delete object
        p.close()

def visualize_point_overlap(points1, points2, feas1, feas2, title, point_size=(10,10), rgb_on=True, saving_gif_path=None, saving_capture_path=None, show=True):
    if isinstance(points1, torch.Tensor):
        points1 = points1.detach().cpu().numpy()
    if isinstance(points2, torch.Tensor):
        points2 = points2.detach().cpu().numpy()
    if isinstance(feas1, torch.Tensor):
        feas1 = feas1.detach().cpu().numpy()
    if isinstance(feas2, torch.Tensor):
        feas2 = feas2.detach().cpu().numpy()

    if isinstance(rgb_on, bool):
        rgb_on = [rgb_on] * 2
    p = pv.Plotter(off_screen= not show)
    # install pyvistaqt for background plotting that plots without pause the program
    # p = pyvistaqt.BackgroundPlotter(off_screen= not show)
    p.add_text(title, font_size=18)
    p.add_mesh(pv.PolyData(points1),
               scalars=feas1,
               cmap="viridis", point_size=point_size[0],
               render_points_as_spheres=True,
               rgb=rgb_on[0],
               opacity="linear",
               lighting=True,
               style="points", show_scalar_bar=True)
    p.add_mesh(pv.PolyData(points2),
               scalars=feas2,
               cmap="magma", point_size=point_size[1],
               render_points_as_spheres=True,
               rgb=rgb_on[1],
               opacity="linear",
               lighting=True,
               style="points", show_scalar_bar=True)
    p.show_grid()
    if show:
        p.show()
    elif saving_capture_path:
        p.show(screenshot=saving_capture_path)

    if saving_gif_path:
        p.open_gif(saving_gif_path)

        # Update camera and write a frame for each updated position
        nframe = 360
        for i in range(nframe):
            p.camera_position = [
                (5 * np.cos(i * np.pi / 180.0), 5 * np.cos(i * np.pi / 180.0), 5 * np.sin(i * np.pi / 180.0)),
                (0, 0, 0),
                (0, 1, 0),
            ]
            p.write_frame()
            p.render()

        # Close movie and delete object
        p.close()


def visualize_point_pair_overlap(points1, points2, feas1, feas2, title1, title2, rgb_on=True, saving_gif_path=None, saving_capture_path=None, show=True):
    if isinstance(points1, torch.Tensor):
        points1 = points1.detach().cpu().numpy()
    if isinstance(points2, torch.Tensor):
        points2 = points2.detach().cpu().numpy()
    if isinstance(feas1, torch.Tensor):
        feas1 = feas1.detach().cpu().numpy()
    if isinstance(feas2, torch.Tensor):
        feas2 = feas2.detach().cpu().numpy()

    if isinstance(rgb_on,bool):
        rgb_on = [rgb_on]* 2

    p = pv.Plotter(notebook=0, shape=(1, 3), border=False, off_screen= not show)
    p.subplot(0, 0)
    p.add_text(title1, font_size=18)
    p.add_mesh(pv.PolyData(points1),
                     scalars=feas1,
                     cmap="viridis", point_size=10,
                     render_points_as_spheres=True,
                     rgb=rgb_on[0],
                     opacity="linear",
                     lighting=True,
                     style="points", show_scalar_bar=True)
    p.subplot(0, 1)
    p.add_text(title2, font_size=18)
    p.add_mesh(pv.PolyData(points2),
                     scalars=feas2,
                     cmap="magma", point_size=10,
                     render_points_as_spheres=True,
                     rgb=rgb_on[1],
                     opacity="linear",
                     lighting=True,
                     style="points", show_scalar_bar=True)
    p.subplot(0, 2)
    p.add_text(title1+"_overlap_"+title2, font_size=18)
    p.add_mesh(pv.PolyData(points1),
               scalars=feas1,
               cmap="viridis", point_size=10,
               render_points_as_spheres=True,
               rgb=rgb_on[0],
               opacity="linear",
               lighting=True,
               style="points", show_scalar_bar=True)
    p.add_mesh(pv.PolyData(points2),
               scalars=feas2,
               cmap="magma", point_size=10,
               render_points_as_spheres=True,
               rgb=rgb_on[1],
               opacity="linear",
               lighting=True,
               style="points", show_scalar_bar=True)

    p.link_views()  # link all the views
    # Set a camera position to all linked views
    p.camera_position = [(5, 5, 0), (0, 0, 0), (0, 1, 0)]

    if show:
        p.show()
    if saving_capture_path:
        p.show(screenshot=saving_capture_path)

    if saving_gif_path:
        p.open_gif(saving_gif_path)

        # Update camera and write a frame for each updated position
        nframe = 360
        for i in range(nframe):
            p.camera_position = [
                (5 * np.cos(i * np.pi / 180.0), 5 * np.cos(i * np.pi / 180.0), 5 * np.sin(i * np.pi / 180.0)),
                (0, 0, 0),
                (0, 1, 0),
            ]
            p.write_frame()
            p.render()

        # Close movie and delete object
        p.close()




def visualize_multi_point(points_list, feas_list, titles_list,rgb_on=True, saving_gif_path=None, saving_capture_path=None, show=True):
    num_views = len(points_list)
    for i,points in enumerate(points_list):
        if isinstance(points, torch.Tensor):
            points_list[i] = points.detach().cpu().numpy()
    for i, feas in enumerate(feas_list):
        if isinstance(feas, torch.Tensor):
            feas_list[i] = feas.detach().cpu().numpy()
    if isinstance(rgb_on,bool):
        rgb_on = [rgb_on]* num_views

    p = pv.Plotter(notebook=0, shape=(1, num_views), border=False, off_screen= not show)
    for i in range(num_views):
        p.subplot(0, i)
        p.add_text(titles_list[i], font_size=18)
        p.add_mesh(pv.PolyData(points_list[i]),
                         scalars=feas_list[i],
                         cmap="magma", point_size=10,
                         render_points_as_spheres=True,
                         rgb=rgb_on[i],
                         opacity="linear",
                         lighting=True,
                         style="points", show_scalar_bar=True)
    p.link_views()  # link all the views
    # Set a camera position to all linked views
    p.camera_position = [(7, 7, 0), (0, 0, 0), (0, 1, 0)]

    if show:
        p.show()
    if saving_capture_path:
        p.show(screenshot=saving_capture_path)

    if saving_gif_path:
        p.open_gif(saving_gif_path)

        # Update camera and write a frame for each updated position
        nframe = 360
        for i in range(nframe):
            p.camera_position = [
                (7 * np.cos(i * np.pi / 180.0), 7 * np.cos(i * np.pi / 180.0), 7 * np.sin(i * np.pi / 180.0)),
                (0, 0, 0),
                (0, 1, 0),
            ]
            p.write_frame()
            p.render()

        # Close movie and delete object
        p.close()


