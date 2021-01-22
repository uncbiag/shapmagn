import numpy as np
import torch
import pyvista as pv

def visualize_point_fea(points, fea, rgb=True, saving_path=None):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(fea, torch.Tensor):
        fea = fea.detach().cpu().numpy()
    plotter = pv.Plotter()
    plotter.add_mesh(pv.PolyData(points),
                     scalars=fea,
                     cmap="magma", point_size=10,
                     render_points_as_spheres=True,
                     rgb=rgb,
                     opacity="linear",
                     lighting=True,
                     style="points", show_scalar_bar=True)
    plotter.show_grid()
    plotter.show()
    if saving_path:
        data = pv.PolyData(points)
        data.point_arrays['value'] = fea
        data.save(saving_path)


def visualize_point_pair(points1, points2, feas1, feas2, title1, title2, rgb=True, saving_path=None):
    if isinstance(points1, torch.Tensor):
        points1 = points1.detach().cpu().numpy()
    if isinstance(points2, torch.Tensor):
        points2 = points2.detach().cpu().numpy()
    if isinstance(feas1, torch.Tensor):
        feas1 = feas1.detach().cpu().numpy()
    if isinstance(feas2, torch.Tensor):
        feas2 = feas2.detach().cpu().numpy()

    if isinstance(rgb,bool):
        rgb = [rgb]* 2

    p = pv.Plotter(notebook=0, shape=(1, 2), border=False)
    p.subplot(0, 0)
    p.add_text(title1, font_size=18)
    p.add_mesh(pv.PolyData(points1),
                     scalars=feas1,
                     cmap="magma", point_size=10,
                     render_points_as_spheres=True,
                     rgb=rgb[0],
                     opacity="linear",
                     lighting=True,
                     style="points", show_scalar_bar=True)
    p.subplot(0, 1)
    p.add_text(title2, font_size=24)
    p.add_mesh(pv.PolyData(points2),
                     scalars=feas2,
                     cmap="magma", point_size=10,
                     render_points_as_spheres=True,
                     rgb=rgb[1],
                     opacity="linear",
                     lighting=True,
                     style="points", show_scalar_bar=True)

    p.link_views()  # link all the views
    # Set a camera position to all linked views
    p.camera_position = [(5, 5, 0), (0, 0, 0), (0, 1, 0)]

    p.show(auto_close=False)

    if saving_path:
        p.open_gif(saving_path)

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





def visualize_multi_point(points_list, feas_list, titles_list,rgb_on=True, saving_path=None):
    num_views = len(points_list)
    for i,points in enumerate(points_list):
        if isinstance(points, torch.Tensor):
            points_list[i] = points.detach().cpu().numpy()
    for i, feas in enumerate(feas_list):
        if isinstance(feas, torch.Tensor):
            feas_list[i] = feas.detach().cpu().numpy()
    if isinstance(rgb_on,bool):
        rgb_on = [rgb_on]* num_views

    p = pv.Plotter(notebook=0, shape=(1, num_views), border=False)
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

    p.show(auto_close=False)

    if saving_path:
        p.open_gif(saving_path)

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
