import torch
import torch.nn  as nn
from shapmagn.global_variable import Shape
from shapmagn.utils.obj_factory import obj_factory
from shapmagn.modules.gradient_flow_module import gradient_flow_guide


class GradFlowPreAlign(nn.Module):
    def __init__(self, opt):
        super(GradFlowPreAlign).__init__()
        self.opt = opt
        self.niter = opt[("niter", 10, "self iteration")]
        self.rel_ftol = opt[("rel_ftol", 1e-2, "relative tolerance")]
        self.plot = opt[("plot", False, "plot the shape")]
        self.method_name = opt[("method_name","affine","affine or rigid")]
        pair_feature_extractor_obj = self.opt[("pair_feature_extractor_obj", "", "feature extraction function")]
        self.pair_feature_extractor = obj_factory(pair_feature_extractor_obj) if pair_feature_extractor_obj else None
        self.get_correspondence_shape = self.solve_correspondence_via_gradflow()
        self.solver = self.solve_affine if self.method_name=="affine" else self.solve_rigid


    def set_mode(self, mode):
       self.prealign=True

    @ staticmethod
    def solve_affine(x,y,w):
        # Optimal affine transform: ================================================
        # A = (X^T  @ diag(w) @ X)^-1   @   (X^T  @ diag(w) @ y)
        #    (B,D+1,N)  (B,N,N)  (B,N,D+1)       (B,D+1,N)  (B,N,N)  (B,N,D)
        #
        #   =        Xt_wX    \      Xt_yw
        #           (B,D+1,D+1)       (B,D+1, D)
        # (x, y, z, 1) array to work easily with affine transforms:
        X = torch.cat((x, torch.ones_like(x[:,:, :1])), dim=2)  # (B,N, D+1)

        Xt_wX = X.transpose(2,1) @ (w * X)  # (B,D+1, N) @ (B,N, D+1) = (B,D+1, D+1)
        Xt_wy = X.transpose(2,1) @ (w * y)  # (B,D+1, N) @ (B,N, D)   = (B,D+1, D)

        # Affine transformation:
        A = torch.solve(Xt_wy, Xt_wX).solution  # (B,D+1, D)
        return A, X@A

    def solve_rigid(self,x,y,w):
        """

        :param x: BxNxD
        :param y: BxNxD
        :param w: BxNx1
        :return:
        """
        B,N,D = x.shape[0], x.shape[1],x.shape[2]
        device = x.device
        sum_w = w.sum(1,keepdim=True)
        mu_x = (x*w).sum(1,keepdim=True)/sum_w
        mu_y = (y*w).sum(1,keepdim=True)/sum_w
        x_hat = x - mu_x
        wx_hat = x_hat*w
        y_hat = y - mu_y
        wy_hat = y_hat*w
        a = wy_hat.transpose(2,1)@wx_hat # BxDxN @ BxNxD  BxDxD
        u,s,v = torch.svd(a)
        c = torch.ones(B,D).to(device)
        c[-1] = torch.det(u @ v) #
        r = (u*c)@v.transpose(2,1)
        tr_atr = torch.diagonal(a.transpose(2,1)@r, dim1=-2, dim2=-1).sum(-1)
        tr_xtwx = torch.diagonal(wx_hat.transpose(2,1)@wx_hat, dim1=-2, dim2=-1).sum(-1)
        s= (tr_atr/tr_xtwx)[:,None]
        t = mu_y-s*(r@mu_x.transpose(2,1)).transpose(2,1)
        A = torch.cat([r.transpose(2,1)*s,t],1)
        X = torch.cat((x, torch.ones_like(x[:,:, :1])), dim=2)  # (B,N, D+1)
        return A, X@A


    def compose_transform(self, A_prev, A_cur):
        D = A_prev.shape[-1]
        A_composed_matrix = A_prev[:,:D,:]@A_cur[:,:D,:] # BxDxD
        A_composed_trans = A_prev[:,D:,:]@A_cur[:,:D,:] + A_cur[:,D:,:] # Bx1XD @ BxDxD   Bx1xD
        return torch.cat([A_composed_matrix,A_composed_trans],1)


    def solve_correspondence_via_gradflow(self):
        from functools import partial
        self.gradflow_mode = self.opt[("gradflow_mode","grad_forward"," 'grad_forward' if only use position info otherwise 'ot_mapping'")]
        self.search_init_transform = self.opt[("search_init_transform",False," the 16(2D)/64(3D) initial transforms (based on position and ot similarity) would be searched and return the best one ")]
        self.geomloss_setting= self.opt[("geomloss", {}, "settings for geomloss")]
        return partial(gradient_flow_guide(self.gradflow_mode),geomloss_setting=self.geomloss_setting,local_iter=torch.tensor([0]))


    def _solve_transform(self, source, flowed):
        return self.solver(source.points, flowed.points, source.weights)


    def extract_point_fea(self, flowed, target, iter=-1):
        flowed.pointfea = flowed.points
        target.pointfea = target.points
        return flowed, target

    def extract_fea(self, flowed, target,iter):
        if not self.pair_feature_extractor:
            return self.extract_point_fea(flowed, target,iter)
        else:
            return self.pair_feature_extractor(flowed,target,iter)


    def find_initial_transform(self,source,target):
        import numpy as np
        from scipy.spatial.transform import Rotation as R

        source_center = source.points.mean(dim=1,keepdim=True)
        target_center = target.points.mean(dim=1,keepdim=True)
        max_diameter = lambda x: (x.points.max(1)[0]-x.points.min(1)[0]).max(1)[0]
        scale = max_diameter(target)/max_diameter(source)
        bias_center = (target_center - source_center)/10  # avoid fail into the identity local minimum
        D = source.points.shape[-1]
        n_init = 16 if D==2 else 64
        r=None
        if D==2:
            angle_comp = np.mgrid[0:271:90,0:271:90].transpose(1,2,0).reshape(-1,D)
            r = R.from_euler('yx', angle_comp, degrees=True)
        elif D==3:
            angle_comp = np.mgrid[0:271:90,0:271:90,0:271:90].transpose(1,2,3,0).reshape(-1,D)
            r = R.from_euler('zyx', angle_comp, degrees=True)
        init_rotation_matrix = torch.tensor(r.as_matrix().astype(np.float32)).to(source.points.device)
        init_best_transformed=[]
        init_best_transform=[]
        for i, (b_source_points, b_target_points, b_source_weights,b_target_weights) in \
                enumerate(zip(source.points, target.points,source.weights, target.weights)):
            b_source_points = b_source_points.repeat(n_init,1,1)
            b_target_points = b_target_points.repeat(n_init,1,1)
            b_source_weights = b_source_weights.repeat(n_init,1,1)
            b_target_weights = b_target_weights.repeat(n_init,1,1)
            b_init_rotation_bias = bias_center[i].repeat(n_init,1,1)
            b_transform = torch.cat([init_rotation_matrix*scale[i],b_init_rotation_bias],1)
            geo_dist = obj_factory(self.geomloss_setting["geom_obj"])
            b_init_transformed = torch.cat((b_source_points,torch.ones_like(b_source_points[:,:, :1])), dim=2)@b_transform
            bdist = geo_dist(b_source_weights[...,0],b_init_transformed,b_target_weights[...,0],b_target_points)
            min_val, min_index = bdist.min(0)
            b_init_best_transformed= b_init_transformed[min_index]
            b_init_best_transform = b_transform[min_index]
            print("the best init transform is {}".format(b_init_best_transform))
            init_best_transformed.append(b_init_best_transformed)
            init_best_transform.append(b_init_best_transform)
        return torch.stack(init_best_transform,0), Shape().set_data_with_refer_to(torch.stack(init_best_transformed,0),source)



    def __call__(self,source, target):
        """
           :param source: Shape with points BxNxD
           :param target_batch: Shape with points BxMxD
           :return: Bx(D+1)xD transform matrix
           """
        toflow = source
        A_prev = None
        A = None
        if self.search_init_transform:
            A_prev, toflow= self.find_initial_transform(source, target)
        for i in range(self.niter):
            toflow, target = self.extract_fea(toflow, target,i)
            flowed = self.get_correspondence_shape(toflow, target)
            A, transforme_points = self._solve_transform(toflow, flowed)
            A = self.compose_transform(A_prev,A) if A_prev is not None else A
            transformed_points = torch.cat((source.points,torch.ones_like(source.points[:,:, :1])), dim=2)@A
            toflow = Shape().set_data_with_refer_to(transformed_points,toflow)
            if i>0 and torch.norm(A-A_prev)<self.rel_ftol:
                print("reach relative tolerance {}".format(torch.norm(A-A_prev).item()))
                break
            A_prev = A
            if self.plot:
                self.visualize(source, toflow, target, self.geomloss_setting, i)
        return A



    def visualize(self,source,transformed,target,geomloss_setting, iter):
        from shapmagn.utils.visualizer import visualize_multi_point
        from shapmagn.demos.demo_utils import get_omt_mapping
        mapped_fea = get_omt_mapping(geomloss_setting,source, target,
                                     source.points[0], p=2, mode="hard", confid=0.0)

        visualize_multi_point(points_list=[source.points, transformed.points, target.points],
                              feas_list=[source.points, source.points, mapped_fea],
                              titles_list=["source", "prealiged_iter{}".format(iter), "target"],
                              rgb_on=[True, True, True])