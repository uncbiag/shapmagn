"""
Coherent Point Drift module, here we use the package https://github.com/neka-nat/probreg

type 	    CPD 	            SVR, GMMReg 	        GMMTree 	    FilterReg 	            BCPD (experimental)
Rigid 	Scale + 6D pose 	    6D pose 	            6D pose 	    6D pose                 -
                                                                        (Point-to-point,
                                                                        Point-to-plane,
                                                                        FPFH-based) 	-

NonRigid Affine, MCT 	        TPS 	                  - 	        Deformable Kinematic    Combined model
                                                                        (experimental) 	        (Rigid + Scale + NonRigid-term)

"""

import numpy as np
import torch
from shapmagn.utils.obj_factory import partial_obj_factory
import probreg
if torch.cuda.is_available():
    to_cpu = cp.asnumpy
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
else:
    cp = np
    to_cpu = lambda x: x

METHOD_POOL =["cpd", "svr", "gmmtree", "filterreg_rigid", "filterreg_nonrigid", "BCPD_nonrigid"]


class ProbReg(object):
    """
    https://github.com/neka-nat/probreg/tree/master
    an interface for probreg package
    """

    def __init__(self, opt):
        super(ProbReg, self).__init__()
        self.opt = opt
        self.method_name = opt[("method_name","cpd","supported method name in probreg:"
                            "{}".format(METHOD_POOL))]
        self.solver = getattr(self,"_init_{}".format(self.method_name))\
            (opt[(self.method_name,{},"settings for {}".format(self.method_name))])


    def set_mode(self, mode):
        if mode == "prealign":
            self.prealign = True
        else:
            self.prealign = False


    def _init_cpd(self,opt):
        """CPD Registraion.

            Args:
                source (numpy.ndarray): Source point cloud data.
                target (numpy.ndarray): Target point cloud data.
                tf_type_name (str, optional): Transformation type('rigid', 'affine', 'nonrigid')
                w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
                maxitr (int, optional): Maximum number of iterations to EM algorithm.
                tol (float, optional): Tolerance for termination.
                callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
                    `callback(probreg.Transformation)`

            Keyword Args:
                update_scale (bool, optional): If this flag is true and tf_type is rigid transformation,
                    then the scale is treated. The default is true.
                tf_init_params (dict, optional): Parameters to initialize transformation (for rigid or affine).
            """

        cpd_obj = opt[("cpd_obj","probreg_module.registration_cpd(tf_type_name='affine', w=0.0, maxiter=50, tol=0.001," \
                         " use_cuda=True,callbacks=[])","cpd object")]
        cpd_solver = partial_obj_factory(cpd_obj)
        return cpd_solver

    def _init_svr(self,opt):
        """Support Vector Registration.

           Args:
               source (numpy.ndarray): Source point cloud data.
               target (numpy.ndarray): Target point cloud data.
               tf_type_name (str, optional): Transformation type('rigid', 'nonrigid')
               maxitr (int, optional): Maximum number of iterations for outer loop.
               tol (float, optional): Tolerance for termination of outer loop.
               opt_maxitr (int, optional): Maximum number of iterations for inner loop.
               opt_tol (float, optional): Tolerance for termination of inner loop.
               callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
                   `callback(probreg.Transformation)`
           """
        svr_obj =  opt[("svr_obj","probreg.svr.registration_svr(tf_type_name='rigid', maxitr=1" \
                         "tol=1.0e-3, opt_maxitr=50,opt_tol=1.0e-3,callbacks=[])","svr object")]
        svr_sovler = partial_obj_factory(svr_obj)
        return svr_sovler

    def _init_gmmtree(self,opt):
        """GMMTree registration

           Args:
               source (numpy.ndarray): Source point cloud data.
               target (numpy.ndarray): Target point cloud data.
               maxitr (int, optional): Maximum number of iterations to EM algorithm.
               tol (float, optional): Tolerance for termination.
               callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
                   `callback(probreg.Transformation)`

           Keyword Args:
               default: tree_level=2, lambda_c=0.01, lambda_s=0.001,
               tree_level (int, optional): Maximum depth level of GMM tree.
               lambda_c (float, optional): Parameter that determine the pruning of GMM tree.
               lambda_s (float, optional): Parameter that tolerance for building GMM tree.
               tf_init_params (dict, optional): Parameters to initialize transformation.
           """
        gmmtree_obj = opt[("gmmtree_obj", "probreg.gmmtree.registration_svr(maxitr=20, tol=1.0e-4" \
                                  "callbacks=[])", "gmmtree object")]
        gmmtree_sovler = partial_obj_factory(gmmtree_obj)
        return gmmtree_sovler


    def _init_filterreg(self,opt):
        """FilterReg registration

           Args:
               source (numpy.ndarray): Source point cloud data.
               target (numpy.ndarray): Target point cloud data.
               target_normals (numpy.ndarray, optional): Normal vectors of target point cloud.
               sigma2 (float, optional): Variance of GMM. If `sigma2` is `None`, `sigma2` is automatically updated.
               w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
               objective_type (str, optional): The type of objective function selected by 'pt2pt' or 'pt2pl'.
               maxitr (int, optional): Maximum number of iterations to EM algorithm.
               tol (float, optional): Tolerance for termination.
               min_sigma2 (float, optional): Minimum variance of GMM.
               feature_fn (function, optional): Feature function. If you use FPFH feature, set `feature_fn=probreg.feature.FPFH()`.
               callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
                   `callback(probreg.Transformation)`

           Keyword Args:
               tf_init_params (dict, optional): Parameters to initialize transformation (for rigid).
           """
        filterreg_obj = opt[("filterreg_obj", "probreg.filterreg.registration_filterreg(sigma2=None, update_sigma2=True, w=0, "
                                            "objective_type='pt2pt', maxiter=50, tol=0.001, min_sigma2=1.0e-4, "
                                            "feature_fn=lambda x: x", "filterreg object")]
        filterreg_sovler = partial_obj_factory(filterreg_obj)
        feature_fn = probreg.features.FPFH()
        filterreg_sovler.keywords["feature_fn"] = feature_fn
        return filterreg_sovler

    def _init_bcpd(self,opt):
        """BCPD Registraion.

            Args:
                source (numpy.ndarray): Source point cloud data.
                target (numpy.ndarray): Target point cloud data.
                w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
                maxitr (int, optional): Maximum number of iterations to EM algorithm.
                tol (float, optional) : Tolerance for termination.
                callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
                    `callback(probreg.Transformation)`
            """

        bcpd_obj = opt[("bcpd_obj", "probreg.bcpd.registration_bcpd(maxitr=20, w=0.0, maxiter=50, tol=0.001, callbacks=[])"
                        , "bcpd object")]
        bcpd_sovler = partial_obj_factory(bcpd_obj)
        return bcpd_sovler

    def __call__(self,source, target):
        """
           :param source: Shape with points BxNxD
           :param target_batch: Shape with points BxMxD
           :return: Bx(D+1)xD transform matrix
           """
        source_batch, target_batch = source.points, target.points
        device = source_batch.device
        source_list, target_list = self._get_input(source_batch, target_batch)
        solution_list = []
        for source, target in zip(source_list, target_list):
            solution = self.solver(source,target)
            solution_list.append(solution)
        return self._convert_transform_format(solution_list, device)




    def _get_input(self, source_batch, target_batch):
        device_warpper = lambda x: cp.asarray(x) if self.method_name in ["cpd"] else np.asarray(x)
        source_list = [device_warpper(source) for source in source_batch.detach().cpu().numpy()]
        target_list = [device_warpper(target) for target in target_batch.detach().cpu().numpy()]
        return source_list, target_list

    def _get_transform_matrix(self, solution):
        if self.method_name in ["cpd"]:
            try:
                return (to_cpu(solution[0].b).T).astype(np.float32)
            except:
                return (to_cpu(solution[0].rot).T * to_cpu(solution[0].scale)).astype(np.float32)
        else:
            return ((solution[0].rot).T * solution[0].scale).astype(np.float32)

    def _get_translation(self,solution):
        if self.method_name in ["cpd"]:
            return (to_cpu(solution[0].t)[None]).astype(np.float32)
        else:
            return ((solution[0].t)[None]).astype(np.float32)


    def _get_deformation(self,solution):
        #todo to implement
        pass


    def _get_prealign_deformation(self,solution_list, device):
        transform_matrix_list = [self._get_transform_matrix(solution) for solution in solution_list]
        translation_list = [self._get_translation(solution) for solution in solution_list]
        batch_transform_matrix = torch.from_numpy(np.stack(transform_matrix_list))
        batch_translation = torch.from_numpy(np.stack(translation_list))
        return_param = torch.cat((batch_transform_matrix, batch_translation), dim=1)
        return return_param.to(device)


    def _convert_transform_format(self,solution_list, device):
        convert_fn = self._get_prealign_deformation if self.prealign else None
        return convert_fn(solution_list,device)





###############   fix cupy  input ########################
def registration_cpd(source, target, tf_type_name='rigid',
                     w=0.0, maxiter=50, tol=0.001,
                     callbacks=[], **kargs):
    """CPD Registraion.

    Args:
        source (numpy.ndarray): Source point cloud data.
        target (numpy.ndarray): Target point cloud data.
        tf_type_name (str, optional): Transformation type('rigid', 'affine', 'nonrigid')
        w (float, optional): Weight of the uniform distribution, 0 < `w` < 1.
        maxitr (int, optional): Maximum number of iterations to EM algorithm.
        tol (float, optional): Tolerance for termination.
        callback (:obj:`list` of :obj:`function`, optional): Called after each iteration.
            `callback(probreg.Transformation)`

    Keyword Args:
        update_scale (bool, optional): If this flag is true and tf_type is rigid transformation,
            then the scale is treated. The default is true.
        tf_init_params (dict, optional): Parameters to initialize transformation (for rigid or affine).
    """
    from probreg.cpd import RigidCPD,AffineCPD,NonRigidCPD
    if tf_type_name == 'rigid':
        cpd = RigidCPD(source, **kargs)
    elif tf_type_name == 'affine':
        cpd = AffineCPD(source, **kargs)
    elif tf_type_name == 'nonrigid':
        cpd = NonRigidCPD(source, **kargs)
    else:
        raise ValueError('Unknown transformation type %s' % tf_type_name)
    cpd.set_callbacks(callbacks)
    return cpd.registration(target, w, maxiter, tol)