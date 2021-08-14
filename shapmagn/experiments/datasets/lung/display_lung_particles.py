import vtk
import math
import numpy as np
from argparse import ArgumentParser
from vtk.util.numpy_support import vtk_to_numpy


class DisplayParticles:
    def __init__(
        self,
        file_list,
        spacing_list,
        feature_type_list,
        irad=1.2,
        h_th_list=[],
        glyph_type="sphere",
        glyph_scale_factor=1,
        use_field_data=True,
        opacity_list=[],
        color_list=[],
        lut_list=[],
        lung=[],
    ):

        for feature_type in feature_type_list:
            print(feature_type)
            assert (
                feature_type == "ridge_line"
                or feature_type == "valley_line"
                or feature_type == "ridge_surface"
                or feature_type == "valley_surface"
                or feature_type == "vessel"
                or feature_type == "airway"
                or feature_type == "fissure"
            ), "Invalid feature type"

        for kk, feature_type in enumerate(feature_type_list):

            if feature_type == "airway":
                feature_type_list[kk] = "valley_line"
            elif feature_type == "vessel":
                feature_type_list[kk] = "ridge_line"
            elif feature_type == "fissure":
                feature_type_list[kk] = "ridge_surface"

        self.no_display = False
        self.mapper_list = list()
        self.actor_list = list()
        self.glyph_list = list()
        self.glyph_type = glyph_type
        self.file_list = file_list
        self.spacing_list = spacing_list
        self.opacity_list = opacity_list
        self.irad = irad
        self.h_th_list = h_th_list
        self.color_list = color_list
        self.lut_list = lut_list
        self.lung = lung
        self.use_field_data = use_field_data
        self.feature_type_list = feature_type_list
        self.normal_map = dict()
        self.normal_map["ridge_line"] = "hevec0"
        self.normal_map["valley_line"] = "hevec2"
        self.normal_map["ridge_surface"] = "hevec2"
        self.normal_map["valley_surface"] = "hevec0"
        self.strength_map = dict()
        self.strength_map["ridge_line"] = "h1"
        self.strength_map["valley_line"] = "h1"
        self.strength_map["ridge_surface"] = "h2"
        self.strength_map["valley_surface"] = "h0"

        self.color_by_array_name = None  # By default we color by the particle radius that is computed from scale

        self.glyph_output = None

        self.coordinate_system = "LPS"

        self.lung_opacity = 0.3

        if feature_type == "ridge_line" or feature_type == "valley_line":
            self.height = irad
            self.radius = 0.5
        elif feature_type == "ridge_surface" or feature_type == "valley_surface":
            self.height = 0.5
            self.radius = irad

        self.min_rad = 0.5
        self.min_rad = 0
        self.max_rad = 6
        self.glyph_scale_factor = glyph_scale_factor

        self.capture_prefix = ""
        self.capture_count = 1

        # Use radius from array name, otherwise use scale
        self.radius_array_name_list = None

        # VTK Objects
        self.ren = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.iren = vtk.vtkRenderWindowInteractor()

        self.image_count = 1

        # Picker capabilities
        self.picker = vtk.vtkCellPicker()

        # Display picking results
        self.textMapper = vtk.vtkTextMapper()
        tprop = self.textMapper.GetTextProperty()
        tprop.SetFontFamilyToArial()
        tprop.SetFontSize(10)
        tprop.BoldOn()
        tprop.ShadowOn()
        tprop.SetColor(1, 0, 0)
        self.textActor = vtk.vtkActor2D()
        self.textActor.VisibilityOff()
        self.textActor.SetMapper(self.textMapper)

        # Point locator to extract particle info from picker info on glyhs
        self.particles_locator = vtk.vtkPointLocator()

    def annotatePick(self, object, event):
        print("pick")
        if self.picker.GetCellId() < 0:
            self.textActor.VisibilityOff()
        else:
            selPt = self.picker.GetSelectionPoint()
            pickPos = self.picker.GetPickPosition()
            pId = self.picker.GetPointId()
            # print pId
            particle_pId = self.particles_locator.FindClosestPoint(pickPos)
            print("(%.6f, %.6f, %.6f)" % pickPos)
            print(particle_pId)
            self.textMapper.SetInput("(%.6f, %.6f, %.6f)" % pickPos)
            self.textActor.SetPosition(selPt[:2])
            self.textActor.VisibilityOn()

    def compute_radius(self, poly, spacing, feature_type, radius_array_name, h_th):
        if self.use_field_data == False:
            scale = poly.GetPointData().GetArray("scale")
            strength = poly.GetPointData().GetArray(self.strength_map[feature_type])
            val = poly.GetPointData().GetArray("val")
            if radius_array_name is not None:
                rad_arr = poly.GetPointData().GetArray(radius_array_name)
        else:
            scale = poly.GetFieldData().GetArray("scale")
            strength = poly.GetFieldData().GetArray(self.strength_map[feature_type])
            val = poly.GetFieldData().GetArray("val")
            if radius_array_name is not None:
                rad_arr = poly.GetPointData().GetArray(radius_array_name)

        numpoints = poly.GetNumberOfPoints()
        print(numpoints)
        radiusA = vtk.vtkDoubleArray()
        radiusA.SetNumberOfTuples(numpoints)
        si = float(0.2)
        s0 = float(0.2)

        arr = vtk_to_numpy(strength)
        print(arr[0])
        for kk in range(numpoints):
            if radius_array_name is not None:
                rad = float(rad_arr.GetValue(kk))
            else:
                ss = float(scale.GetValue(kk))
                rad = np.sqrt(2.0) * (
                    np.sqrt(spacing ** 2 * (ss ** 2 + si ** 2)) - 1.0 * spacing * s0
                )
                # rad=np.sqrt(2.0)*spacing*ss
                # rad=np.sqrt(2.0)*np.sqrt(spacing**2 * (ss**2 + si**2) )
            if h_th != None:
                if feature_type == "ridge_line":
                    test = arr[kk] > h_th
                elif feature_type == "valley_line":
                    test = arr[kk] < h_th
                elif feature_type == "ridge_surface":
                    test = arr[kk] > h_th
                elif feature_type == "valley_surface":
                    test = arr[kk] < h_th
            else:
                test = False

            if test == True:
                rad = 0
            if rad < spacing / 2.0:
                print("Setting point to zero " + str(kk))
                rad = 0
            radiusA.SetValue(kk, rad)

        poly.GetPointData().SetScalars(radiusA)
        return poly

    def create_glyphs(self, poly):
        if self.glyph_type == "sphere":
            glyph = vtk.vtkSphereSource()
            glyph.SetRadius(1)
            glyph.SetPhiResolution(8)
            glyph.SetThetaResolution(8)
        elif self.glyph_type == "cylinder":
            glyph = vtk.vtkCylinderSource()
            glyph.SetHeight(self.height)
            glyph.SetRadius(self.radius)
            glyph.SetCenter(0, 0, 0)
            glyph.SetResolution(10)
            glyph.CappingOn()

        tt = vtk.vtkTransform()
        tt.RotateZ(90)
        tf = vtk.vtkTransformPolyDataFilter()
        tf.SetInputConnection(glyph.GetOutputPort())
        tf.SetTransform(tt)
        tf.Update()

        # Alternative use of a Glypher that scales independently along X,Y or Z.
        #        try:
        #          glypher = vtk.vtkGlyph3DWithScaling()
        #          glypher.ScalingXOff()
        #          glypher.ScalingYOn()
        #          glypher.ScalingZOn()
        #        except NameError:
        #          glypher = vtk.vtkGlyph3D()

        glypher = vtk.vtkGlyph3D()

        print(glypher.GetClassName())
        glypher.SetInputData(poly)
        glypher.SetSourceConnection(tf.GetOutputPort())
        glypher.SetVectorModeToUseNormal()
        glypher.SetScaleModeToScaleByScalar()
        glypher.SetScaleFactor(self.glyph_scale_factor)
        glypher.Update()

        return glypher

    def create_lut(self, lut_arr):
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfColors(num_colors)
        lut.SetHueRange(0, 1)
        lut.SetSaturationRange(0, 1)
        lut.SetValueRange(0, 1)
        lut.Build()
        for ii, cc in enumerate(lut_arr):
            lut.SetTableValue(ii, lut_arr[ii])

        return lut

    def create_actor(
        self,
        glyph,
        opacity=1,
        color=[0.1, 0.1, 0.1],
        color_by_array_name=None,
        lut=None,
    ):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetColorModeToMapScalars()
        if color_by_array_name is not None:
            glyph.GetOutput().GetPointData().SetScalars(
                glyph.GetOutput().GetPointData().GetArray(color_by_array_name)
            )
            aa = glyph.GetOutput().GetPointData().GetArray(color_by_array_name)
            range = aa.GetRange()
            mapper.SetScalarRange(range[0], range[1])
            if lut is not None:
                mapper.SetLookupTable(lut)
        else:
            mapper.SetScalarRange(self.min_rad, self.max_rad)
        if len(color) > 0:
            mapper.ScalarVisibilityOff()
        # mapper.SetScalarRange(self.min_rad,self.max_rad)
        # else:
        #    mapper.SetColorModeToDefault()
        print(color)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        if len(color) > 0:
            actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(opacity)
        self.mapper_list.append(mapper)
        self.actor_list.append(actor)
        for aa in self.actor_list:
            self.ren.AddActor(aa)
            self.ren.SetBackground(1, 1, 1)
        return actor

    def add_color_bar(self):
        colorbar = vtk.vtkScalarBarActor()
        colorbar.SetMaximumNumberOfColors(400)
        colorbar.SetLookupTable(self.mapper_list[0].GetLookupTable())
        colorbar.SetWidth(0.09)
        colorbar.SetPosition(0.91, 0.1)
        colorbar.SetLabelFormat("%.3g mm")
        colorbar.VisibilityOn()

        if len(self.color_list) == 0:
            self.ren.AddActor(colorbar)

    def render(self, widht=800, height=800):

        # Now at the end of the pick event call the above function.
        self.picker.AddObserver("EndPickEvent", self.annotatePick)

        # create a rendering window and renderer
        self.renWin.AddRenderer(self.ren)
        self.renWin.SetSize(widht, height)
        #  self.renWin.SetAAFrames(0)

        # create a renderwindowinteractor
        self.iren.SetRenderWindow(self.renWin)

        self.iren.SetPicker(self.picker)

        # add actor
        self.ren.AddActor2D(self.textActor)

        # enable user interface interactor
        # Set observer
        self.iren.AddObserver("KeyPressEvent", self.capture_window, -1.0)

        self.iren.Initialize()
        self.renWin.Render()
        self.iren.Start()

    def execute(self):
        for kk, file_name in enumerate(self.file_list):
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(file_name)
            reader.Update()

            # Locator is link to last particle file for now
            self.particles_locator.SetDataSet(reader.GetOutput())
            self.particles_locator.BuildLocator()

            if len(self.h_th_list) == 0:
                h_th = None
            else:
                h_th = self.h_th_list[kk]

            if self.radius_array_name_list is not None:
                radius_array_name = radius_array_name_list[kk]
                if radius_array_name == "":
                    radius_array_name = None
            else:
                radius_array_name = None

            poly = self.compute_radius(
                reader.GetOutput(),
                self.spacing_list[kk],
                self.feature_type_list[kk],
                radius_array_name,
                h_th,
            )
            if self.use_field_data == False:
                poly.GetPointData().SetNormals(
                    poly.GetPointData().GetArray(
                        self.normal_map[self.feature_type_list[kk]]
                    )
                )
            else:
                poly.GetPointData().SetNormals(
                    poly.GetFieldData().GetArray(
                        self.normal_map[self.feature_type_list[kk]]
                    )
                )

            glypher = self.create_glyphs(poly)
            if len(self.color_list) <= kk:
                color = []
            else:
                color = self.color_list[kk]
            if len(self.opacity_list) <= kk:
                opacity = 1
            else:
                opacity = self.opacity_list[kk]

            if len(self.lut_list) <= kk:
                lut = None
            else:
                lut = self.create_lut(lut_list[kk])

            self.create_actor(
                glypher,
                color=color,
                opacity=opacity,
                lut=lut,
                color_by_array_name=self.color_by_array_name,
            )

            if self.glyph_output is not None:
                tt = vtk.vtkTransform()
                tt.Identity()
                if self.coordinate_system == "RAS":
                    print("Transforming to RAS")
                tt.GetMatrix().SetElement(0, 0, -1)
                tt.GetMatrix().SetElement(1, 1, -1)

                tf = vtk.vtkTransformPolyDataFilter()
                tf.SetTransform(tt)
                tf.SetInputData(glypher.GetOutput())
                tf.SetTransform(tt)
                tf.Update()
                writer = vtk.vtkPolyDataWriter()
                writer.SetInputData(tf.GetOutput())
                writer.SetFileName(self.glyph_output)
                writer.SetFileTypeToBinary()
                writer.Write()

        if len(self.lung) > 0:
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(self.lung)
            reader.Update()
            tt = vtk.vtkTransform()
            tt.Identity()
            if self.coordinate_system == "RAS":
                tt.GetMatrix().SetElement(0, 0, -1)
                tt.GetMatrix().SetElement(1, 1, -1)

            tf = vtk.vtkTransformPolyDataFilter()
            tf.SetTransform(tt)
            tf.SetInputConnection(reader.GetOutputPort())
            tf.SetTransform(tt)
            color = [0.6, 0.6, 0.05]
            # color=[0.8,0.4,0.01]
            self.create_actor(tf, self.lung_opacity, color)

        if self.no_display == False:
            self.add_color_bar()
            self.render()

    def capture_window(self, obj, event):
        if self.capture_prefix == "":
            return
        key = obj.GetKeySym()
        print("Key press " + key)
        if key == "s":
            ff = vtk.vtkWindowToImageFilter()
            sf = vtk.vtkPNGWriter()

            ff.SetInput(self.renWin)
            ff.SetMagnification(4)
            sf.SetInputData(ff.GetOutput())
            sf.SetFileName(self.capture_prefix + "%03d.png" % self.capture_count)
            self.renWin.Render()
            ff.Modified()
            sf.Write()
            self.capture_count = 1 + self.capture_count


if __name__ == "__main__":
    desc = " Visualization of particles vtk files"

    parser = ArgumentParser(description=desc)

    parser.add_argument("-i", help="Input particle files to render", dest="file_name")
    parser.add_argument("-s", help="Input spacing", dest="spacing")
    parser.add_argument(
        "--feature",
        help="Feature type for each particle point. Options are: valley_line (or vessel), ridge_line (or airway), ridge_surface (or fissure) and valley_surface",
        dest="feature_type",
        default="vessel",
    )
    parser.add_argument(
        "--irad", help="Interparticle distance", dest="irad", default=1.2
    )
    parser.add_argument(
        "--hth", help="Threshold on particle strength", dest="hth", default=None
    )
    parser.add_argument("--color", help="RGB color", dest="color_list", default=None)
    parser.add_argument(
        "--opacity", help="Opacity values", dest="opacity_list", default=None
    )
    parser.add_argument(
        "--lut",
        help="Look up table file list for each particle file (comma separated values with R,G,B,Alpha values)",
        dest="lut_list",
        default=None,
    )
    parser.add_argument("-l", help="Lung mesh", dest="lung_filename", default=None)
    parser.add_argument(
        "--useFieldData",
        help="Enable if particle features are stored in Field data instead of Point Data",
        dest="use_field_data",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--glyphScale",
        help="Scaling factor for glyph",
        dest="glyph_scale_factor",
        default=1,
    )
    parser.add_argument(
        "--colorBy", help="Array name to color by", dest="color_by", default=None
    )
    parser.add_argument(
        "--ras",
        help="Set output for RAS",
        dest="ras_coordinate_system",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--glyphOutput",
        help="Output vtk with glpyh poly data",
        dest="glyph_output",
        default=None,
    )
    parser.add_argument(
        "--capturePrefix",
        help='Prefix filename to save screenshots. This options enables screen capture. Press the "s" key to capture a screenshot.',
        dest="capture_prefix",
        default=None,
    )

    parser.add_argument(
        "--radius_name",
        help="Array name with the radius information (optional).\
                                        If this is not provided the radius will be computed from the scale information.",
        dest="radius_array_name",
        metavar="<float>",
        default=None,
    )
    parser.add_argument(
        "--no-display",
        help="No display mode. Objects will be created but not render. It can be used for off-line saving of glyh vtk file",
        dest="nodisplay",
        action="store_true",
    )

    options = parser.parse_args()

    translate_color = dict()
    translate_color["red"] = [1, 0.1, 0.1]
    translate_color["green"] = [0.1, 0.8, 0.1]
    translate_color["orange"] = [0.95, 0.5, 0.01]
    translate_color["blue"] = [0.1, 0.1, 0.9]

    file_list = [i for i in str.split(options.file_name, ",")]
    use_field_data = options.use_field_data
    if options.spacing is not None:
        spacing_list = [float(i) for i in str.split(options.spacing, ",")]

    if options.lung_filename == None:
        lung_filename = ""
    else:
        lung_filename = options.lung_filename

    feature_type_list = [i for i in str.split(options.feature_type, ",")]

    if options.opacity_list == None:
        opacity_list = []
    else:
        opacity_list = [float(i) for i in str.split(options.opacity_list, ",")]

    if options.color_list == None:
        color_list = []
    else:
        color_list = [
            translate_color[val] for val in str.split(options.color_list, ",")
        ]

    if options.hth == None:
        hth_list = []
    else:
        hth_list = [float(i) for i in str.split(options.hth, ",")]

    if options.lut_list == None:
        lut_list = []
    else:
        lut_list = []
        for lut_file in str.split(options.lut_list, ","):
            _df = pd.read_csv(lut_file)
            lut_list.append(_df.values())

    if options.radius_array_name == "" or options.radius_array_name is None:
        radius_array_name_list = None
    else:
        radius_array_name_list = []
        radius_array_name_list = [
            str(i) for i in str.split(options.radius_array_name, ",")
        ]

    dv = DisplayParticles(
        file_list,
        spacing_list,
        feature_type_list,
        float(options.irad),
        hth_list,
        "cylinder",
        float(options.glyph_scale_factor),
        use_field_data,
        opacity_list,
        color_list,
        lut_list,
        lung_filename,
    )
    if options.color_by is not None:
        dv.color_by_array_name = options.color_by
    if options.glyph_output is not None:
        dv.glyph_output = options.glyph_output
    if options.ras_coordinate_system:
        dv.coordinate_system = "RAS"

    dv.no_display = options.nodisplay

    dv.radius_array_name_list = radius_array_name_list

    dv.capture_prefix = options.capture_prefix
    dv.execute()
