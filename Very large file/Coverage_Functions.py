import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import trimesh
import pyrender


def plot_mesh(mesh):
    '''
    This function allows you to render the mesh file with a rotating effect

    :param mesh: Mesh file
    :return: No return
    - Must install pyrender
        pip install pyrender
    - Must install pyglet
        conda install -c conda-forge pyglet
    - Info on pyrender:
        https://readthedocs.org/projects/pyrender/downloads/pdf/stable/

     @author: ashao, jmbouteiller
    '''
    scene = pyrender.Scene()
    render_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    # oc: camera Orthographic
    # oc = pyrender.OrthographicCamera(xmag=0.01, ymag=0.01,znear=10.0)
    # pc: camera perspective
    # pc = pyrender.PerspectiveCamera(yfov=np.pi / 1.5, aspectRatio=1.414)
    # nm: render node
    nm = pyrender.Node(mesh=render_mesh, matrix=np.eye(4))
    # nc: camera node
    # nc = pyrender.Node(camera=oc, matrix=np.eye(4))
    scene.add_node(nm)
    # scene.add_node(nc)
    pyrender.Viewer(scene, use_raymond_lighting=True
                    , rotate=False
                    , rotate_axis=[0, 0, 1], view_center=[0, -30, -20]
                    , use_perspective_cam=False
                    , show_mesh_axes=True)

    # NOTES:
    #     1. camera should point center of gravity. Trimesh provides access to C of G.
    #     2. Heatmap
    #     3. Refine figure on coverage % (x axis is velocity, y axis is percentage coverage)
    #     4. Save gif
    #     5. Save different views
    #
    #   JMB+Javad: rotation and C of G on Trimesh
    #   Note: Pyrender: ability to add cameras to scene
    #   Arthur: heatmap done
    #   JMB: save figure
    #   Arthur: coverage %


def Particle_dep(mesh_file, particle_file, excel_index):
    '''

    :param mesh_file:
    :param particle_file:
    :param excel_index:
    :return:Array with each Particles landed on the mesh file
    '''

    my_data = pd.read_csv(particle_file, delim_whitespace=True, comment='%', header=0, names=excel_index)
    clean_data = my_data.dropna()
    your_mesh = trimesh.load(mesh_file)
    P0 = clean_data[['x', 'y', 'z']].to_numpy()
    # Retrieve number of particles
    length = clean_data['size'].to_numpy()
    ##Calculate whether a particle falls into the triangle
    [closid, dist, id] = trimesh.proximity.closest_point(your_mesh,P0)
    # closid is the closest point on triangles for each point,
    # dist is the distance to mesh,
    # id is the index of triangle containing closest point.
    UniqID = list(set(id))  # UniquID shows each individual triangle id which particles land on
    # To determine if a particle falls into a triangle, the distance must be smaller or equal to the radius of the particle. Dist here is the distance and length/2 is the radius
    count = 0
    # Define opacity/transparency for all triangles (last number as 100 instead of 255)
    IDs = []
    for i in range(len(dist)):
        if dist[i] < length[i] / 2.0:
            IDs.append(P0[i])
    return(IDs)

def total_area(mesh_file, particle_file, excel_index, var, plot, export):
    '''
    This function determines whether a particle lands on a mesh file. Depending on how many particles the mesh has been
    landed on, the color of the mesh will change which forms a heat map. You can directly render the modified mesh from
    the function and export the gif.

    :param mesh_file: Mesh file txt file from COMSOL
    :param particle_file: Particle position txt file from COMSOL
    :param excel_index: Which parameter you choose to export from COMSOL
    :param var: Variable of interest you want to analyze eg: velocity, cone angle, just for export filename purpose
    :param plot: Whether you want to render the colored mesh
    :param export: Whether you want to exported the colored mesh
    :return: Return Percentage of coverage

    Notes:
    - Must first install trimesh. To do this from anaconda command prompt:
        > conda install -c conda-forge trimesh
    - might want to install networkx too:
        > conda install -c conda-forge networkx
    - Must install numpy-stl:
        conda install -c conda-forge numpy-stl
    - Must install pandas too:
        conda install -c conda-forge pandas
    - Must install pyrender
        pip install pyrender
    - Must install pyglet
        conda install -c conda-forge pyglet

    - more info on trimesh here:
        https://github.com/mikedh/trimesh
    - Has the functionality to evaluate distance between point and mesh
        more info here: https://github.com/mikedh/trimesh/blob/master/examples/nearest.ipynb
    - Info on pyrender:
        https://readthedocs.org/projects/pyrender/downloads/pdf/stable/

    -other note: choose python library
        https://softwarerecs.stackexchange.com/questions/64807/which-mesh-processing-library-for-python-to-chose

    - Specs for OBJ file: https://en.wikipedia.org/wiki/Wavefront_.obj_file

    @author: ashao, jmbouteiller
    '''
    cmap = plt.get_cmap('gist_heat')
    my_data = pd.read_csv(particle_file, delim_whitespace=True, comment='%', header=0, names=excel_index)
    clean_data = my_data.dropna()
    your_mesh = trimesh.load(mesh_file)
    P0 = clean_data[['x', 'y', 'z']].to_numpy()
    # Retrieve number of particles
    length = clean_data['size'].to_numpy()
    fileName = '' + str(date.today().isoformat()) + '_deposition_profile_along_' + var + '_for_' + mesh_file
    ##Calculate whether a particle falls into the triangle
    [closid, dist, id] = trimesh.proximity.closest_point(your_mesh,
                                                         P0)  # closid is the closest point on triangles for each point, dist is the distance to mesh, id is the index of triangle containing closest point.
    UniqID = list(set(id))  # UniquID shows each individual triangle id which particles land on
    # To determine if a particle falls into a triangle, the distance must be smaller or equal to the radius of the particle. Dist here is the distance and length/2 is the radius
    count = 0
    # Define opacity/transparency for all triangles (last number as 100 instead of 255)
    your_mesh.visual.face_colors = [100, 100, 100, 255]  # change the color of the triangle
    IDs = np.zeros(max(id))
    for i in range(len(dist)):
        if dist[i] < length[i] / 2.0:
            count = count + 1
            IDs[id[i] - 1] += 1
    rangR = max(IDs)
    for i in range(len(IDs)):

        if IDs[i] != 0:
            your_mesh.visual.face_colors[i][:2] = np.array(cmap((IDs[i] / rangR))[:2]) * 255

    ttarea = trimesh.triangles.area(your_mesh.triangles, sum=True)
    area = trimesh.triangles.area(your_mesh.triangles[UniqID], sum=True)
    if export:
        your_mesh.export(fileName)
    if plot:
        your_mesh.export(fileName)
        importmesh = trimesh.load(fileName)
        plot_mesh(importmesh)

    Percent = area / ttarea * 100
    print('Converage Percentage: ' + str(area / ttarea * 100) + '%')
    print('Converage Area: ' + str(area / 100) + 'cm^2')
    print(Percent)
    return (Percent)

def total_area_Data(mesh_file, particle_data, var, plot, export):
    '''
    This function determines whether a particle lands on a mesh file. Depending on how many particles the mesh has been
    landed on, the color of the mesh will change which forms a heat map. You can directly render the modified mesh from
    the function and export the gif.

    :param mesh_file: Mesh file txt file from COMSOL
    :param particle_file: Particle position txt file from COMSOL
    :param excel_index: Which parameter you choose to export from COMSOL
    :param var: Variable of interest you want to analyze eg: velocity, cone angle, just for export filename purpose
    :param plot: Whether you want to render the colored mesh
    :param export: Whether you want to exported the colored mesh
    :return: Return Percentage of coverage

    Notes:
    - Must first install trimesh. To do this from anaconda command prompt:
        > conda install -c conda-forge trimesh
    - might want to install networkx too:
        > conda install -c conda-forge networkx
    - Must install numpy-stl:
        conda install -c conda-forge numpy-stl
    - Must install pandas too:
        conda install -c conda-forge pandas
    - Must install pyrender
        pip install pyrender
    - Must install pyglet
        conda install -c conda-forge pyglet

    - more info on trimesh here:
        https://github.com/mikedh/trimesh
    - Has the functionality to evaluate distance between point and mesh
        more info here: https://github.com/mikedh/trimesh/blob/master/examples/nearest.ipynb
    - Info on pyrender:
        https://readthedocs.org/projects/pyrender/downloads/pdf/stable/

    -other note: choose python library
        https://softwarerecs.stackexchange.com/questions/64807/which-mesh-processing-library-for-python-to-chose

    - Specs for OBJ file: https://en.wikipedia.org/wiki/Wavefront_.obj_file

    @author: ashao, jmbouteiller
    '''
    cmap = plt.get_cmap('gist_heat')
    clean_data = particle_data
    if clean_data.shape[0]:
        your_mesh = trimesh.load(mesh_file)
        P0 = clean_data[['x', 'y', 'z']].to_numpy()
        # Retrieve number of particles
        length = clean_data['size'].to_numpy()
        fileName = '' + str(date.today().isoformat()) + '_deposition_profile_for_' + mesh_file
        ##Calculate whether a particle falls into the triangle
        [closid, dist, id] = trimesh.proximity.closest_point(your_mesh,
                                                         P0)  # closid is the closest point on triangles for each point, dist is the distance to mesh, id is the index of triangle containing closest point.
        count = 0
        IDs = np.zeros(len(id))
        for i in range(len(id) - 1):
            if dist[i] < length[i] / 2.0:
                IDs[count] = id[i]
                count = count + 1
        UniqID = np.unique(id)  # UniquID shows each individual triangle id which particles land on
        # To determine if a particle falls into a triangle, the distance must be smaller or equal to the radius of the particle. Dist here is the distance and length/2 is the radius

        #Visualize Deposition
        count = 0
        # Define opacity/transparency for all triangles (last number as 100 instead of 255)
        your_mesh.visual.face_colors = [100, 100, 100, 255]  # change the color of the triangle
        IDs = np.zeros(max(id))
        for i in range(len(dist)):
            if dist[i] < length[i] / 2.0:
                count = count + 1
                IDs[id[i] - 1] += 1
        rangR = max(IDs)
        for i in range(len(IDs)):
            if IDs[i] != 0:
                your_mesh.visual.face_colors[i][:2] = np.array(cmap((IDs[i] / rangR))[:2]) * 255


        ttarea = trimesh.triangles.area(your_mesh.triangles, sum=True)
        area = trimesh.triangles.area(your_mesh.triangles[UniqID], sum=True)
        if export:
            your_mesh.export(fileName)
        if plot:
            your_mesh.export(fileName)
            importmesh = trimesh.load(fileName)
            plot_mesh(importmesh)

        Percent = area / ttarea * 100
        print('Converage Percentage: ' + str(area / ttarea * 100) + '%')
        print('Converage Area: ' + str(area / 100) + 'cm^2')
        print(Percent)
    else:
        Percent = 0
    return (Percent)

def filter_Spray(mesh_file, particle_data):
    '''
    This function determines whether a particle lands on a mesh file. Depending on how many particles the mesh has been
    landed on, the color of the mesh will change which forms a heat map. You can directly render the modified mesh from
    the function and export the gif.

    :param mesh_file: Mesh file txt file from COMSOL
    :param particle_file: Particle position txt file from COMSOL
    :param excel_index: Which parameter you choose to export from COMSOL
    :param var: Variable of interest you want to analyze eg: velocity, cone angle, just for export filename purpose
    :param plot: Whether you want to render the colored mesh
    :param export: Whether you want to exported the colored mesh
    :return: Return Percentage of coverage

    Notes:
    - Must first install trimesh. To do this from anaconda command prompt:
        > conda install -c conda-forge trimesh
    - might want to install networkx too:
        > conda install -c conda-forge networkx
    - Must install numpy-stl:
        conda install -c conda-forge numpy-stl
    - Must install pandas too:
        conda install -c conda-forge pandas
    - Must install pyrender
        pip install pyrender
    - Must install pyglet
        conda install -c conda-forge pyglet

    - more info on trimesh here:
        https://github.com/mikedh/trimesh
    - Has the functionality to evaluate distance between point and mesh
        more info here: https://github.com/mikedh/trimesh/blob/master/examples/nearest.ipynb
    - Info on pyrender:
        https://readthedocs.org/projects/pyrender/downloads/pdf/stable/

    -other note: choose python library
        https://softwarerecs.stackexchange.com/questions/64807/which-mesh-processing-library-for-python-to-chose

    - Specs for OBJ file: https://en.wikipedia.org/wiki/Wavefront_.obj_file

    @author: ashao, jmbouteiller
    '''
    clean_data = particle_data
    your_mesh = trimesh.load(mesh_file)
    P0 = clean_data[['x', 'y', 'z']].to_numpy()
    length = clean_data['size'].to_numpy()
    ##Calculate whether a particle falls into the triangle
    [closid, dist, id] = trimesh.proximity.closest_point(your_mesh,
                                                         P0)
    # closid is the closest point on triangles for each point, dist is the distance to mesh, id is the index of triangle containing closest point.
    # To determine if a particle falls into a triangle, the distance must be smaller or equal to the radius of the particle. Dist here is the distance and length/2 is the radius
    count = 0
    IDs = np.zeros(len(id))
    for i in range(len(id)-1):
        if dist[i] < length[i] / 2.0:
            IDs[count] = id[i]
            count = count + 1
    UniqID = np.unique(IDs[IDs != 0])  # UniquID shows each individual triangle id which particles land on
    return (UniqID)

def total_area_filterData(mesh_file, filter_ID,particle_data, plot, export):
    '''
    This function determines whether a particle lands on a mesh file. Depending on how many particles the mesh has been
    landed on, the color of the mesh will change which forms a heat map. You can directly render the modified mesh from
    the function and export the gif.

    :param mesh_file: Mesh file txt file from COMSOL
    :param particle_file: Particle position txt file from COMSOL
    :param excel_index: Which parameter you choose to export from COMSOL
    :param var: Variable of interest you want to analyze eg: velocity, cone angle, just for export filename purpose
    :param plot: Whether you want to render the colored mesh
    :param export: Whether you want to exported the colored mesh
    :return: Return Percentage of coverage

    Notes:
    - Must first install trimesh. To do this from anaconda command prompt:
        > conda install -c conda-forge trimesh
    - might want to install networkx too:
        > conda install -c conda-forge networkx
    - Must install numpy-stl:
        conda install -c conda-forge numpy-stl
    - Must install pandas too:
        conda install -c conda-forge pandas
    - Must install pyrender
        pip install pyrender
    - Must install pyglet
        conda install -c conda-forge pyglet

    - more info on trimesh here:
        https://github.com/mikedh/trimesh
    - Has the functionality to evaluate distance between point and mesh
        more info here: https://github.com/mikedh/trimesh/blob/master/examples/nearest.ipynb
    - Info on pyrender:
        https://readthedocs.org/projects/pyrender/downloads/pdf/stable/

    -other note: choose python library
        https://softwarerecs.stackexchange.com/questions/64807/which-mesh-processing-library-for-python-to-chose

    - Specs for OBJ file: https://en.wikipedia.org/wiki/Wavefront_.obj_file

    @author: ashao, jmbouteiller
    '''
    cmap = plt.get_cmap('gist_heat')
    clean_data = particle_data
    if clean_data.shape[0]:
        your_mesh = trimesh.load(mesh_file)
        P0 = clean_data[['x', 'y', 'z']].to_numpy()
        # Retrieve number of particles
        length = clean_data['size'].to_numpy()
        fileName = '' + str(date.today().isoformat()) + '_deposition_profile_for_' + mesh_file
        ##Calculate whether a particle falls into the triangle
        [closid, dist, id] = trimesh.proximity.closest_point(your_mesh,
                                                         P0)  # closid is the closest point on triangles for each point, dist is the distance to mesh, id is the index of triangle containing closest point.
        count = 0
        IDs = np.zeros(len(id))
        for i in range(len(id) - 1):
            if dist[i] < length[i] / 2.0:
                IDs[count] = id[i]
                count = count + 1
        UniqID = np.unique(id)

        for tepid in filter_ID:
            UniqID = UniqID[UniqID != tepid]
        UniqID = UniqID[UniqID != 0]# UniquID shows each individual triangle id which particles land on
    # To determine if a particle falls into a triangle, the distance must be smaller or equal to the radius of the particle. Dist here is the distance and length/2 is the radius
    # count = 0
    # # Define opacity/transparency for all triangles (last number as 100 instead of 255)
    # your_mesh.visual.face_colors = [100, 100, 100, 255]  # change the color of the triangle
    # IDs = np.zeros(max(id))
    # for i in range(len(dist)):
    #     if dist[i] < length[i] / 2.0:
    #         count = count + 1
    #         IDs[id[i] - 1] += 1
    # rangR = max(IDs)
    # for i in range(len(IDs)):
    #
    #     if IDs[i] != 0:
    #         your_mesh.visual.face_colors[i][:2] = np.array(cmap((IDs[i] / rangR))[:2]) * 255
    #
    # def condition(x): return x != 0
    # deposID = np.where(condition(IDs))
    # deposID = np.unique([deposID,filter_ID])

        ttarea = trimesh.triangles.area(your_mesh.triangles, sum=True)
        area = trimesh.triangles.area(your_mesh.triangles[UniqID], sum=True)
        if export:
            your_mesh.export(fileName)
        if plot:
            your_mesh.export(fileName)
            importmesh = trimesh.load(fileName)
            plot_mesh(importmesh)

        Percent = area / ttarea * 100
        print('Converage Percentage: ' + str(area / ttarea * 100) + '%')
        print('Converage Area: ' + str(area / 100) + 'cm^2')
        print(Percent)
    else:
        Percent = 0
    return (Percent)

def total_Volume_Data(mesh_file, particle_data, var, plot, export):
    '''
    This function determines whether a particle lands on a mesh file. Depending on how many particles the mesh has been
    landed on, the color of the mesh will change which forms a heat map. You can directly render the modified mesh from
    the function and export the gif.

    :param mesh_file: Mesh file txt file from COMSOL
    :param particle_file: Particle position txt file from COMSOL
    :param excel_index: Which parameter you choose to export from COMSOL
    :param var: Variable of interest you want to analyze eg: velocity, cone angle, just for export filename purpose
    :param plot: Whether you want to render the colored mesh
    :param export: Whether you want to exported the colored mesh
    :return: Return Percentage of coverage

    Notes:
    - Must first install trimesh. To do this from anaconda command prompt:
        > conda install -c conda-forge trimesh
    - might want to install networkx too:
        > conda install -c conda-forge networkx
    - Must install numpy-stl:
        conda install -c conda-forge numpy-stl
    - Must install pandas too:
        conda install -c conda-forge pandas
    - Must install pyrender
        pip install pyrender
    - Must install pyglet
        conda install -c conda-forge pyglet

    - more info on trimesh here:
        https://github.com/mikedh/trimesh
    - Has the functionality to evaluate distance between point and mesh
        more info here: https://github.com/mikedh/trimesh/blob/master/examples/nearest.ipynb
    - Info on pyrender:
        https://readthedocs.org/projects/pyrender/downloads/pdf/stable/

    -other note: choose python library
        https://softwarerecs.stackexchange.com/questions/64807/which-mesh-processing-library-for-python-to-chose

    - Specs for OBJ file: https://en.wikipedia.org/wiki/Wavefront_.obj_file

    @author: ashao, jmbouteiller
    '''
    cmap = plt.get_cmap('gist_heat')
    clean_data = particle_data
    if clean_data.shape[0]:
        your_mesh = trimesh.load(mesh_file)
        P0 = clean_data[['x', 'y', 'z']].to_numpy()
        # Retrieve number of particles
        length = clean_data['size'].to_numpy() #length is the diameter of each particle
        fileName = '' + str(date.today().isoformat()) + '_deposition_profile_for_' + mesh_file
        ##Calculate whether a particle falls into the triangle
        [closid, dist, id] = trimesh.proximity.closest_point(your_mesh,
                                                         P0)  # closid is the closest point on triangles for each point, dist is the distance to mesh, id is the index of triangle containing closest point.
        vol = 0
        # count = 0
        # IDs = np.zeros(len(id))
        for i in range(len(id) - 1):
            if dist[i] < length[i] / 2.0:
                r = length[i]/2
                # IDs[count] = id[i]
                # count = count + 1
                vol = r**3*np.pi*4/3 + vol
        # UniqID = np.unique(id)  # UniquID shows each individual triangle id which particles land on
        # # To determine if a particle falls into a triangle, the distance must be smaller or equal to the radius of the particle. Dist here is the distance and length/2 is the radius
        # count = 0
        # # Define opacity/transparency for all triangles (last number as 100 instead of 255)
        # your_mesh.visual.face_colors = [100, 100, 100, 255]  # change the color of the triangle
        # IDs = np.zeros(max(id))
        # for i in range(len(dist)):
        #     if dist[i] < length[i] / 2.0:
        #         count = count + 1
        #         IDs[id[i] - 1] += 1
        # rangR = max(IDs)
        # for i in range(len(IDs)):
        #     if IDs[i] != 0:
        #         your_mesh.visual.face_colors[i][:2] = np.array(cmap((IDs[i] / rangR))[:2]) * 255
        # # def condition(x): return x != 0
        # # deposID = np.where(condition(IDs))
        # # deposID = np.unique(deposID)
        # ttarea = trimesh.triangles.area(your_mesh.triangles, sum=True)
        # area = trimesh.triangles.area(your_mesh.triangles[UniqID], sum=True)
        if export:
            your_mesh.export(fileName)
        if plot:
            your_mesh.export(fileName)
            importmesh = trimesh.load(fileName)
            plot_mesh(importmesh)

        # Percent = area / ttarea * 100
        # print('Converage Percentage: ' + str(area / ttarea * 100) + '%')
        # print('Converage Area: ' + str(area / 100) + 'cm^2')
        # print(Percent)
        print(vol)
    else:
        # Percent = 0
        vol = 0
    return (vol)

def total_Volume_filterData(mesh_file, filter_ID, particle_data, plot, export):
    '''
    This function determines whether a particle lands on a mesh file. Depending on how many particles the mesh has been
    landed on, the color of the mesh will change which forms a heat map. You can directly render the modified mesh from
    the function and export the gif.

    :param mesh_file: Mesh file txt file from COMSOL
    :param particle_file: Particle position txt file from COMSOL
    :param excel_index: Which parameter you choose to export from COMSOL
    :param var: Variable of interest you want to analyze eg: velocity, cone angle, just for export filename purpose
    :param plot: Whether you want to render the colored mesh
    :param export: Whether you want to exported the colored mesh
    :return: Return Percentage of coverage

    Notes:
    - Must first install trimesh. To do this from anaconda command prompt:
        > conda install -c conda-forge trimesh
    - might want to install networkx too:
        > conda install -c conda-forge networkx
    - Must install numpy-stl:
        conda install -c conda-forge numpy-stl
    - Must install pandas too:
        conda install -c conda-forge pandas
    - Must install pyrender
        pip install pyrender
    - Must install pyglet
        conda install -c conda-forge pyglet

    - more info on trimesh here:
        https://github.com/mikedh/trimesh
    - Has the functionality to evaluate distance between point and mesh
        more info here: https://github.com/mikedh/trimesh/blob/master/examples/nearest.ipynb
    - Info on pyrender:
        https://readthedocs.org/projects/pyrender/downloads/pdf/stable/

    -other note: choose python library
        https://softwarerecs.stackexchange.com/questions/64807/which-mesh-processing-library-for-python-to-chose

    - Specs for OBJ file: https://en.wikipedia.org/wiki/Wavefront_.obj_file

    @author: ashao, jmbouteiller
    '''
    cmap = plt.get_cmap('gist_heat')
    clean_data = particle_data
    if clean_data.shape[0]:
        your_mesh = trimesh.load(mesh_file)
        P0 = clean_data[['x', 'y', 'z']].to_numpy()
        # Retrieve number of particles
        length = clean_data['size'].to_numpy()
        fileName = '' + str(date.today().isoformat()) + '_deposition_profile_for_' + mesh_file
        ##Calculate whether a particle falls into the triangle
        [closid, dist, id] = trimesh.proximity.closest_point(your_mesh,
                                                         P0)  # closid is the closest point on triangles for each point, dist is the distance to mesh, id is the index of triangle containing closest point.
        vol = 0
        # count = 0
        # IDs = np.zeros(len(id))
        for i in range(len(id) - 1):
            if dist[i] < length[i] / 2.0 and id[i] not in filter_ID:
                r = length[i] / 2
                vol = r ** 3 * np.pi * 4 / 3 + vol
                # IDs[count] = id[i]
                # count = count + 1
        # UniqID = np.unique(id)

        # for tepid in filter_ID:
        #     UniqID = UniqID[UniqID != tepid]
        # UniqID = UniqID[UniqID != 0]# UniquID shows each individual triangle id which particles land on
    # To determine if a particle falls into a triangle, the distance must be smaller or equal to the radius of the particle. Dist here is the distance and length/2 is the radius
    # count = 0
    # # Define opacity/transparency for all triangles (last number as 100 instead of 255)
    # your_mesh.visual.face_colors = [100, 100, 100, 255]  # change the color of the triangle
    # IDs = np.zeros(max(id))
    # for i in range(len(dist)):
    #     if dist[i] < length[i] / 2.0:
    #         count = count + 1
    #         IDs[id[i] - 1] += 1
    # rangR = max(IDs)
    # for i in range(len(IDs)):
    #
    #     if IDs[i] != 0:
    #         your_mesh.visual.face_colors[i][:2] = np.array(cmap((IDs[i] / rangR))[:2]) * 255
    #
    # def condition(x): return x != 0
    # deposID = np.where(condition(IDs))
    # deposID = np.unique([deposID,filter_ID])

        # ttarea = trimesh.triangles.area(your_mesh.triangles, sum=True)
        # area = trimesh.triangles.area(your_mesh.triangles[UniqID], sum=True)
        if export:
            your_mesh.export(fileName)
        if plot:
            your_mesh.export(fileName)
            importmesh = trimesh.load(fileName)
            plot_mesh(importmesh)

        # Percent = area / ttarea * 100
        # print('Converage Percentage: ' + str(area / ttarea * 100) + '%')
        # print('Converage Area: ' + str(area / 100) + 'cm^2')
        # print(Percent)
        print(vol)
    else:
        # Percent = 0
        vol = 0
    return (vol)

def filter_Spray_Refined(mesh_file, particle_data,threshold):
    '''
    This function determines whether a particle lands on a mesh file. Depending on how many particles the mesh has been
    landed on, the color of the mesh will change which forms a heat map. You can directly render the modified mesh from
    the function and export the gif.

    :param mesh_file: Mesh file txt file from COMSOL
    :param particle_file: Particle position txt file from COMSOL
    :param excel_index: Which parameter you choose to export from COMSOL
    :param var: Variable of interest you want to analyze eg: velocity, cone angle, just for export filename purpose
    :param plot: Whether you want to render the colored mesh
    :param export: Whether you want to exported the colored mesh
    :return: Return Percentage of coverage

    Notes:
    - Must first install trimesh. To do this from anaconda command prompt:
        > conda install -c conda-forge trimesh
    - might want to install networkx too:
        > conda install -c conda-forge networkx
    - Must install numpy-stl:
        conda install -c conda-forge numpy-stl
    - Must install pandas too:
        conda install -c conda-forge pandas
    - Must install pyrender
        pip install pyrender
    - Must install pyglet
        conda install -c conda-forge pyglet

    - more info on trimesh here:
        https://github.com/mikedh/trimesh
    - Has the functionality to evaluate distance between point and mesh
        more info here: https://github.com/mikedh/trimesh/blob/master/examples/nearest.ipynb
    - Info on pyrender:
        https://readthedocs.org/projects/pyrender/downloads/pdf/stable/

    -other note: choose python library
        https://softwarerecs.stackexchange.com/questions/64807/which-mesh-processing-library-for-python-to-chose

    - Specs for OBJ file: https://en.wikipedia.org/wiki/Wavefront_.obj_file

    @author: ashao, jmbouteiller
    '''
    clean_data = particle_data
    your_mesh = trimesh.load(mesh_file)
    P0 = clean_data[['x', 'y', 'z']].to_numpy()
    length = (clean_data['size'].to_numpy()*1000).astype(int) + 1
    ##Calculate whether a particle falls into the triangle
    [closid, dist, id] = trimesh.proximity.closest_point(your_mesh,
                                                         P0)
    # closid is the closest point on triangles for each point, dist is the distance to mesh, id is the index of triangle containing closest point.
    # To determine if a particle falls into a triangle, the distance must be smaller or equal to the radius of the particle. Dist here is the distance and length/2 is the radius
    count = 0
    IDs = np.zeros(len(id)).astype(int)
    for i in range(len(id)-1):
        if dist[i] < length[i] / 2.0:
            IDs[count] = id[i]
            count = count + 1
    UniqID = np.unique(IDs)  # UniquID shows each individual triangle id which particles land on
    for i in UniqID:
        triangle = your_mesh.triangles[i]
        triangle = triangle.reshape((1, 3, 3))
        triarea = trimesh.triangles.area(triangle)
        ttpart = np.where(id == i)[0]
        ttarea = 0
        for t in ttpart:
            ttarea = ttarea + (length[t] / 2) ** 2 * np.pi
        if ttarea / triarea < threshold:
            UniqID[UniqID == i] = 0
    UniqID = UniqID[UniqID != 0]
    return (UniqID)

def total_area_Data_Refined(mesh_file, particle_data, var, plot, export, threshold):
    '''
    #Calculate original viral file coverage area
    This function determines whether a particle lands on a mesh file. Depending on how many particles the mesh has been
    landed on, the color of the mesh will change which forms a heat map. You can directly render the modified mesh from
    the function and export the gif.

    :param mesh_file: Mesh file txt file from COMSOL
    :param particle_file: Particle position txt file from COMSOL
    :param excel_index: Which parameter you choose to export from COMSOL
    :param var: Variable of interest you want to analyze eg: velocity, cone angle, just for export filename purpose
    :param plot: Whether you want to render the colored mesh
    :param export: Whether you want to exported the colored mesh
    :return: Return Percentage of coverage

    Notes:
    - Must first install trimesh. To do this from anaconda command prompt:
        > conda install -c conda-forge trimesh
    - might want to install networkx too:
        > conda install -c conda-forge networkx
    - Must install numpy-stl:
        conda install -c conda-forge numpy-stl
    - Must install pandas too:
        conda install -c conda-forge pandas
    - Must install pyrender
        pip install pyrender
    - Must install pyglet
        conda install -c conda-forge pyglet

    - more info on trimesh here:
        https://github.com/mikedh/trimesh
    - Has the functionality to evaluate distance between point and mesh
        more info here: https://github.com/mikedh/trimesh/blob/master/examples/nearest.ipynb
    - Info on pyrender:
        https://readthedocs.org/projects/pyrender/downloads/pdf/stable/

    -other note: choose python library
        https://softwarerecs.stackexchange.com/questions/64807/which-mesh-processing-library-for-python-to-chose

    - Specs for OBJ file: https://en.wikipedia.org/wiki/Wavefront_.obj_file

    @author: ashao, jmbouteiller
    '''
    cmap = plt.get_cmap('gist_heat')
    clean_data = particle_data
    if clean_data.shape[0]:
        your_mesh = trimesh.load(mesh_file)
        P0 = clean_data[['x', 'y', 'z']].to_numpy()
        # Retrieve number of particles
        length = clean_data['size'].to_numpy()
        fileName = '' + str(date.today().isoformat()) + '_deposition_profile_for_' + mesh_file
        ##Calculate whether a particle falls into the triangle
        [closid, dist, id] = trimesh.proximity.closest_point(your_mesh,
                                                         P0)  # closid is the closest point on triangles for each point, dist is the distance to mesh, id is the index of triangle containing closest point.
        count = 0
        IDs = np.zeros(len(id))
        for i in range(len(id) - 1):
            if dist[i] < length[i] / 2.0:
                IDs[count] = id[i]
                count = count + 1
        UniqID = np.unique(id)  # UniquID shows each individual triangle id which particles land on
        for i in UniqID:
            triangle = your_mesh.triangles[i]
            triangle = triangle.reshape((1, 3, 3))
            triarea = trimesh.triangles.area(triangle)
            ttpart = np.where(id == i)[0]
            ttarea = 0
            for t in ttpart:
                ttarea = ttarea + (length[t] / 2) ** 2 * np.pi
            if ttarea / triarea < threshold:
                UniqID[UniqID == i] = 0
        UniqID = UniqID[UniqID != 0]
        # To determine if a particle falls into a triangle, the distance must be smaller or equal to the radius of the particle. Dist here is the distance and length/2 is the radius
        count = 0
        # Define opacity/transparency for all triangles (last number as 100 instead of 255)
        your_mesh.visual.face_colors = [100, 100, 100, 255]  # change the color of the triangle
        IDs = np.zeros(max(id))
        for i in range(len(dist)):
            if dist[i] < length[i] / 2.0:
                count = count + 1
                IDs[id[i] - 1] += 1
        rangR = max(IDs)
        for i in range(len(IDs)):
            if IDs[i] != 0:
                your_mesh.visual.face_colors[i][:2] = np.array(cmap((IDs[i] / rangR))[:2]) * 255
        ttarea = trimesh.triangles.area(your_mesh.triangles, sum=True)
        area = trimesh.triangles.area(your_mesh.triangles[UniqID], sum=True)
        if export:
            your_mesh.export(fileName)
        if plot:
            your_mesh.export(fileName)
            importmesh = trimesh.load(fileName)
            plot_mesh(importmesh)

        Percent = area / ttarea * 100
        print('Converage Percentage: ' + str(area / ttarea * 100) + '%')
        print('Converage Area: ' + str(area / 100) + 'cm^2')
        print(Percent)
    else:
        Percent = 0
    return (Percent)

def total_area_filterData_Refined(mesh_file, filter_ID,particle_data, plot, export, threshold):
    '''
    This function determines whether a particle lands on a mesh file. Depending on how many particles the mesh has been
    landed on, the color of the mesh will change which forms a heat map. You can directly render the modified mesh from
    the function and export the gif.

    :param mesh_file: Mesh file txt file from COMSOL
    :param particle_file: Particle position txt file from COMSOL
    :param excel_index: Which parameter you choose to export from COMSOL
    :param var: Variable of interest you want to analyze eg: velocity, cone angle, just for export filename purpose
    :param plot: Whether you want to render the colored mesh
    :param export: Whether you want to exported the colored mesh
    :return: Return Percentage of coverage

    Notes:
    - Must first install trimesh. To do this from anaconda command prompt:
        > conda install -c conda-forge trimesh
    - might want to install networkx too:
        > conda install -c conda-forge networkx
    - Must install numpy-stl:
        conda install -c conda-forge numpy-stl
    - Must install pandas too:
        conda install -c conda-forge pandas
    - Must install pyrender
        pip install pyrender
    - Must install pyglet
        conda install -c conda-forge pyglet

    - more info on trimesh here:
        https://github.com/mikedh/trimesh
    - Has the functionality to evaluate distance between point and mesh
        more info here: https://github.com/mikedh/trimesh/blob/master/examples/nearest.ipynb
    - Info on pyrender:
        https://readthedocs.org/projects/pyrender/downloads/pdf/stable/

    -other note: choose python library
        https://softwarerecs.stackexchange.com/questions/64807/which-mesh-processing-library-for-python-to-chose

    - Specs for OBJ file: https://en.wikipedia.org/wiki/Wavefront_.obj_file

    @author: ashao, jmbouteiller
    '''
    cmap = plt.get_cmap('gist_heat')
    clean_data = particle_data
    if clean_data.shape[0]:
        your_mesh = trimesh.load(mesh_file)
        P0 = clean_data[['x', 'y', 'z']].to_numpy()
        # Retrieve number of particles
        length = clean_data['size'].to_numpy()
        fileName = '' + str(date.today().isoformat()) + '_deposition_profile_for_' + mesh_file
        ##Calculate whether a particle falls into the triangle
        [closid, dist, id] = trimesh.proximity.closest_point(your_mesh,
                                                         P0)  # closid is the closest point on triangles for each point, dist is the distance to mesh, id is the index of triangle containing closest point.
        count = 0
        IDs = np.zeros(len(id))
        for i in range(len(id) - 1):
            if dist[i] < length[i] / 2.0:
                IDs[count] = id[i]
                count = count + 1
        UniqID = np.unique(id)
        for i in UniqID:
            triangle = your_mesh.triangles[i]
            triangle = triangle.reshape((1, 3, 3))
            triarea = trimesh.triangles.area(triangle)
            ttpart = np.where(id == i)[0]
            ttarea = 0
            for t in ttpart:
                ttarea = ttarea + (length[t] / 2) ** 2 * np.pi
            if ttarea / triarea < threshold:
                UniqID[UniqID == i] = 0
        UniqID = UniqID[UniqID != 0]
        for tepid in filter_ID:
            UniqID = UniqID[UniqID != tepid]
        UniqID = UniqID[UniqID != 0]# UniquID shows each individual triangle id which particles land on
    # To determine if a particle falls into a triangle, the distance must be smaller or equal to the radius of the particle. Dist here is the distance and length/2 is the radius
    # count = 0
    # # Define opacity/transparency for all triangles (last number as 100 instead of 255)
    # your_mesh.visual.face_colors = [100, 100, 100, 255]  # change the color of the triangle
    # IDs = np.zeros(max(id))
    # for i in range(len(dist)):
    #     if dist[i] < length[i] / 2.0:
    #         count = count + 1
    #         IDs[id[i] - 1] += 1
    # rangR = max(IDs)
    # for i in range(len(IDs)):
    #
    #     if IDs[i] != 0:
    #         your_mesh.visual.face_colors[i][:2] = np.array(cmap((IDs[i] / rangR))[:2]) * 255
    #
    # def condition(x): return x != 0
    # deposID = np.where(condition(IDs))
    # deposID = np.unique([deposID,filter_ID])

        ttarea = trimesh.triangles.area(your_mesh.triangles, sum=True)
        area = trimesh.triangles.area(your_mesh.triangles[UniqID], sum=True)
        if export:
            your_mesh.export(fileName)
        if plot:
            your_mesh.export(fileName)
            importmesh = trimesh.load(fileName)
            plot_mesh(importmesh)

        Percent = area / ttarea * 100
        print('Converage Percentage: ' + str(area / ttarea * 100) + '%')
        print('Converage Area: ' + str(area / 100) + 'cm^2')
        print(Percent)
    else:
        Percent = 0
    return (Percent)

def filter_Spray_RefinedV2(particle_data,viral_data,viralindex):
    '''
    #Calculate spray filtered viral file coverage area
    This function determines whether a particle lands on a mesh file. Depending on how many particles the mesh has been
    landed on, the color of the mesh will change which forms a heat map. You can directly render the modified mesh from
    the function and export the gif.

    :param mesh_file: Mesh file txt file from COMSOL
    :param particle_file: Particle position txt file from COMSOL
    :param excel_index: Which parameter you choose to export from COMSOL
    :param var: Variable of interest you want to analyze eg: velocity, cone angle, just for export filename purpose
    :param plot: Whether you want to render the colored mesh
    :param export: Whether you want to exported the colored mesh
    :return: Return Percentage of coverage

    Notes:
    - Must first install trimesh. To do this from anaconda command prompt:
        > conda install -c conda-forge trimesh
    - might want to install networkx too:
        > conda install -c conda-forge networkx
    - Must install numpy-stl:
        conda install -c conda-forge numpy-stl
    - Must install pandas too:
        conda install -c conda-forge pandas
    - Must install pyrender
        pip install pyrender
    - Must install pyglet
        conda install -c conda-forge pyglet

    - more info on trimesh here:
        https://github.com/mikedh/trimesh
    - Has the functionality to evaluate distance between point and mesh
        more info here: https://github.com/mikedh/trimesh/blob/master/examples/nearest.ipynb
    - Info on pyrender:
        https://readthedocs.org/projects/pyrender/downloads/pdf/stable/

    -other note: choose python library
        https://softwarerecs.stackexchange.com/questions/64807/which-mesh-processing-library-for-python-to-chose

    - Specs for OBJ file: https://en.wikipedia.org/wiki/Wavefront_.obj_file

    @author: ashao, jmbouteiller
    '''
    clean_data = particle_data #Spray particle position and size
    viral_data = viral_data
    Ps = clean_data
    Pv = viral_data
    Psarray = Ps.to_numpy()
    Pvarray = Pv.to_numpy()
    length = clean_data['size'].to_numpy()
    Vnewind = np.linspace(0, len(Pv)-1, len(Pv)).astype(int)
    Snewind = np.linspace(0, len(Ps)-1, len(Ps)).astype(int)
    Pv = Pv.reindex(Vnewind)
    Ps = Ps.reindex(Snewind)
    intrue = np.ones(len(Pv))
    # def dist(p1, p2):
    #     dist = np.sqrt(np.sum((p1 - p2) ** 2, axis=0))
    #     return dist
    for i in range(len(Ps)):
        Ptep = 0
        Ptep = Pv[np.abs(Pv['x'] - Ps['x'][i]) < length[i]*3]
        Ptep = Ptep[np.abs(Ptep['y'] - Ps['y'][i]) < length[i]*3]
        Ptep = Ptep[np.abs(Ptep['z'] - Ps['z'][i]) < length[i]*3].to_numpy()
        if Ptep.any:
            for t in range(len(Ptep)):
                index = Pv[Pv['x'] == Ptep[t][0]].index[0]
                intrue[index] = 0
    Pvarray = Pvarray[np.where(intrue == 1)]
    Pvarray = pd.DataFrame(data=Pvarray, columns = viralindex)
    return Pvarray


def filter_Spray_RefinedV3(particle_data, viral_data,viralindex):
    Ps = particle_data.reset_index(drop=True)
    Pv = viral_data.reset_index(drop=True)

    Pvarray = Pv.to_numpy()
    length = particle_data['size'].to_numpy()
    intrue = np.ones(len(Pv))
    bound_ratio = 3
    for i in range(len(Ps)):
        Ptep = Pv[np.abs(Pv['x'] - Ps['x'][i]) < length[i] * bound_ratio]
        Ptep = Ptep[np.abs(Ptep['y'] - Ps['y'][i]) < length[i] * bound_ratio]
        Ptep = Ptep[np.abs(Ptep['z'] - Ps['z'][i]) < length[i] * bound_ratio]

        if Ptep.any:
            ps_idx = Ptep.index.tolist()
            intrue[ps_idx] = 0

    Pvarray = Pvarray[np.where(intrue == 1)]
    Pvarray = pd.DataFrame(data=Pvarray, columns=viralindex)
    return Pvarray