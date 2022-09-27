import opensim
import numpy as np
import os.path
import pickle
import multiprocessing
import NeMuGeometrySubfunctions
from NeMuGeometrySubfunctions import get_muscle_and_coordinate_information

if __name__ == '__main__':

    #####################################################################
    # FLOW CONTROL
    #####################################################################
    generate_randomly_scaled_models = 1
    generate_ground_truth_data_all_muscles = 1
    activation_function = 'relu'
    model_name = 'Hamner_modified.osim'
    number_of_randomly_scaled_models = 2000
    root_path = os.path.dirname(os.path.abspath('NeMuGeometry_Approximation.py'))
    model_path = root_path + '/Models/'
    save_path = root_path + '/Models/NeMu/' + activation_function + '/'

    # list of bodies to be scaled - bodies with identical scaling factors should be grouped in a sublist
    bodies_scaling_list = ["torso",
                           "pelvis",
                           "femur_l",
                           "tibia_l",
                           ["talus_l", "calcn_l", "toes_l"]]

    # Import OpenSim
    os.add_dll_directory(r"C:/OpenSim 4.3/bin")
    opensim.Logger.setLevelString('error') # Do not print errors to console

    # load OpenSim model
    model_ = os.path.join(model_path, model_name)
    model_opensim = opensim.Model(model_)

    # extract information on the included muscles and coordinates from the opensim model (just look at left side of the model)
    included_muscles, indices_of_included_coordinates, coordinate_names = get_muscle_and_coordinate_information(
        model_opensim, bodies_scaling_list)
    coordinate_names = coordinate_names[:10]
    indices_of_included_coordinates = indices_of_included_coordinates[:10]
    ######################################################################
    # --- PART 1: Generate scaled models over which we will randomize
    ######################################################################
    if generate_randomly_scaled_models == 1:
        # INPUT
        # - model name on which the approximation for moment arms and muscle tendon lengths will be based
        # - name of default scale tool .xml setup file
        # - number of randomly scaled models we want to consider
        # - properties of how scaling factors are distributed for the randomly scaled model
        default_scale_tool_xml_name = model_path + '/scaleTool_Default.xml'
        correlation_scaling_factors = 0.8
        standard_deviation_scaling_factors = 0.08

        # OUTPUT
        # - generate & store scaling vectors, generate & store scaled versions of the original model
        # number_of_bodies = model_opensim.getNumBodies()
        number_of_scaling_factors = 3*len(bodies_scaling_list)
        mean_scaling_factor = np.ones(number_of_scaling_factors)
        variance_scaling_factors = standard_deviation_scaling_factors**2
        covariance_scaling_factors = variance_scaling_factors*np.identity(number_of_scaling_factors)
        indices_off_diagonal = np.where(~np.eye(covariance_scaling_factors.shape[0],dtype=bool))
        covariance_scaling_factors[indices_off_diagonal] = correlation_scaling_factors*np.sqrt(variance_scaling_factors)**2


        scale_vectors = np.random.multivariate_normal(mean_scaling_factor, covariance_scaling_factors, size=number_of_randomly_scaled_models)
        # minimum = 0.9 * np.ones((np.shape(scale_vectors)))
        # maximum = 1.1 * np.ones((np.shape(scale_vectors)))
        # scale_vectors = np.maximum(minimum, scale_vectors)
        # scale_vectors = np.minimum(maximum, scale_vectors)
        iterable = []
        for i in range(0, number_of_randomly_scaled_models):
            iterable.append((i, default_scale_tool_xml_name, scale_vectors[i, :], root_path, save_path, model_name))
            # generateScaledModels(i, default_scale_tool_xml_name, scale_vectors[i, :], root_path, save_path, model_name)

        # NeMuGeometrySubfunctions.generateScaledModels(1, default_scale_tool_xml_name, scale_vectors[1,:], root_path,
        #                      save_path, model_name)
        # Parallel generation of randomly scaled models
        pool = multiprocessing.Pool(processes=16)
        pool.starmap(NeMuGeometrySubfunctions.generateScaledModels, iterable)
        pool.close()
        pool.join()
        # Save scale vectors to file
        file_pi = open(save_path + 'ScaledModels/scale_vectors', 'wb')
        pickle.dump(scale_vectors, file_pi)
        file_pi.close()
    else:
        print('Skipped generating randomly scaled models')
        with open(save_path + 'ScaledModels/scale_vectors', 'rb') as f:
            scale_vectors = pickle.load(f)

    ######################################################################
    # --- PART 2: Randomly sample moment arms and muscle tendon length for muscles included in the model
    ######################################################################
    if generate_ground_truth_data_all_muscles == 1:
        # --- PART 2.a: Provide range of motion for different coordinates

        max_angle_hip_flexion = 70 * np.pi / 180
        min_angle_hip_flexion = -30 * np.pi / 180
        ROM_hip_flexion = max_angle_hip_flexion - min_angle_hip_flexion

        max_angle_hip_adduction = 20 * np.pi / 180
        min_angle_hip_adduction = -20 * np.pi / 180
        ROM_hip_adduction = max_angle_hip_adduction - min_angle_hip_adduction

        max_angle_hip_rotation = 35 * np.pi / 180
        min_angle_hip_rotation = -20 * np.pi / 180
        ROM_hip_rotation = max_angle_hip_rotation - min_angle_hip_rotation

        max_angle_knee = 10 * np.pi / 180
        min_angle_knee = -120 * np.pi / 180
        ROM_knee = max_angle_knee - min_angle_knee

        max_angle_ankle = 50 * np.pi / 180
        min_angle_ankle = -50 * np.pi / 180
        ROM_ankle = max_angle_ankle - min_angle_ankle

        max_angle_subtalar = 35 * np.pi / 180
        min_angle_subtalar = -35 * np.pi / 180
        ROM_subtalar = max_angle_subtalar - min_angle_subtalar

        max_angle_mtp = 60 * np.pi / 180
        min_angle_mtp = -10 * np.pi / 180
        ROM_mtp = max_angle_mtp - min_angle_mtp

        max_angle_lumbar_extension = 45 * np.pi / 180
        min_angle_lumbar_extension = -45 * np.pi / 180
        ROM_lumbar_extension = max_angle_lumbar_extension - min_angle_lumbar_extension

        max_angle_lumbar_bending = 45 * np.pi / 180
        min_angle_lumbar_bending = -45 * np.pi / 180
        ROM_lumbar_bending = max_angle_lumbar_bending - min_angle_lumbar_bending

        max_angle_lumbar_rotation = 45 * np.pi / 180
        min_angle_lumbar_rotation = -45 * np.pi / 180
        ROM_lumbar_rotation = max_angle_lumbar_rotation - min_angle_lumbar_rotation

        ROM_all = np.array(
            [ROM_hip_flexion, ROM_hip_adduction, ROM_hip_rotation, ROM_knee, ROM_ankle, ROM_subtalar, ROM_mtp, ROM_lumbar_extension, ROM_lumbar_bending, ROM_lumbar_rotation])
        min_angle_all = np.array(
            [min_angle_hip_flexion, min_angle_hip_adduction, min_angle_hip_rotation, min_angle_knee, min_angle_ankle,
             min_angle_subtalar, min_angle_mtp, min_angle_lumbar_extension, min_angle_lumbar_bending, min_angle_lumbar_rotation])
        max_angle_all = np.array(
            [max_angle_hip_flexion, max_angle_hip_adduction, max_angle_hip_rotation, max_angle_knee, max_angle_ankle,
             max_angle_subtalar, max_angle_mtp, max_angle_lumbar_extension, max_angle_lumbar_bending, max_angle_lumbar_rotation])

        if not len(ROM_all) == len(coordinate_names) or not len(min_angle_all) == len(coordinate_names) or not len(max_angle_all) == len(coordinate_names):
            raise Exception("Please provide ROM/min/max for all included dofs and in the following order: " + str(coordinate_names[:]))

        # --- PART 2.b: Sample throughout state space for different muscles (in parallel)

        iterable = []
        for i in range(len(included_muscles)):
            iterable.append((included_muscles[i], indices_of_included_coordinates, ROM_all, min_angle_all, max_angle_all, number_of_randomly_scaled_models, scale_vectors, save_path, root_path))

        # Parallel generation of randomly scaled models
        pool = multiprocessing.Pool(processes=16)
        pool.starmap(NeMuGeometrySubfunctions.sampleMomentArmsMuscleTendonLengths, iterable)
        pool.close()
        pool.join()


    ######################################################################
    # --- PART 3: Train NeMu's based on generated data - tensorflow models
    ######################################################################

    os.add_dll_directory(r"C:\OpenSim 4.3\bin")
    muscle_list = os.listdir(save_path)

    iterable = []
    for i in range(len(included_muscles)):
        iterable.append((included_muscles[i].name, save_path, activation_function))

    pool = multiprocessing.Pool(processes=16)
    pool.starmap(NeMuGeometrySubfunctions.trainNeMu_Geometry, iterable)
    pool.close()
    pool.join()




