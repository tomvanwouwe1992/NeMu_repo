import os.path
import numpy as np
import pickle
from tensorflow import keras
import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense

class MuscleInfo:
    def __init__(self, name, index, actuated_coordinate_names, actuated_coordinate_full_names,
                 actuated_coordinate_indices, actuated_body_names, actuated_body_indices,
                 maximal_isometric_force,
                 optimal_fiber_length, tendon_slack_length, pennation_at_optimal_fiber_length,
                 maximal_fiber_velocity, muscle_width, tendon_stiffness):
        self.name = name
        self.index = index
        self.actuated_coordinate_names = actuated_coordinate_names
        self.actuated_coordinate_full_names = actuated_coordinate_full_names
        self.actuated_coordinate_indices = actuated_coordinate_indices
        self.actuated_body_names = actuated_body_names
        self.actuated_body_indices = actuated_body_indices
        self.maximal_isometric_force = maximal_isometric_force
        self.optimal_fiber_length = optimal_fiber_length
        self.tendon_slack_length = tendon_slack_length
        self.pennation_at_optimal_fiber_length = pennation_at_optimal_fiber_length
        self.maximal_fiber_velocity = maximal_fiber_velocity
        self.muscle_width = muscle_width
        self.tendon_stiffness = tendon_stiffness
        self.q_samples = []
        self.moment_arm_samples = []
        self.muscle_tendon_length_samples = []
        self.scaling_vector_samples = []
        self.NeMu_LMT_dM = []
        self.min_muscle_tendon_length_generic = []
        self.max_muscle_tendon_length_generic = []


def generateScaledModels(i, default_scale_tool_xml_name, scale_vector, root_path,
                         save_path, model_name):
    import opensim
    opensim.Logger.setLevelString('error')
    ScaleTool_OS = opensim.ScaleTool(default_scale_tool_xml_name)
    ModelMaker_OS = ScaleTool_OS.getGenericModelMaker()
    ModelMaker_OS.setModelFileName(model_name)

    # Get model scaler and scaleset - then adapt the weights in the scalesets
    ModelScaler_OS = ScaleTool_OS.getModelScaler()
    ScaleSet_OS = ModelScaler_OS.getScaleSet()

    # torso
    Scale_OS = ScaleSet_OS.get(11)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[0])
    newScaleFactors.set(1, scale_vector[1])
    newScaleFactors.set(2, scale_vector[2])
    Scale_OS.setScaleFactors(newScaleFactors)

    # pelvis
    Scale_OS = ScaleSet_OS.get(0)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[3])
    newScaleFactors.set(1, scale_vector[4])
    newScaleFactors.set(2, scale_vector[5])
    Scale_OS.setScaleFactors(newScaleFactors)

    # femur_l
    Scale_OS = ScaleSet_OS.get(6)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[6])
    newScaleFactors.set(1, scale_vector[7])
    newScaleFactors.set(2, scale_vector[8])
    Scale_OS.setScaleFactors(newScaleFactors)

    # tibia_l
    Scale_OS = ScaleSet_OS.get(7)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[9])
    newScaleFactors.set(1, scale_vector[10])
    newScaleFactors.set(2, scale_vector[11])
    Scale_OS.setScaleFactors(newScaleFactors)

    # talus_l - calcaneus_l - toes (same scaling values)
    Scale_OS = ScaleSet_OS.get(8)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[12])
    newScaleFactors.set(1, scale_vector[13])
    newScaleFactors.set(2, scale_vector[14])
    Scale_OS.setScaleFactors(newScaleFactors)

    Scale_OS = ScaleSet_OS.get(9)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[12])
    newScaleFactors.set(1, scale_vector[13])
    newScaleFactors.set(2, scale_vector[14])
    Scale_OS.setScaleFactors(newScaleFactors)

    Scale_OS = ScaleSet_OS.get(10)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[12])
    newScaleFactors.set(1, scale_vector[13])
    newScaleFactors.set(2, scale_vector[14])
    Scale_OS.setScaleFactors(newScaleFactors)

    ScaleTool_OS.setName('scaledModel_' + str(i))
    scaled_model_path = os.path.join(save_path, 'ScaledModels/scaledModel_' + str(i) + '.osim')
    ModelScaler_OS.setOutputModelFileName(scaled_model_path)
    print('Scaling model #' + str(i))
    # Run the scale tool
    ScaleTool_OS.run()
    scaled_model_opensim = opensim.Model(scaled_model_path)
    knee_joint_r = scaled_model_opensim.getJointSet().get('knee_r')
    knee_joint_r_dc = opensim.CustomJoint_safeDownCast(knee_joint_r)
    knee_spatial_transform = knee_joint_r_dc.getSpatialTransform()
    # tranform axis translation 1
    TA_translation1 = knee_spatial_transform.getTransformAxis(3)
    TA_MultiplierFunction = opensim.MultiplierFunction_safeDownCast(TA_translation1.get_function())
    TA_MultiplierFunction.setScale(scale_vector[6])
    knee_spatial_transform.updTransformAxis(3)
    # tranform axis translation 2
    TA_translation2 = knee_spatial_transform.getTransformAxis(4)
    TA_MultiplierFunction = opensim.MultiplierFunction_safeDownCast(TA_translation2.get_function())
    TA_MultiplierFunction.setScale(scale_vector[7])
    knee_spatial_transform.updTransformAxis(4)
    # tranform axis translation 3
    TA_translation3 = knee_spatial_transform.getTransformAxis(5)
    TA_MultiplierFunction = opensim.MultiplierFunction_safeDownCast(TA_translation3.get_function())
    TA_MultiplierFunction.setScale(scale_vector[8])
    knee_spatial_transform.updTransformAxis(5)
    scaled_model_opensim.printToXML(scaled_model_path)



def generateScaledModels_full(i, default_scale_tool_xml_name, scale_vector, root_path,
                         save_path, model_name):
    import opensim
    opensim.Logger.setLevelString('error')
    ScaleTool_OS = opensim.ScaleTool(default_scale_tool_xml_name)
    ModelMaker_OS = ScaleTool_OS.getGenericModelMaker()
    ModelMaker_OS.setModelFileName(model_name)

    # Get model scaler and scaleset - then adapt the weights in the scalesets
    ModelScaler_OS = ScaleTool_OS.getModelScaler()
    ScaleSet_OS = ModelScaler_OS.getScaleSet()

    # torso
    Scale_OS = ScaleSet_OS.get(11)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[0])
    newScaleFactors.set(1, scale_vector[1])
    newScaleFactors.set(2, scale_vector[2])
    Scale_OS.setScaleFactors(newScaleFactors)

    # pelvis
    Scale_OS = ScaleSet_OS.get(0)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[3])
    newScaleFactors.set(1, scale_vector[4])
    newScaleFactors.set(2, scale_vector[5])
    Scale_OS.setScaleFactors(newScaleFactors)

    # femur_l
    Scale_OS = ScaleSet_OS.get(6)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[6])
    newScaleFactors.set(1, scale_vector[7])
    newScaleFactors.set(2, scale_vector[8])
    Scale_OS.setScaleFactors(newScaleFactors)

    # tibia_l
    Scale_OS = ScaleSet_OS.get(7)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[9])
    newScaleFactors.set(1, scale_vector[10])
    newScaleFactors.set(2, scale_vector[11])
    Scale_OS.setScaleFactors(newScaleFactors)

    # talus_l - calcaneus_l - toes (same scaling values)
    Scale_OS = ScaleSet_OS.get(8)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[12])
    newScaleFactors.set(1, scale_vector[13])
    newScaleFactors.set(2, scale_vector[14])
    Scale_OS.setScaleFactors(newScaleFactors)

    Scale_OS = ScaleSet_OS.get(9)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[12])
    newScaleFactors.set(1, scale_vector[13])
    newScaleFactors.set(2, scale_vector[14])
    Scale_OS.setScaleFactors(newScaleFactors)

    Scale_OS = ScaleSet_OS.get(10)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[12])
    newScaleFactors.set(1, scale_vector[13])
    newScaleFactors.set(2, scale_vector[14])
    Scale_OS.setScaleFactors(newScaleFactors)

    # femur_r
    Scale_OS = ScaleSet_OS.get(1)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[15])
    newScaleFactors.set(1, scale_vector[16])
    newScaleFactors.set(2, scale_vector[17])
    Scale_OS.setScaleFactors(newScaleFactors)

    # tibia_r
    Scale_OS = ScaleSet_OS.get(2)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[18])
    newScaleFactors.set(1, scale_vector[19])
    newScaleFactors.set(2, scale_vector[20])
    Scale_OS.setScaleFactors(newScaleFactors)

    # talus_r - calcaneus_r - toes (same scaling values)
    Scale_OS = ScaleSet_OS.get(3)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[21])
    newScaleFactors.set(1, scale_vector[22])
    newScaleFactors.set(2, scale_vector[23])
    Scale_OS.setScaleFactors(newScaleFactors)

    Scale_OS = ScaleSet_OS.get(4)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[21])
    newScaleFactors.set(1, scale_vector[22])
    newScaleFactors.set(2, scale_vector[23])
    Scale_OS.setScaleFactors(newScaleFactors)

    Scale_OS = ScaleSet_OS.get(5)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[21])
    newScaleFactors.set(1, scale_vector[22])
    newScaleFactors.set(2, scale_vector[23])
    Scale_OS.setScaleFactors(newScaleFactors)

    # humerus_r
    Scale_OS = ScaleSet_OS.get(12)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[24])
    newScaleFactors.set(1, scale_vector[25])
    newScaleFactors.set(2, scale_vector[26])
    Scale_OS.setScaleFactors(newScaleFactors)

    # radius_r
    Scale_OS = ScaleSet_OS.get(13)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[27])
    newScaleFactors.set(1, scale_vector[28])
    newScaleFactors.set(2, scale_vector[29])
    Scale_OS.setScaleFactors(newScaleFactors)

    # unlna_r
    Scale_OS = ScaleSet_OS.get(14)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[27])
    newScaleFactors.set(1, scale_vector[28])
    newScaleFactors.set(2, scale_vector[29])
    Scale_OS.setScaleFactors(newScaleFactors)

    # hand_r
    Scale_OS = ScaleSet_OS.get(15)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[30])
    newScaleFactors.set(1, scale_vector[31])
    newScaleFactors.set(2, scale_vector[32])
    Scale_OS.setScaleFactors(newScaleFactors)

    # humerus_l
    Scale_OS = ScaleSet_OS.get(16)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[33])
    newScaleFactors.set(1, scale_vector[34])
    newScaleFactors.set(2, scale_vector[35])
    Scale_OS.setScaleFactors(newScaleFactors)

    # radius_l
    Scale_OS = ScaleSet_OS.get(17)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[36])
    newScaleFactors.set(1, scale_vector[37])
    newScaleFactors.set(2, scale_vector[38])
    Scale_OS.setScaleFactors(newScaleFactors)

    # unlna_l
    Scale_OS = ScaleSet_OS.get(18)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[36])
    newScaleFactors.set(1, scale_vector[37])
    newScaleFactors.set(2, scale_vector[38])
    Scale_OS.setScaleFactors(newScaleFactors)

    # hand_l
    Scale_OS = ScaleSet_OS.get(19)
    newScaleFactors = opensim.Vec3(1)
    newScaleFactors.set(0, scale_vector[39])
    newScaleFactors.set(1, scale_vector[40])
    newScaleFactors.set(2, scale_vector[41])
    Scale_OS.setScaleFactors(newScaleFactors)

    ScaleTool_OS.setName('scaledModel_' + str(i))
    scaled_model_path = os.path.join(save_path, 'ScaledModels/scaledModel_' + str(i) + '.osim')
    ModelScaler_OS.setOutputModelFileName(scaled_model_path)
    print('Scaling model #' + str(i))
    # Run the scale tool
    ScaleTool_OS.run()
    scaled_model_opensim = opensim.Model(scaled_model_path)
    knee_joint_r = scaled_model_opensim.getJointSet().get('knee_r')
    knee_joint_r_dc = opensim.CustomJoint_safeDownCast(knee_joint_r)
    knee_spatial_transform = knee_joint_r_dc.getSpatialTransform()
    # tranform axis translation 1
    TA_translation1 = knee_spatial_transform.getTransformAxis(3)
    TA_MultiplierFunction = opensim.MultiplierFunction_safeDownCast(TA_translation1.get_function())
    TA_MultiplierFunction.setScale(scale_vector[6])
    knee_spatial_transform.updTransformAxis(3)
    # tranform axis translation 2
    TA_translation2 = knee_spatial_transform.getTransformAxis(4)
    TA_MultiplierFunction = opensim.MultiplierFunction_safeDownCast(TA_translation2.get_function())
    TA_MultiplierFunction.setScale(scale_vector[7])
    knee_spatial_transform.updTransformAxis(4)
    # tranform axis translation 3
    TA_translation3 = knee_spatial_transform.getTransformAxis(5)
    TA_MultiplierFunction = opensim.MultiplierFunction_safeDownCast(TA_translation3.get_function())
    TA_MultiplierFunction.setScale(scale_vector[8])
    knee_spatial_transform.updTransformAxis(5)
    scaled_model_opensim.printToXML(scaled_model_path)


def sampleMomentArmsMuscleTendonLengths(muscle, indices_of_included_coordinates, ROM_all, min_angle_all, max_angle_all, number_of_used_scaled_models, scale_vectors, save_path, root_path):
        import opensim
        opensim.Logger.setLevelString('error') # Do not print errors to console
        number_of_actuated_coordinates = len(muscle.actuated_coordinate_indices)
        if number_of_actuated_coordinates > 0:

            # We first get to LMT and dM since these only depend on q
            number_of_samples = 1
            for d in range(number_of_actuated_coordinates):
                actuated_coordinate_index_in_model = muscle.actuated_coordinate_indices[d]
                index_d = indices_of_included_coordinates.index(actuated_coordinate_index_in_model)
                # Take a sample every 1deg
                number_of_samples = int(number_of_samples * (ROM_all[index_d] / np.pi * 180))
            number_of_samples = int(number_of_samples / (np.power(7, (number_of_actuated_coordinates - 1))))
            if number_of_actuated_coordinates < 2:
                number_of_samples = number_of_samples * 4
            if number_of_samples < 20000:
                number_of_samples = 20000
            number_of_samples = int(number_of_samples / number_of_used_scaled_models) * number_of_used_scaled_models
            print(number_of_samples)

            q_samples = np.zeros((number_of_samples, number_of_actuated_coordinates))
            moment_arm_samples = np.zeros((number_of_samples, number_of_actuated_coordinates))
            muscle_tendon_length_samples = np.zeros((number_of_samples, 1))
            scaling_vector_samples = np.zeros((number_of_samples, np.shape(scale_vectors)[1]))

            scaled_model_optimal_fiber_length_samples = np.zeros((number_of_samples, 1))
            scaled_model_tendon_slack_length_samples = np.zeros((number_of_samples, 1))
            scaled_model_maximal_fiber_velocity_samples = np.zeros((number_of_samples, 1))
            scaled_model_muscle_width_samples = np.zeros((number_of_samples, 1))
            scaled_model_tendon_stiffness_samples = np.zeros((number_of_samples, 1))

            for d in range(number_of_actuated_coordinates):
                actuated_coordinate_index_in_model = muscle.actuated_coordinate_indices[d]
                index_d = indices_of_included_coordinates.index(actuated_coordinate_index_in_model)
                q_samples[:, d] = ROM_all[index_d] * (np.random.random_sample(number_of_samples) - 0.5) + (
                        min_angle_all[index_d] + max_angle_all[index_d]) / 2

            # We are going to take samples for moment arms and muscle tendon lengths from differently scaled models
            # We generated 2000 randomly scaled models (with some correlation between different scaling variables)
            blocks = int(number_of_samples / number_of_used_scaled_models)
            for j in range(number_of_used_scaled_models):
                q_samples_block = q_samples[j * blocks:(j + 1) * blocks, :]
                model_name = 'ScaledModels/scaledModel_' + str(j) + '.osim'
                model_path = os.path.join(save_path, model_name)
                model_opensim = opensim.Model(model_path)
                muscles = model_opensim.getMuscles()
                opensim_muscle = muscles.get(muscle.name)
                muscle_tendon_length_samples_block, moment_arm_samples_block = get_mtu_length_and_moment_arm(model_opensim,
                                                                                                             muscle,
                                                                                                             q_samples_block)
                moment_arm_samples[j * blocks:(j + 1) * blocks, :] = moment_arm_samples_block
                muscle_tendon_length_samples[j * blocks:(j + 1) * blocks, :] = muscle_tendon_length_samples_block
                scaling_vector_samples[j * blocks:(j + 1) * blocks, :] = np.tile(scale_vectors[j, :], (blocks, 1))
                scaled_model_optimal_fiber_length_samples[j * blocks:(j + 1) * blocks, :] = opensim_muscle.getOptimalFiberLength()
                scaled_model_tendon_slack_length_samples[j * blocks:(j + 1) * blocks, :] = opensim_muscle.getTendonSlackLength()
                scaled_model_maximal_fiber_velocity_samples[j * blocks:(j + 1) * blocks, :] = opensim_muscle.getMaxContractionVelocity()
                scaled_model_muscle_width_samples[j * blocks:(j + 1) * blocks, :] = opensim_muscle.getOptimalFiberLength() * np.sin(
                    opensim_muscle.getPennationAngleAtOptimalFiberLength())
                scaled_model_tendon_stiffness_samples[j * blocks:(j + 1) * blocks, :] = 35

            muscle.q_samples = q_samples
            muscle.moment_arm_samples = moment_arm_samples
            muscle.muscle_tendon_length_samples = muscle_tendon_length_samples
            scaling_vector_indices = np.zeros(3*len(muscle.actuated_body_indices))
            for i in range(len(muscle.actuated_body_indices)):
                scaling_vector_indices[3 * i + 0] = 3 * muscle.actuated_body_indices[i] + 0
                scaling_vector_indices[3 * i + 1] = 3 * muscle.actuated_body_indices[i] + 1
                scaling_vector_indices[3 * i + 2] = 3 * muscle.actuated_body_indices[i] + 2
            muscle.scaling_vector_samples = scaling_vector_samples[:, scaling_vector_indices.astype(int)]

            file_pi = open(save_path + muscle.name, 'wb')
            pickle.dump(muscle, file_pi)
            file_pi.close()
            print(muscle.name + ' data generated with ' + str(number_of_samples) + ' samples, for ' + str(
                number_of_used_scaled_models) + ' different models')

def get_muscle_and_coordinate_information(model_os, bodies_scaling_list):
    import opensim
    state = model_os.initSystem()
    model_os.equilibrateMuscles(state)

    coordinate_set = model_os.getCoordinateSet()
    number_of_coordinates = coordinate_set.getSize()

    coordinate_names = []
    coordinate_indices_in_model = []
    for i in range(0, number_of_coordinates):
        coordinate_name = coordinate_set.get(i).getName()
        if np.char.endswith(coordinate_name, '_l') or not np.char.endswith(coordinate_name, '_r') and not np.char.startswith(coordinate_name, 'pelvis_'):
            coordinate_names.append(coordinate_name)
            coordinate_indices_in_model.append(i)

    muscle_set = model_os.getMuscles()
    number_of_muscles = muscle_set.getSize()

    included_muscles = []
    for m in range(0, number_of_muscles):
        muscle = model_os.getMuscles().get(m)
        muscle_name = muscle.getName()
        if np.char.endswith(muscle_name, '_l'):
            actuated_coordinate_names_for_this_muscle = []
            actuated_coordinate_full_names_for_this_muscle = []
            index_of_actuated_coordinate_for_this_muscle = []
            actuated_body_names_for_this_muscle = []
            for c in range(len(coordinate_indices_in_model)):
                index_of_coordinate_in_model_coordinates = coordinate_indices_in_model[c]
                coordinate_of_interest_os = model_os.getCoordinateSet().get(index_of_coordinate_in_model_coordinates)
                name_joint_of_coordinate_of_interest_os = coordinate_of_interest_os.getJoint().getName()
                moment_arm = muscle.computeMomentArm(state, coordinate_of_interest_os)
                if abs(moment_arm) > 0.0001:
                    actuated_coordinate_names_for_this_muscle.append(coordinate_of_interest_os.getName())
                    full_name_actuated_coordinate = name_joint_of_coordinate_of_interest_os + '/' + coordinate_of_interest_os.getName()
                    actuated_coordinate_full_names_for_this_muscle.append(full_name_actuated_coordinate)
                    index_of_actuated_coordinate_for_this_muscle.append(index_of_coordinate_in_model_coordinates)
                    joint_os = coordinate_of_interest_os.getJoint()
                    child_frame = joint_os.getChildFrame().getName()
                    if child_frame.endswith('_offset'):
                        child_frame = child_frame[:-7]
                    parent_frame = joint_os.getParentFrame().getName()
                    if parent_frame.endswith('_offset'):
                        parent_frame = parent_frame[:-7]
                    if child_frame not in actuated_body_names_for_this_muscle:
                        actuated_body_names_for_this_muscle.append(child_frame)
                    if parent_frame not in actuated_body_names_for_this_muscle:
                        actuated_body_names_for_this_muscle.append(parent_frame)

            actuated_body_indices_for_this_muscle = []
            for i in range(len(actuated_body_names_for_this_muscle)):
                for j in range(len(bodies_scaling_list)):
                    if actuated_body_names_for_this_muscle[i] in bodies_scaling_list[j]:
                        actuated_body_indices_for_this_muscle.append(j)
            actuated_body_indices_for_this_muscle = list(set(actuated_body_indices_for_this_muscle))
            actuated_body_indices_for_this_muscle.sort()

            included_muscle = MuscleInfo(muscle_name, m,
                                         actuated_coordinate_names_for_this_muscle,
                                         actuated_coordinate_full_names_for_this_muscle,
                                         index_of_actuated_coordinate_for_this_muscle,
                                         actuated_body_names_for_this_muscle,
                                         actuated_body_indices_for_this_muscle,
                                         muscle.getMaxIsometricForce(),
                                         muscle.getOptimalFiberLength(),
                                         muscle.getTendonSlackLength(),
                                         muscle.getPennationAngleAtOptimalFiberLength(),
                                         muscle.getMaxContractionVelocity(),
                                         muscle.getOptimalFiberLength() * np.sin(
                                             muscle.getPennationAngleAtOptimalFiberLength()),
                                         35)
            included_muscles.append(included_muscle)

    return included_muscles, coordinate_indices_in_model, coordinate_names

def get_mtu_length_and_moment_arm(model_os, muscle, q):
    import opensim
    number_of_samples = np.shape(q)[0]
    actuated_coordinate_indices = muscle.actuated_coordinate_indices
    trajectory_mtu_length = np.zeros((number_of_samples, 1))
    trajectory_moment_arm = np.zeros((number_of_samples, len(actuated_coordinate_indices)))
    state = model_os.initSystem()
    for i in range(number_of_samples):
        for j in range(len(actuated_coordinate_indices)):
            actuated_coordinate_name = muscle.actuated_coordinate_full_names[j]
            state_name_q = '/jointset/' + actuated_coordinate_name + '/value'
            value_q = q[i, j]
            model_os.setStateVariableValue(state, state_name_q, value_q)
        state = model_os.updWorkingState()
        model_os.realizePosition(state)

        muscle_os = model_os.getMuscles().get(muscle.index)

        trajectory_mtu_length[i] = muscle_os.getLength(state)
        coordinate_set = model_os.getCoordinateSet()
        for k in range(len(actuated_coordinate_indices)):
            coordinate = coordinate_set.get(actuated_coordinate_indices[k])
            trajectory_moment_arm[i, k] = muscle_os.computeMomentArm(state,
                                                                     coordinate)
    return trajectory_mtu_length, trajectory_moment_arm


def trainNeMu_Geometry(muscle_name, save_path, activation_function):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    with open(save_path  + muscle_name, 'rb') as f:
        muscle = pickle.load(f)
    in_full = np.concatenate((muscle.scaling_vector_samples, muscle.q_samples), axis=1)
    out_full = np.concatenate((muscle.moment_arm_samples, muscle.muscle_tendon_length_samples), axis=1)

    permutation = np.random.permutation(np.shape(in_full)[0])
    in_full = in_full[permutation, :]
    out_full = out_full[permutation, :]

    #

    # Normalize predictions (better cost function scaling)
    out_normalization_shift = np.mean((out_full), axis=0)
    out_normalization_range = np.amax(np.abs(out_full), axis=0)# - np.amin(out_full, axis=0)
    weights_array = 1 / out_normalization_range
    weights_dict = {}
    for i in range(len(weights_array)):
        weights_dict[i] = weights_array[i]
    out_full_normalized = out_full #(out_full - out_normalization_shift) / out_normalization_range
    length = np.shape(in_full)[0]
    x_train = in_full[0:int(length * 0.9), :]
    x_test = in_full[int(length * 0.9):, :]

    y_train = out_full_normalized[0:int(length * 0.9), :]
    y_test = out_full_normalized[int(length * 0.9):, :]
    model = keras.models.Sequential()

    if len(muscle.actuated_coordinate_indices) == 1:
        model.add(keras.layers.Dense(8, input_dim=np.shape(in_full)[1], activation=activation_function))
        # model.add(keras.layers.Dense(8, input_dim=np.shape(in_full)[1], activation=activation_function))
        model.add(keras.layers.Dense(np.shape(out_full)[1], activation='linear'))

    elif len(muscle.actuated_coordinate_indices) == 2:
        model.add(keras.layers.Dense(12, input_dim=np.shape(in_full)[1], activation=activation_function))
        # model.add(keras.layers.Dense(8, activation=activation_function))
        model.add(keras.layers.Dense(np.shape(out_full)[1], activation='linear'))

    elif len(muscle.actuated_coordinate_indices) == 3:
        model.add(keras.layers.Dense(16, input_dim=np.shape(in_full)[1], activation=activation_function))
        model.add(keras.layers.Dense(8, activation=activation_function))
        model.add(keras.layers.Dense(np.shape(out_full)[1], activation='linear'))

    elif len(muscle.actuated_coordinate_indices) > 3:
        model.add(keras.layers.Dense(16, input_dim=np.shape(in_full)[1], activation=activation_function))
        # model.add(keras.layers.Dense(16, activation=activation_function))
        model.add(keras.layers.Dense(8, activation=activation_function))

        model.add(keras.layers.Dense(np.shape(out_full)[1], activation='linear'))

    n_epoch = 250
    model.compile(loss='mean_squared_error', optimizer='adam') #  loss_weights=weights_array.tolist()
    history = model.fit(x_train, y_train, epochs=n_epoch, validation_split=0.10, batch_size=64)
    score = model.evaluate(x_test, y_test, batch_size=64)

    print(score)

    model.save(save_path + muscle.name + '_Geometry.h5')
    tf.keras.models.save_model(model, save_path + '/NeMuGeometry_NeMU/' + muscle.name + '_Geometry')
    with open(save_path + muscle.name + '_Geometry_trainingHistory', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
