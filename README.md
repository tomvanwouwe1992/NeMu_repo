# NeMu_repo
Toolbox to replace the computation of OpenSim muscle lengths and moment arms by a Neural Network
The Neural Network Muscle (NeMu) is useful in optimal control applications to have a differentiable function that gives muscle lengths and moment arms from the joint positions of a model.
The NeMu is also conditioned on the scaling factor of the OpenSim model making the scaling of the OpenSim model an additional potential optimization variable.

- NeMuGeometry_Approximation.py:
      - will first generate 2000 randomly scaled OpenSim models. 
      - Then it will evaluate muscle lenghts and moment arms of these models in many different joint positions and save these results. (ground truth dataset)
      - Finally we train neural networks for each muscle to get a differentiable function from [scaling vector, joint position] --> [muscle length, moment arm]
- conversionNeMu.py: optional function to convert tensorflow neural networks to NumPy implementations
- NeMyGeometrySubfunctions.py: auxilary functions that allow scaling opensim models, reading out information of opensim models, define neural networks, define machine learning algorithm
