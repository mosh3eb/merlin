merlin.algorithms.kernels module
================================

.. automodule:: merlin.algorithms.kernels
   :members:
   :undoc-members:
   :show-inheritance:

.. note::

   When the wrapped :class:`~merlin.algorithms.kernels.FeatureMap` exposes a
   :class:`perceval.Experiment`, fidelity kernels compose the attached
   :class:`perceval.NoiseModel` (photon loss) before applying any detector
   transforms. The resulting kernel values therefore reflect both survival
   probabilities and detector post-processing.
