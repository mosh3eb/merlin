MerlinProcessor
===============

MerlinProcessor is a PyTorch-friendly orchestration layer that runs models
containing quantum layers either locally or on Quandela Cloud using one
simple API.

You can pass a full ``nn.Module`` (e.g., ``nn.Sequential``). Merlin walks the
module, runs classical parts locally, and offloads quantum leaves to the cloud
when allowed. Results are always returned as PyTorch tensors.


Contents
--------

- What you are looking at
- Quick start
- Local vs remote execution (policy and introspection)
- Futures (async API)
- Resume existing jobs
- Shots and backend commands
- Batching and limits
- Devices and dtypes
- Errors and timeouts
- Best practices
- API reference


What you are looking at
-----------------------

- A single async surface:

  ``forward_async(module, x, *, shots=None, timeout=None) -> torch.futures.Future``

- A synchronous wrapper:

  ``forward(module, x, *, shots=None, timeout=None) -> torch.Tensor``

- Per-layer execution policy (local vs remote) driven by introspection:

  - Execution leaves are discovered during traversal.
  - For each leaf, Merlin decides whether to offload or to run locally.

- Futures expose convenience controls:

  - ``future.cancel_remote()`` to best-effort cancel the in-flight cloud job.
  - ``future.status()`` to read a minimal status dict.
  - ``future.job_ids`` listing cloud job ids in layer order.

- Results are always PyTorch tensors. No raw RPC payloads leak out.


Quick start
-----------

Create a quantum layer, build a processor, and run a batch.

::

    import torch
    import torch.nn as nn
    from merlin.core.merlin_processor import MerlinProcessor
    from merlin.algorithms import QuantumLayer
    from merlin.builder.circuit_builder import CircuitBuilder
    from merlin.sampling.strategies import OutputMappingStrategy

    # Build a simple QuantumLayer that outputs a raw probability distribution
    b = CircuitBuilder(n_modes=6)
    b.add_rotations(trainable=True, name="theta")
    b.add_angle_encoding(modes=[0, 1], name="px")
    b.add_entangling_layer()

    q = QuantumLayer(
        input_size=2,
        output_size=None,                   # raw distribution (no internal mapping)
        builder=b,
        n_photons=2,
        no_bunching=True,
        output_mapping_strategy=OutputMappingStrategy.NONE,
    ).eval()

    # Bind to a Quandela Cloud backend
    proc = MerlinProcessor.from_platform("sim:slos", token="<YOUR_TOKEN>", timeout=60.0)

    x = torch.rand(4, 2)

    # Synchronous call (blocks until done or timeout)
    y = proc.forward(q, x, shots=2000)      # -> torch.Tensor, shape (4, 15)

    # Asynchronous call (returns a Future)
    fut = proc.forward_async(q, x, shots=2000)
    print(fut.status())    # {'state': ..., 'progress': ..., 'message': ...}
    print(fut.job_ids)     # job ids appear during execution
    y_async = fut.wait()   # -> torch.Tensor


Local vs remote execution (policy and introspection)
----------------------------------------------------

Merlin executes your model **leaf by leaf**. A **leaf** is a module that should
run as one atomic unit (e.g., a QuantumLayer). Leaves are found by depth-first
traversal using this rule:

- If a module has ``merlin_leaf == True``, it is treated as a leaf and Merlin
  **does not** recurse into its children (useful when a leaf wraps internal
  mapping modules like ``nn.Identity``).

For each leaf, Merlin decides whether to offload or run locally:

1) If the leaf implements ``should_offload(remote_processor, shots) -> bool``,
   Merlin calls it and uses the returned boolean.

2) Otherwise, Merlin checks for a cloud-capable quantum leaf and a local override:

   - Offload if the leaf exposes a callable ``export_config()``, **and**
     ``getattr(leaf, "force_simulation", False)`` is ``False``.
   - If either of those is not true, Merlin calls the leaf's own ``forward(...)``
     **locally** (never a no-op passthrough).

QuantumLayer is ready out of the box:

- ``merlin_leaf = True`` so it is treated as a leaf.
- ``export_config()`` provides the circuit and metadata for cloud execution.
- ``force_simulation: bool`` (default False). When True, the layer always runs locally.
- Temporary local mode:

  ::

      q = QuantumLayer(...).eval()
      with q.as_simulation():
          y_local = proc.forward(q, torch.rand(2, q.input_size), shots=2000)
      # after the context, offloading resumes as usual


Futures (async API)
-------------------

``forward_async(...)`` returns a ``torch.futures.Future`` and immediately returns
control to you. The future has three helpers:

- ``cancel_remote()``

  Best-effort cancel of the current cloud job (if any). Waiting on the future
  then raises ``concurrent.futures.CancelledError``.

- ``status() -> dict``

  Lightweight dict containing keys such as ``state``, ``progress``, and
  ``message`` when available. After completion, it returns
  ``{"state": "COMPLETE", "progress": 1.0, "message": None}``.

- ``job_ids: list[str]``

  One id per offloaded leaf, populated in execution order.

Example:

::

    fut = proc.forward_async(q, x, shots=5000, timeout=30.0)
    while not fut.done():
        print(fut.status())
        time.sleep(0.1)
    out = fut.wait()  # -> tensor


Resume existing jobs
--------------------

Attach to an existing cloud job by its id and get results mapped as a tensor.

::

    resumed = proc.resume(
        job_id="<job_id>",
        layer=q,                    # the exact leaf corresponding to the job
        batch_size=4,               # number of iterations submitted
        shots=2000,                 # optional
        device=torch.device("cuda"),
        dtype=torch.float32,
        timeout=60.0,
    )
    y = resumed.wait()              # -> tensor, dtype/device as requested


Shots and backend commands
--------------------------

Backends expose different command types:

- If the backend supports ``"probs"``, Merlin returns **exact probabilities** and
  ignores ``shots``.
- Otherwise Merlin uses sample-based endpoints (``"sample_count"`` or ``"samples"``).
  Pass ``shots`` per call, or use ``shots=None`` to let Merlin choose a sensible default.


Batching and limits
-------------------

Each batch row is submitted as one iteration in a single remote job. Merlin enforces
a per-submission maximum batch size (``max_batch_size``). If your input exceeds the
limit, a ``ValueError`` is raised; split the batch and submit multiple calls.


Devices and dtypes
------------------

- If the input tensor is on CUDA, the output tensor is returned on CUDA.
- The dtype is preserved (e.g., ``float64`` in --> ``float64`` out).
- Any internal CPU staging for transport is handled automatically.


Errors and timeouts
-------------------

- Training mode is not allowed for remote execution. Ensure ``.eval()`` is set
  on models that contain quantum leaves when using MerlinProcessor.
- ``TimeoutError`` is raised by ``forward(...)`` or by waiting on the future if
  the overall timeout is exceeded. Merlin issues a best-effort remote cancel.
- ``concurrent.futures.CancelledError`` is raised when you call
  ``future.cancel_remote()`` and then wait.
- If the remote backend returns an error, Merlin raises ``RuntimeError`` with
  the message reported by the backend.


Best practices
--------------

- Always put quantum leaves in ``.eval()`` mode before remote execution.
- In composite models, detect adapter widths with a single local call to the leaf,
  then wire your classical ``nn.Linear`` adapters accordingly.

  ::

      # discover distribution width from q1 with a quick local forward
      dist1 = q1(torch.rand(2, q1.input_size)).shape[1]
      adapter = nn.Linear(dist1, q2.input_size, bias=False)

- Use the context manager when you want to temporarily keep a leaf local:

  ::

      with q.as_simulation():
          # this call runs locally
          y = proc.forward(q, x, shots=2000)

- If you implement your own execution leaf:
  - set ``merlin_leaf = True``
  - optionally implement ``should_offload(remote_processor, shots) -> bool``
  - implement ``export_config()`` if you want cloud execution


API reference
-------------

Class
~~~~~

``MerlinProcessor(remote_processor, max_batch_size=32, timeout=60.0)``

Construct an orchestrator bound to a ``perceval.runtime.RemoteProcessor``.

Attributes
~~~~~~~~~~

- ``remote_processor``: the bound ``RemoteProcessor``
- ``max_batch_size``: maximum iterations per submission
- ``default_timeout``: default timeout used when none is provided
- ``available_commands``: list of backend command names, when available

Factory
~~~~~~~

``MerlinProcessor.from_platform(platform, token=None, url="https://api.cloud.quandela.com", **kwargs) -> MerlinProcessor``

Convenience constructor for common setups.

Core methods
~~~~~~~~~~~~

- ``forward(module, input, *, shots=None, timeout=None) -> torch.Tensor``

  Synchronous call. Enforces timeout. Returns a tensor on the same device and
  dtype as the input.

- ``forward_async(module, input, *, shots=None, timeout=None) -> torch.futures.Future``

  Asynchronous call. Returns a Future with ``cancel_remote()``, ``status()``,
  and ``job_ids``. Call ``wait()`` or ``value()`` to retrieve the tensor.

- ``resume(job_id, *, layer, batch_size, shots=None, device=None, dtype=None, timeout=None) -> torch.futures.Future``

  Attach to an existing remote job and receive a mapped tensor. Useful for
  dashboards, deferred runs, or retry flows.

Introspection details
~~~~~~~~~~~~~~~~~~~~~

Traversal:

- Depth-first, left-to-right.
- If a module has ``merlin_leaf == True``, it is yielded as an execution leaf and
  traversal does not descend into its children.
- If a module has no children, it is also treated as a leaf (classical ops).

Per-leaf offload decision:

- If the leaf defines ``should_offload(remote_processor, shots) -> bool``,
  that value is used.
- Otherwise, offload when the leaf exposes a callable ``export_config()`` and
  the boolean attribute ``force_simulation`` is False.
- If not offloading, the leaf’s own ``forward(...)`` is executed locally.

This model ensures that a quantum leaf that wraps an internal mapping
(e.g., an ``nn.Identity`` used as a learnable post-mapping placeholder) is not
split apart by traversal, and that setting ``force_simulation = True`` reliably
keeps it local without changing model structure.
