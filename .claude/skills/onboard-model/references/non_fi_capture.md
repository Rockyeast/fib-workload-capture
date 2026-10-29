# Capture contracts

A reviewed definition uses exactly one capture point:

- `fi_api:flashinfer.module.callable`: native FlashInfer logger and existing sanitizer.
- `sglang_module:fully.qualified.Class`: SGLang Dumper plus argument/attribute sidecars.
- `sglang_callable:fully.qualified.function`: wrapper in serving worker processes.

Use `sglang_input:x=arg:0`, `sglang_input:x=arg:argument_name`, or
`sglang_input:weight=attr:weight` when definition inputs differ from the runtime signature.
The module adapter selects one representative module instance per class from the inventory;
this is not an all-layers or all-model-operators completeness claim. Multiple definitions can
cover different layouts of that instance. The coverage check takes the union of successful
matches; individual variant rejections are expected, unmatched calls are not.

For paged Attention, verify rank and NHD/HND layout from the actual cache. Capture matching
`plan()` metadata (indices, indptr, last-page lengths, scale) alongside `run()`; do not infer
head counts by assuming every cache tensor has four dimensions.

For GDN, verify whether gates are raw parameters or already computed coefficients, whether
Q/K normalization is inside the callable, and the state layout and mutation contract.
A prefill definition cannot stand in for a decode layout or a recurrent state update.

Use the existing Definition schema and validator. References must expose `run`, match all
outputs, and preserve scalar values and structural indices. See
[definition standards](definition_standards.md) for the publication contract.
