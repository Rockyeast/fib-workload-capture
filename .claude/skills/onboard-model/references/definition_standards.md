# Definition standards

Store reviewed JSON at `definitions/<op_type>/<name>.json`. Reuse the existing
`flashinfer_bench.data.Definition` schema and dataset validator; do not duplicate them.

- Name axes from actual inputs, checking ranks, dtype, strides/layout, and head ratios.
- Keep inputs and outputs distinct; include updated residual/state outputs when applicable.
- Preserve source/capture metadata in the reviewed definition and run evidence. Publication
  copies omit internal `sglang_*` tags.
- Use `status:unverified` until numerical tests establish correctness. Source review and
  successful schema validation do not establish numerical equivalence.
- Record runtime versions and exact requests. Capture freezes reviewed definitions by digest;
  changing a definition requires a new capture, not relabeling old data.
