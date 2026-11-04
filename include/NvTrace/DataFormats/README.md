# NvTrace Data Formats

This directory defines the C data structures exchanged through NvTrace provider buffers. These headers are wire and shared-memory format contracts, not ordinary implementation headers.

## Compatibility

Provider-buffer formats preserve these compatibility properties:

- `NvTrace_ProviderBufferVersion` is the fixed prefix used to identify the provider-buffer major version. Its layout does not change.
- Within a major format version, existing serialized field meanings and enum numeric values remain stable.
- Additive record changes use `*_MINOR_VERSION_*` field boundaries so readers can consume the fields present in the producing version and ignore unavailable newer fields.
- The current provider-buffer format declares major version `2`, and current readers retain explicit support for the legacy version-1 formats described by the `_v1.h` headers.

These properties support the bidirectional service-consumer guarantee across SDK releases. They do not imply that an old consumer understands a feature or record type introduced later. A new incompatible layout requires a new major version and explicit reader support; an unknown major version must not be interpreted as compatible.

## Modification guidance

Do not make mechanical or cosmetic changes to these structures.

- Do not reorder, remove, rename, resize, or repurpose existing serialized fields.
- Do not change existing enum numeric values.
- Do not change a structure explicitly marked as fixed or immutable.
- Add fields only with the required minor-version boundary and matching reader or observer handling.
- Introduce a new major version for an incompatible layout and add explicit reader or observer handling. Retain old definitions for versions that remain supported.

Changes to these headers require review as data-format changes because an apparently source-compatible edit can silently corrupt data exchanged with another process or release.
