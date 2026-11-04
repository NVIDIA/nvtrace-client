# Third-Party Notices

Dependency implementations are supplied by the aggregate build. Preserve the notices below when redistributing source or binaries containing these dependencies.
The complete applicable license texts are reproduced in
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md); preserve that file with
these notices when distributing the corresponding code.

- **flatbuffers**: [upstream source](https://github.com/google/flatbuffers/tree/v23.5.26); see the upstream copyright and license terms in [license](https://github.com/google/flatbuffers/blob/v23.5.26/LICENSE).
- **protobuf**: [upstream source](https://github.com/protocolbuffers/protobuf/tree/v21.12); see the upstream copyright and license terms in [license](https://github.com/protocolbuffers/protobuf/blob/v21.12/LICENSE).

- **DuckDB 1.5.5**: optional table-export dependency, including the configured
  `core_functions` and `parquet` extensions. See [the SDK notices](https://github.com/NVIDIA/nsight-perf-sdk-for-ai/blob/main/THIRD_PARTY_NOTICES.md)
  for provenance. Retain DuckDB's license and the applicable notices for its
  bundled dependencies and selected extensions when distributing binaries that
  incorporate them.

The modified Perfetto schema is distributed by the aggregate under Apache-2.0; see its [notices](https://github.com/NVIDIA/nsight-perf-sdk-for-ai/blob/main/THIRD_PARTY_NOTICES.md). NvPerf Core and the proprietary service retain their independently supplied terms.

PerfWorks runtime dependencies and license terms are listed in
[the runtime notices](https://github.com/NVIDIA/nsight-perf-daemon/blob/main/THIRD_PARTY_NOTICES.md#perfworks-runtime).
Redistribute those notices along with any redistribution of the runtime.

## Apache notices for DuckDB-bundled components

When distributing code or binaries incorporating these DuckDB-bundled
components, preserve the following upstream notices. These components are
licensed under Apache-2.0; see [LICENSE](LICENSE).

```text
Apache Arrow ADBC
Copyright 2022 The Apache Software Foundation

Apache Arrow nanoarrow
Copyright 2023 The Apache Software Foundation

Apache Parquet Format
Copyright 2014 The Apache Software Foundation

Apache Thrift
Copyright (C) 2006 - 2019, The Apache Software Foundation

This product includes software developed at
The Apache Software Foundation (http://www.apache.org/).
```

## Repository policy documents

`CODE_OF_CONDUCT.md` is based on Contributor Covenant version 1.4 and retains its
upstream attribution and license link. `CONTRIBUTING.md` reproduces Developer
Certificate of Origin version 1.1, copyright 2004 and 2006 The Linux Foundation
and its contributors, under the permission stated in that text.
