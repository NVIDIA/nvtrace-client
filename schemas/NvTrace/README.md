# NvTrace FlatBuffers Schemas

This directory contains the FlatBuffers schema sources used by NvTrace client configuration, routing, system-information, and codec-event interfaces.

## Generated headers

The corresponding `*_generated.h` files under `include/NvTrace/Target` are generated artifacts.

- Do not edit generated headers directly.
- Make schema changes in the applicable `.fbs` file and regenerate every affected header.
- Keep generated headers and the schemas from which they were produced in the same change.

The shipped generated headers were produced by FlatBuffers `flatc` 23.5.26. They compile against the FlatBuffers 23.5.26 runtime headers; using another runtime version triggers the generated version check.

## Regenerating headers

The SDK includes the schemas and generated headers, but not `flatc`. Use the `flatc` executable from the same FlatBuffers version as the runtime headers in the consuming build.

From the SDK root:

```shell
FLATC=/path/to/matching/flatc
SCHEMA_DIR=nvtrace-client/schemas/NvTrace
TARGET_DIR=nvtrace-client/include/NvTrace/Target

for schema in "${SCHEMA_DIR}"/*.fbs; do
    case "$(basename "${schema}")" in
        NvTraceRouterObserverInterface.fbs)
            output_dir="${TARGET_DIR}/Router"
            ;;
        NvTraceCodecEvents.fbs)
            output_dir="${TARGET_DIR}/Consumer"
            ;;
        *)
            output_dir="${TARGET_DIR}"
            ;;
    esac

    "${FLATC}" --no-warnings --cpp \
        -I "${SCHEMA_DIR}" \
        -o "${output_dir}" \
        "${schema}"
done
```

Regenerate every schema so no header retains the shipped generator version. The following command prints any header that still contains the FlatBuffers 23.5.26 version guard; it should produce no output after regeneration:

```shell
grep -rlzP \
    'FLATBUFFERS_VERSION_MAJOR == 23 &&\s+FLATBUFFERS_VERSION_MINOR == 5 &&\s+FLATBUFFERS_VERSION_REVISION == 26' \
    --include='*_generated.h' \
    nvtrace-client/include/NvTrace/Target |
    tr '\0' '\n'
```

Keep each generated header in its existing output directory because generated headers include sibling files by filename. A newer `flatc` alone is insufficient: the generated headers and runtime headers must use the same version, and a changed generated C++ API can require corresponding client-source changes. No general supported FlatBuffers version range has been established.

## Compatibility

FlatBuffers schemas used for service-consumer runtime communication are evolved to preserve the bidirectional compatibility guarantee across SDK releases. Existing field identifiers, enum values, union alternatives, and field meanings remain stable. Compatible additions use new identifiers and defaults so an older endpoint can ignore data it does not understand and a newer endpoint can accept data that omits newer fields.

Do not change existing field identifiers, enum values, union alternatives, or field meanings without treating the change as a serialized-format compatibility change. Adding or removing schema elements does not by itself establish compatibility with another SDK release.