# Attribution and third-party notices

The root `LICENSE` contains SpaceFlow's Apache-2.0 license. It does not replace
licenses or notices attached to upstream source, checkpoints, or dependencies.

| Component | Source and notice |
| --- | --- |
| GuideFlow3D | [Upstream project](https://github.com/GradientSpaces/GuideFlow3D); SpaceFlow builds on its generation and similarity-guidance pipeline. |
| TRELLIS | [Upstream](https://github.com/microsoft/TRELLIS); retained [MIT license](third_party/TRELLIS/LICENSE). |
| PartField | [Upstream](https://github.com/nv-tlabs/PartField); retained [NVIDIA license](third_party/PartField/LICENSE). |
| Native extensions | Sources/revisions are recorded in `requirements/native-revisions.json`; installed dependencies retain their own notices. |
| React, Three.js, and editor dependencies | Versions and package integrity records are in `sq_ui/app/package-lock.json`; dependencies retain their own licenses. |

## PartField use limitation

PartField's license limits use to **non-commercial research and educational
purposes** (with the exception specified for NVIDIA and its affiliates).
Keep its license and notices with any redistribution of that component, and
review the [original terms](third_party/PartField/LICENSE) for the complete rules.
The same license constraints apply when using the PartField checkpoint.

Model revision records and checkpoint hashes identify the downloaded artifacts;
they do not grant additional rights. Download models from their recorded upstream
repositories and retain the associated notices. Generated research media and
experiment outputs are separate artifacts from the thin source release.
