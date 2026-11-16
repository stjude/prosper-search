# Path placeholders

The released code is the code that was executed on institutional storage.
Absolute paths in it pointed at that storage. Before release, those path
prefixes were replaced mechanically by the placeholders below. No other path
was rewritten, and no analysis logic was changed. Each replacement is counted
per file in [`CODE_MANIFEST.tsv`](CODE_MANIFEST.tsv) (class
`path-literal substitution`).

| Placeholder | Stands for |
|---|---|
| `/PROSPER_WORKSPACE` | The parent directory that held the revision workspace (`prosper_revision/`), the development clone of this package (`prosper-search/`) and the earlier COMET pipeline trees (`comet-*`). Both storage mount points of that directory, and the same directory seen through a workstation mount, map to this one placeholder. |
| `/PROSPER_CORE` | The parent of `/PROSPER_WORKSPACE`, a shared image-analysis area that held other project directories |
| `/PROSPER_DEPT` | The department storage root |
| `/PROSPER_GROUP_DATA/comet` | The pathology group's COMET project directory: whole-slide images, mosaics, embeddings and annotations (controlled access) |
| `/PROSPER_GROUP_DATA` | The pathology group's projects directory |
| `/PROSPER_GROUP` | The pathology group's storage root |
| `/PROSPER_STORAGE` | The institutional storage root, used when nothing more specific applies |
| `/HOME` | The analyst's home directory, which held the Python installation and conda environments, e.g. `/HOME/miniconda3/bin/python3` |
| `user` | The analyst's login name where it appeared on its own, e.g. in cluster user fields or in paths beside `/HOME` |
| `pathologist_annotated_` | The prefix of the region-of-interest annotation directories, whose original name included the annotating pathologist's first name |

Where a path in a comment or docstring started with the storage root and no
more specific prefix applied, it was mapped to `/PROSPER_STORAGE`.
Partial forms of the workspace path in comments and docstrings (for example
`.../Core/<name>`) were mapped to `/PROSPER_WORKSPACE`, and references in
comments to internal working documents of the workspace were replaced by
`[internal record]`; both are logged as `comment scrub`.

## Running the code elsewhere

These placeholders do not exist on another system. To rerun a script, either
create the corresponding directories, for example with symbolic links, or edit
the path constants at the top of the script. Most revision scripts locate the
workspace relative to their own position (`Path(__file__).parents[N]`). The
`revision/` directory takes the place of the original workspace root, and
outputs are written under `$PROSPER_OUT`.

The original literal values are kept with the release records and are not
published.
