"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

package_name = "DeepSaki"
skip_files = ["__main__", "__version__","__init__"]

for path in sorted(Path(package_name).rglob("__init__.py")):
    module_path = path.relative_to(".").with_suffix("")
    doc_path = path.relative_to(".").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    parts = parts[:-1]
    doc_path = doc_path.with_name("index.md")
    full_doc_path = full_doc_path.with_name("index.md")

    nav[parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        for file in module_path.parent.glob("*.py"):
            if file.stem in skip_files:
                #print(f"skipped: {file}")
                continue
            #print(f"chosen: {file}")
            ident = ".".join([*parts, file.stem])
            fd.write(f"::: {ident}\n")
            #fd.write("    options:\n")
            #fd.write("        ignore_init_summary: true\n")


        mkdocs_gen_files.set_edit_path(full_doc_path, Path("../") / path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
