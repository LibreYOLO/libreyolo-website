# build-augmentation

Generates the data behind the augmentation pages:

- `/docs/reference/augmentation-matrix`: knob table, family matrix, defaults by family
- `/docs/train/augmentations`: the example figures

Both scripts import the installed LibreYOLO, so run them in an environment that
has the release being documented, from the website root:

```bash
python -m venv /tmp/aug && /tmp/aug/bin/pip install "libreyolo==1.8.0"
/tmp/aug/bin/python scripts/build-augmentation/collect.py         # src/data/docs/augmentation.json
/tmp/aug/bin/python scripts/build-augmentation/render_figures.py  # public/augmentation/, src/data/docs/augmentation-figures.json
```

`render_figures.py` downloads `LibreYOLO9t.pt` once, to find the person box on
the sample image. Both scripts are deterministic for a given release.

`collect.py` fails rather than writing a claim the release no longer supports:
it checks the declared ranges against the library's own validation, and checks
each entry in `CORRECTIONS` (where the spec says more than the pipeline does)
against the pipeline source. When a check fails, update the declaration it
names, rerun, and review the page prose for the same fact.

After regenerating, bump `last_verified` on both pages and check their prose
against the new tables: the text names defaults and families in a few places.
