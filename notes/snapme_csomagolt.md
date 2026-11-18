# SNAPMe: csomagolt termékek a tesztelt mintában (2026-11-18)

A `benchmark.py` SNAPMe-szűrője `query="has_file"`: a csomagolt termékek csomagolás-elejéről
készült fotói is a mintában maradtak (a konfig megjegyzésében szereplő `and not packaged`
kizárta volna őket). Utólag, a mintán nem változtatva, megszámoltam őket.

Eredmény: a döntési mérés 199 tesztelt SNAPMe-étele közül 26 csomagolt termék.

Reprodukálás (az adathalmazok mappájában, mert a `packaged` jelző a SNAPMe-adatból jön):

    cd ~/bcont-boxes/opencode/datasets
    venv/bin/python - <<'EOF'
    import importlib.util, sys; from pathlib import Path
    sys.path.insert(0, "../thesis"); import prereg_tests as P
    spec = importlib.util.spec_from_file_location("l", "snapme/loader.py"); l = importlib.util.module_from_spec(spec); spec.loader.exec_module(l)
    items, images = l.load_snapme("snapme/snapme_cs_db")
    d = P.load_run("final_snapme_lite_macros", Path("../thesis/results"))[0]
    pk = images["packaged"].reindex(d.index.astype(str))
    print(len(d), "tesztelt;", int(pk.sum()), "csomagolt; nem talált:", int(pk.isna().sum()))
    EOF

Kimenet (2026-11-18): `199 tesztelt; 26 csomagolt; nem talált: 0`
