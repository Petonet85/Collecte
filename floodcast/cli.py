"""Point d'entree : `python -m floodcast.cli --station K055001010`."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import archive
from .forecast import build_context, run
from .web.build import build as build_html

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="floodcast", description="Prevision de crue probabiliste sur station Hub'Eau")
    ap.add_argument("--station", required=False, default="K055001010",
                    help="code station (10 car.) ou site (8 car.) Hub'Eau")
    ap.add_argument("--horizon", type=int, default=72, help="echeance en heures")
    ap.add_argument("--out", default=OUT_DIR, help="repertoire de sortie")
    ap.add_argument("--recalibrer", action="store_true", help="force un nouveau calage")
    ap.add_argument("--iters", type=int, default=1200, help="budget DDS de calage")
    ap.add_argument("--archive", action="store_true", help="affiche l'etat de l'archive locale")
    ap.add_argument("--sans-validation", action="store_true",
                    help="saute le rejeu historique (plus rapide)")
    ap.add_argument("--chercher", metavar="TEXTE",
                    help="cherche une station par nom de riviere, commune ou station")
    ap.add_argument("--departement", metavar="DEP", help="restreint la recherche a un departement")
    ap.add_argument("--proche", metavar="LON,LAT", help="stations dans un rayon autour d'un point")
    ap.add_argument("--rayon", type=float, default=30.0, help="rayon de recherche en km")
    args = ap.parse_args(argv)

    if args.chercher is not None or args.proche or args.departement:
        from .sources.hubeau import rechercher
        lon = lat = None
        if args.proche:
            lon, lat = (float(v) for v in args.proche.split(","))
        df = rechercher(args.chercher or "", args.departement, lon, lat, args.rayon)
        if df.empty:
            print("aucune station trouvee")
            return 1
        cols = ["code_station", "libelle_station", "libelle_cours_eau",
                "libelle_commune", "code_departement"]
        if "dist_km" in df.columns:
            df["dist_km"] = df["dist_km"].round(1)
            cols.append("dist_km")
        print(df[cols].to_string(index=False))
        return 0

    if args.archive:
        df = archive.summary()
        print(df.to_string(index=False) if not df.empty else "archive vide")
        return 0

    t = time.time()
    print(f"FloodCast — station {args.station}", flush=True)
    ctx = build_context(args.station, force=args.recalibrer, iters=args.iters)
    res = run(ctx, horizon_h=args.horizon)
    if not args.sans_validation:
        from .evaluate import validate
        try:
            res["validation"] = validate(ctx)
        except Exception as exc:  # noqa: BLE001 - la validation ne doit pas bloquer la prevision
            print(f"  [validation] indisponible : {exc}")

    os.makedirs(args.out, exist_ok=True)
    code = res["meta"]["code_station"]
    js = os.path.join(args.out, f"forecast_{code}.json")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, default=str)
    html = build_html(res, os.path.join(args.out, f"floodcast_{code}.html"))
    print(f"\n  JSON  {js}\n  HTML  {html}\n  ({time.time() - t:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
