# Collecte — Sèvre Nantaise

Archivage automatique des données nécessaires à la prévision de crue sur la
Sèvre Nantaise. Toutes les sources exploitées ne conservent qu'une **fenêtre
glissante** : ce qui n'est pas capté est perdu définitivement. C'est la seule
raison d'être de ce dépôt.

## Ce qui est collecté

| Source | Contenu | Fenêtre à la source |
|---|---|---|
| Hub'Eau hydrométrie | hauteurs (H) et débits (Q) de 4 stations | ~30 jours |
| Hub'Eau piézométrie | niveau de nappe temps réel | ~100 jours |
| Météo-France DPRadar | lame d'eau radar moyennée par bassin | **5 minutes** |

### Stations

| Code | Station | BV | Grandeurs |
|---|---|---|---|
| M703243010 | Sèvre Nantaise à **Saint-Laurent-sur-Sèvre** | 576 km² | H |
| M702241010 | Sèvre Nantaise à Saint-Mesmin [La Branle] | 359 km² | H, Q |
| M704401010 | Ouin à Mauléon [La Voie Moulins] | 61 km² | H, Q |
| M711241020 | Sèvre Nantaise à Tiffauges | 814 km² | H, Q |

Saint-Laurent est la station d'intérêt, mais elle **ne publie aucun débit,
nulle part** — ni en temps réel, ni en historique journalier. C'est une station
limnimétrique seule. En revanche 73 % de son bassin est jaugé en amont
(Saint-Mesmin 359 km² + Ouin 61 km²), et Tiffauges l'encadre 18 km en aval :
la prévision peut s'y faire par propagation, directement en hauteur, sans
courbe de tarage.

## Arborescence

```
donnees/
  M703243010/H/2026-09.csv          instant_utc, hauteur_cm
  M702241010/Q/2026-09.csv          instant_utc, debit_m3s
  nappes/05092X0009_P/2026-09.csv   instant_utc, niveau_ngf
  radar/M703243010/2026-09.csv      instant_utc, duree_min, lame_mm, ...
```

Des CSV mensuels plutôt qu'une base binaire : git versionne alors des ajouts de
lignes, et l'historique reste lisible. Tout est idempotent — relancer une
collecte ne crée aucun doublon.

## Utilisation

```bash
python collecte.py --depot donnees --radar   # collecte complète
python collecte.py --rapport                 # état de l'archive, sans réseau
python collecte.py --verifier                # contrôle les codes configurés
python collecte.py --importer donnees        # reconstruit historique.db
python radar.py --boucle 5                   # capture radar en continu
```

Dépendances : `requests`, `numpy`, `h5py`.

## La clé Météo-France

Lue dans la variable d'environnement `METEOFRANCE_API_KEY` (secret du dépôt côté
GitHub Actions), ou à défaut dans `~/.config/floodcast/env` en local. Sans elle,
la collecte radar est simplement ignorée, le reste fonctionne.

La clé est un JWT qui **contient la liste de ses propres abonnements**. Après
s'être abonné à une nouvelle API sur le portail, il faut donc **régénérer le
jeton** : l'ancien continuera de renvoyer `403`.

## Le problème de cadence du radar

L'API `DPRadar` ne sert que le dernier pas de 5 minutes. Le workflow `radar.yml`
demande donc une exécution toutes les 5 minutes — mais **GitHub ne l'honore pas**.

Mesure du 08/09/2026, dépôt public, minutes illimitées : sur 74 minutes,
**5 exécutions au lieu de 15**, à 12:35, 12:50, 13:00, 13:15 et 13:30. Le
planificateur étrangle les crons fréquents à environ une exécution par quart
d'heure. **Couverture réelle : 33 %.** Inexploitable pour reconstituer un cumul.

La parade n'est pas d'insister sur la cadence mais de changer de produit :
**`DonneesPubliquesPaquetRadar` rend le dernier quart d'heure par appel**, soit
exactement l'intervalle réel observé. La couverture passerait à près de 100 %,
avec du recouvrement pour absorber les retards. C'est donc l'abonnement qui rend
ce workflow utile, pas la fréquence demandée.

À défaut, faire tourner `radar.py --boucle 5` sur une machine allumée en
permanence reste la solution sûre.

### launchd (macOS)

`~/Library/LaunchAgents/local.collecte-radar.plist` :

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>local.collecte-radar</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/python3</string>
    <string>/CHEMIN/VERS/Collecte/radar.py</string>
    <string>--depot</string><string>/CHEMIN/VERS/Collecte/donnees</string>
  </array>
  <key>StartInterval</key><integer>300</integer>
  <key>StandardOutPath</key><string>/tmp/collecte-radar.log</string>
  <key>StandardErrorPath</key><string>/tmp/collecte-radar.log</string>
</dict></plist>
```

```bash
launchctl load ~/Library/LaunchAgents/local.collecte-radar.plist
```

## Sources et licence

Ce dépôt **rediffuse** des données publiques. Elles restent la propriété de leurs
producteurs et sont republiées ici sous leurs licences respectives.

| Donnée | Producteur | Licence |
|---|---|---|
| Hauteurs, débits (Hub'Eau hydrométrie) | SCHAPI / OFB — Eaufrance | Licence Ouverte 2.0 |
| Niveaux de nappe (Hub'Eau piézométrie / ADES) | BRGM / OFB — Eaufrance | Licence Ouverte 2.0 |
| Lame d'eau radar (API DPRadar) | **Météo-France** | Licence Ouverte 2.0 |
| Emprises de bassin (`bassins.json`, dérivé du RGE ALTI) | **IGN** | Licence Ouverte 2.0 |

La Licence Ouverte autorise la réutilisation et la rediffusion, y compris
commerciale, à condition de mentionner la source et la date de mise à jour.
Les fichiers de `donnees/` sont des données dérivées : rééchantillonnées pour la
piézométrie, moyennées sur le bassin versant pour la lame d'eau radar. Elles ne
sauraient être présentées comme des données brutes de leurs producteurs.

Le code de ce dépôt est libre d'usage.

## Détails techniques

**Doublons Hub'Eau.** Interrogé par code site, Hub'Eau renvoie chaque mesure de
débit deux fois : une ligne portant le code station, une ligne agrégée où
`code_station` vaut `null`. Le script filtre sur la station configurée.

**Lame d'eau radar.** Le produit est une mosaïque ODIM_H5 de 3 472 × 3 472 pixels
de 500 m (2 Mo), en projection stéréographique polaire. `radar.py` la reprojette
sans dépendance cartographique — la formule est calée sur le coin supérieur
gauche déclaré dans le fichier, ce qui la rend exacte au pixel près — puis n'en
garde que la moyenne sur le bassin, soit une cinquantaine d'octets par pas de
temps. Le masque de bassin s'auto-calibre pour que sa surface corresponde à la
surface officielle du bassin versant (écart constaté : 0,1 %).

La distinction `undetect` / `nodata` est respectée : le premier signifie que le
radar a regardé et n'a rien vu (un vrai zéro), le second qu'il n'y a pas eu de
mesure. Les confondre fabriquerait de la sécheresse artificielle.

**Emprises des bassins.** `bassins.json` contient les mailles retenues pour
chaque bassin, dérivées du MNT RGE ALTI de l'IGN.
