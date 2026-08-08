"""
12b_sample_pool.py
==================
Estrae un campione stratificato e bilanciato dal prompt pool, per contenere il
tempo di attacco su RunPod entro il budget disponibile.

PERCHÉ CAMPIONARE. Il pool completo (~28.000 prompt) richiede 8-12h di GPU con
batching. Un probe lineare non ha bisogno di quel volume: la letteratura di
probing lavora abitualmente con qualche migliaio di esempi per classe, e
oltre una certa soglia la curva accuratezza-per-layer non cambia in modo
apprezzabile. Ridurre il campione è il taglio meno costoso scientificamente —
molto meno dannoso che ridurre il budget di token (che reintrodurrebbe il
confound del Cap. 1) o il numero di layer estratti (che eliminerebbe la curva).

CAMPIONAMENTO A LIVELLO DI GRUPPO. L'unità campionata è source_prompt_id, non
la riga. Un prompt tradotto in tre lingue forma un gruppo di quattro righe:
o entrano tutte o nessuna. Campionare righe singole spezzerebbe la struttura
parallela, che è ciò che permette di attribuire le differenze cross-linguali
al modello invece che alla difficoltà della richiesta.

BILANCIAMENTO. Le lingue sottili (francese, spagnolo) entrano per intero; le
lingue abbondanti vengono limitate da un tetto. Il risultato è più equilibrato
del pool di partenza, dove l'inglese pesa il 43%.

Output: prompt_pool_sampled.parquet
"""

import os
import json

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = "/Users/tommasomilanino/Developer/THESIS"
DATA_DIR  = os.path.join(REPO_ROOT, "phase3_internals", "dataset_building")

INPUT_FILE  = os.path.join(DATA_DIR, "prompt_pool_with_metrics.parquet")
OUTPUT_FILE = os.path.join(DATA_DIR, "prompt_pool_sampled.parquet")
STATS_FILE  = os.path.join(DATA_DIR, "sampling_stats.json")

# ── Configurazione ────────────────────────────────────────────────────────────

# Tetto per lingua. Le lingue sotto il tetto entrano per intero.
LANGUAGE_CAPS = {
    "english": 6000,
    "russian": 3000,
    "arabic":  3000,
    "german":  3000,
    "french":  2500,
    "spanish": 2500,
}

# Fonti prese integralmente prima della stratificazione, fino al tetto
# indicato. Sono i benchmark pubblici consolidati: senza di loro il campione
# resterebbe dominato dal corpus generato, e un probe addestrato quasi solo su
# prompt sintetici rischia di apprendere la firma del generatore invece del
# fenomeno. Sono anche le fonti su cui poggia il Capitolo 1, quindi la
# continuità fra i capitoli richiede che siano rappresentate.
#
# Il problema che questo risolve: quasi tutti i benchmark hanno category
# assente, quindi confluiscono nello strato "other". Nella rotazione per
# categoria quello strato riceve la stessa quota di ogni altro, e sei fonti
# se la dividono — con l'effetto di ridurre AdvBench a una decina di righe.
PRIORITY_SOURCES = {
    "advbench":               None,   # None = prendi tutto
    "xstest":                 None,
    "harmbench":              None,
    "do_not_answer":          None,
    "wildjailbreak":          1200,
    "multilingual_safety_en": 1200,
    "aya_redteaming":         None,   # unica fonte con prompt nativi umani
}

RANDOM_SEED = 42


# ── Campionamento ─────────────────────────────────────────────────────────────

def build_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Un record per source_prompt_id, con la lingua principale e la categoria
    usate come strati. La lingua principale è quella della prima riga del
    gruppo: per i gruppi monolingua è l'unica, per quelli paralleli è
    l'inglese di origine.
    """
    groups = df.groupby("source_prompt_id").agg(
        n_rows=("prompt_id", "size"),
        languages=("language", lambda s: tuple(sorted(set(s)))),
        category=("category", "first"),
        source=("source", "first"),
    ).reset_index()

    # Lingua di stratificazione: l'inglese se presente nel gruppo (è
    # l'originale), altrimenti la prima disponibile.
    groups["strat_language"] = groups["languages"].apply(
        lambda langs: "english" if "english" in langs else langs[0])

    return groups


def select_priority_groups(groups: pd.DataFrame, df: pd.DataFrame,
                           rng) -> set:
    """
    Gruppi delle fonti prioritarie, presi prima della stratificazione. Il
    conteggio è sui gruppi, non sulle righe: prendere un gruppo significa
    prendere tutte le sue versioni linguistiche.
    """
    selected = set()

    for source, cap in PRIORITY_SOURCES.items():
        candidates = groups[groups["source"] == source]
        if len(candidates) == 0:
            print(f"  {source:<24} assente dal pool")
            continue

        if cap is None or len(candidates) <= cap:
            chosen = candidates
        else:
            # Stratifica per categoria anche dentro la fonte, così un
            # sottocampione di WildJailbreak non finisce concentrato su un
            # tipo di attacco.
            per_cat = max(1, cap // candidates["category"].nunique())
            chosen = (candidates.groupby("category", group_keys=False)
                      .apply(lambda g: g.sample(min(len(g), per_cat),
                                                random_state=RANDOM_SEED)))
            if len(chosen) < cap:
                extra = candidates[~candidates["source_prompt_id"]
                                   .isin(chosen["source_prompt_id"])]
                n_extra = min(cap - len(chosen), len(extra))
                if n_extra > 0:
                    chosen = pd.concat([
                        chosen, extra.sample(n_extra, random_state=RANDOM_SEED)])

        selected.update(chosen["source_prompt_id"])
        print(f"  {source:<24} {len(chosen):>5} gruppi su {len(candidates)}")

    return selected


def sample_groups(groups: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """
    Due fasi. Prima le fonti prioritarie, prese integralmente o fino al loro
    tetto. Poi il riempimento stratificato per lingua e categoria, che si
    ferma quando il tetto della lingua è raggiunto — contando anche le righe
    già entrate con la fase prioritaria.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    lang_counts = (df.groupby(["source_prompt_id", "language"])
                     .size().unstack(fill_value=0))

    print("\nFase 1 — fonti prioritarie:")
    selected_group_ids = select_priority_groups(groups, df, rng)

    # Righe già acquisite per lingua dalla fase prioritaria
    rows_per_language = {lang: 0 for lang in LANGUAGE_CAPS}
    for gid in selected_group_ids:
        if gid in lang_counts.index:
            for lang in lang_counts.columns:
                n = int(lang_counts.loc[gid, lang])
                if n and lang in rows_per_language:
                    rows_per_language[lang] += n

    print(f"\n  righe acquisite: "
          f"{ {k: v for k, v in rows_per_language.items() if v} }")

    print("\nFase 2 — riempimento stratificato:")
    availability = df["language"].value_counts()
    languages_by_scarcity = sorted(
        LANGUAGE_CAPS.keys(), key=lambda l: availability.get(l, 0))

    for language in languages_by_scarcity:
        cap = LANGUAGE_CAPS[language]
        if rows_per_language[language] >= cap:
            print(f"  {language:<10} già al tetto "
                  f"({rows_per_language[language]}/{cap})")
            continue

        candidates = groups[
            (~groups["source_prompt_id"].isin(selected_group_ids)) &
            (groups["languages"].apply(lambda ls: language in ls))
        ]
        if len(candidates) == 0:
            continue

        by_category = {
            cat: grp.sample(frac=1, random_state=RANDOM_SEED).to_dict("records")
            for cat, grp in candidates.groupby("category")
        }
        category_cycle = list(by_category.keys())

        start = rows_per_language[language]
        idx = 0
        while rows_per_language[language] < cap and any(by_category.values()):
            cat = category_cycle[idx % len(category_cycle)]
            idx += 1
            if not by_category[cat]:
                continue

            group = by_category[cat].pop()
            gid = group["source_prompt_id"]
            selected_group_ids.add(gid)

            if gid in lang_counts.index:
                for lang in lang_counts.columns:
                    n = int(lang_counts.loc[gid, lang])
                    if n and lang in rows_per_language:
                        rows_per_language[lang] += n

        print(f"  {language:<10} {start} → {rows_per_language[language]} "
              f"(tetto {cap})")

    return df[df["source_prompt_id"].isin(selected_group_ids)].reset_index(drop=True)


# ── Report ────────────────────────────────────────────────────────────────────

def report(original: pd.DataFrame, sample: pd.DataFrame) -> dict:
    print("\n" + "=" * 66)
    print("CAMPIONE STRATIFICATO")
    print("=" * 66)
    print(f"Pool originale: {len(original)} righe")
    print(f"Campione:       {len(sample)} righe "
          f"({len(sample)/len(original)*100:.1f}%)")

    print(f"\nPer lingua:")
    orig_lang = original["language"].value_counts()
    samp_lang = sample["language"].value_counts()
    for lang in sorted(set(orig_lang.index) | set(samp_lang.index)):
        o, s = orig_lang.get(lang, 0), samp_lang.get(lang, 0)
        print(f"  {lang:<10} {o:>6} → {s:>6}  ({s/o*100 if o else 0:.0f}%)")

    print(f"\nPer categoria:")
    orig_cat = original["category"].value_counts()
    samp_cat = sample["category"].value_counts()
    for cat in sorted(set(orig_cat.index) | set(samp_cat.index)):
        o, s = orig_cat.get(cat, 0), samp_cat.get(cat, 0)
        print(f"  {cat:<22} {o:>6} → {s:>6}")

    print(f"\nPer fonte (originale → campione):")
    orig_src = original["source"].value_counts()
    samp_src = sample["source"].value_counts()
    for src in orig_src.index:
        o, s = orig_src.get(src, 0), samp_src.get(src, 0)
        flag = "  ←priorità" if src in PRIORITY_SOURCES else ""
        print(f"  {src:<28} {o:>6} → {s:>6}  ({s/o*100 if o else 0:>3.0f}%){flag}")

    n_groups = sample["source_prompt_id"].nunique()
    print(f"\nGruppi (source_prompt_id): {n_groups}")
    print(f"Righe per gruppo: {len(sample)/n_groups:.2f} in media")

    # Stima dei tempi sul pod
    print(f"\nStima attacco su A6000 con batching:")
    for rate in [1.0, 1.5, 2.0, 3.0]:
        print(f"  a {rate:.1f} prompt/s → {len(sample)/rate/3600:.1f}h")

    return {
        "original_rows": len(original),
        "sampled_rows": len(sample),
        "by_language": samp_lang.to_dict(),
        "by_category": samp_cat.to_dict(),
        "by_source": samp_src.to_dict(),
        "n_groups": int(n_groups),
        "language_caps": LANGUAGE_CAPS,
        "priority_sources": PRIORITY_SOURCES,
        "seed": RANDOM_SEED,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"{INPUT_FILE} non trovato.")

    df = pd.read_parquet(INPUT_FILE)
    print(f"Pool caricato: {len(df)} righe")
    print(df["language"].value_counts().to_string())

    groups = build_groups(df)
    print(f"\nGruppi identificati: {len(groups)}")
    print(f"  monolingua: {(groups['n_rows'] == 1).sum()}")
    print(f"  paralleli:  {(groups['n_rows'] > 1).sum()}")

    sample = sample_groups(groups, df)
    stats = report(df, sample)

    sample.to_parquet(OUTPUT_FILE, engine="pyarrow")
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"\nSalvato in {OUTPUT_FILE}")
    print(f"\nQuesto è il file da caricare sul pod. Il pool completo resta "
          f"disponibile per le analisi descrittive dei Cap. 1-2, dove non c'è "
          f"vincolo di tempo GPU.")