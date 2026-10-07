# Tableau de bord - enquetes d'evaluation du CNFSDP 2025-2026

import math
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

DOSSIER = Path(__file__).parent
LOGO = DOSSIER / "logo.jpg"

st.set_page_config(
    page_title="Enquetes CNFSDP",
    page_icon=Image.open(LOGO) if LOGO.exists() else None,
    layout="wide",
)

# Parametres
SERVEUR = "https://kf.kobotoolbox.org"
TOKEN = st.secrets["TOKEN"]
OBJECTIF = 95
MIN_REPONSES = 5

ROUGE = "#C8102E"
NOIR = "#1A1A1A"
GRIS = "#7A7A7A"
GRIS_CLAIR = "#E3E3E3"
COULEURS = [ROUGE, NOIR, GRIS]

CLASSES = {"TSSP1": 25, "TSSP2": 35, "LSPL1": 34, "LSPL2": 32}

enquetes = {
    "Administration vue par les enseignants": {
        "uid": st.secrets["UID_1"],
        "type": "admin",
        "classes": None,
        "cible": 62,
        "col_global": "I/i_global",
        "col_evolution": "I/i_evolution",
        "col_priorites": "I/i_priorites",
    },
    "Administration vue par les etudiants": {
        "uid": st.secrets["UID_2"],
        "type": "admin",
        "classes": CLASSES,
        "col_global": "J/j_global",
        "col_evolution": "J/j_evolution",
        "col_priorites": "J/j_priorites",
    },
    "Evaluation des enseignants": {
        "uid": st.secrets["UID_3"],
        "type": "evaluation",
        "classes": CLASSES,
    },
}
couleur_de = dict(zip(enquetes, COULEURS))

ECHELLE = ["Excellent", "Tres bien", "Bien", "Moyen", "Mauvais", "Tres mauvais"]
SATISFAITS = ["Excellent", "Tres bien", "Bien"]
COULEURS_ECHELLE = {
    "Excellent": "#1A1A1A", "Tres bien": "#4D4D4D", "Bien": "#8C8C8C",
    "Moyen": "#D0D0D0", "Mauvais": "#E57A8A", "Tres mauvais": "#C8102E",
}

# Codage des notes du questionnaire "Evaluation des enseignants"
NOTES = {"5": "Excellent", "4": "Tres bien", "3": "Bien", "moyen": "Moyen", "2": "Mauvais", "1": "Tres mauvais"}
SCORE = {"Excellent": 5, "Tres bien": 4, "Bien": 3, "Moyen": 2, "Mauvais": 1, "Tres mauvais": 0}
MODELE_EVAL = re.compile(r"^g_([^/]+)/g_t\d+_([^/]+)/t\d+_.+_([a-z]+\d+)$")

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRIS, "axes.titleweight": "bold", "figure.facecolor": "white",
})

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
h1, h2, h3 {color: #1A1A1A;}
[data-testid="stMetricValue"] {font-size: 2rem;}
</style>
""", unsafe_allow_html=True)

# Mot de passe (facultatif)
mdp = st.secrets.get("MOT_DE_PASSE")
if mdp:
    saisie = st.text_input("Mot de passe", type="password")
    if saisie != mdp:
        st.stop()

# Lecture des donnees Kobo
@st.cache_data(ttl=1800)
def lire_kobo(uid):
    url = f"{SERVEUR}/api/v2/assets/{uid}/data/?format=json"
    entete = {"Authorization": f"Token {TOKEN}"}
    lignes = []
    while url:
        rep = requests.get(url, headers=entete)
        rep.raise_for_status()
        contenu = rep.json()
        lignes += contenu["results"]
        url = contenu["next"]
    return pd.DataFrame(lignes)

# Nettoyage des modalites
def nettoyer(texte):
    t = unicodedata.normalize("NFD", str(texte).lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return t.replace("_", " ").replace("-", " ").strip()

codes_echelle = {nettoyer(e).replace(" ", ""): e for e in ECHELLE}

def niveau(valeur):
    cle = nettoyer(valeur).replace(" ", "")
    if cle.startswith("trs"):
        cle = "tres" + cle[3:]
    return codes_echelle.get(cle)

def niveau_note(valeur):
    return NOTES.get(nettoyer(valeur).replace(" ", "")) or niveau(valeur)

def evolution_modalite(valeur):
    v = nettoyer(valeur)
    if "amelior" in v:
        return "Amelioree"
    if "identique" in v:
        return "Restee identique"
    if "degrad" in v:
        return "Degradee"
    if "sais" in v:
        return "Je ne sais pas"
    return None

def libelle(c):
    return c.split("/")[-1].replace("_", " ").capitalize()

def est_meta(c):
    return c.startswith("_") or c.startswith("meta") or c in ("jour", "formhub/uuid")

# Projection de la date d'atteinte de l'objectif
def projeter(cumul, cible):
    reste = OBJECTIF / 100 * cible - cumul.iloc[-1]
    if reste <= 0:
        return "objectif atteint"
    rythme = cumul.diff().tail(7).mean()
    if not rythme or rythme <= 0:
        return "rythme trop faible pour projeter"
    jours = math.ceil(reste / rythme)
    fin = pd.Timestamp(date.today()) + pd.Timedelta(days=jours)
    return fin.strftime("%d/%m/%Y")

def statut(taux):
    if taux >= OBJECTIF:
        return ":green[Objectif atteint]"
    if taux >= 70:
        return ":gray[En cours]"
    return ":red[A relancer]"

# Classe de chaque repondant
def classe_de(df, p):
    col = p.get("col_classe")
    if col:
        serie = df[col]
    else:
        trouvees = [c for c in df.columns if "classe" in c.lower()]
        cp = [c for c in df.columns if "parcours" in c.lower()]
        cn = [c for c in df.columns if "niveau" in c.lower()]
        if trouvees:
            serie = df[trouvees[0]]
        else:
            parcours = df[cp[0]].astype(str).str.upper().str.extract(r"(TSSP|LSPL)")[0]
            niveau_ = df[cn[0]].astype(str).str.extract(r"(\d)")[0]
            serie = parcours + niveau_
    return serie.astype(str).str.upper().str.replace(r"[ _]", "", regex=True)

# Calcul des indicateurs de progression
def indicateurs(p):
    df = lire_kobo(p["uid"])
    cible = sum(p["classes"].values()) if p["classes"] else p["cible"]
    df["jour"] = pd.to_datetime(df["_submission_time"]).dt.tz_localize(None).dt.normalize()
    par_jour = df.groupby("jour").size()
    jours = pd.date_range(par_jour.index.min(), pd.Timestamp(date.today()))
    cumul = par_jour.reindex(jours, fill_value=0).cumsum()

    par_classe = None
    recus_classe = None
    if p["classes"]:
        n = classe_de(df, p).value_counts()
        recus_classe = pd.Series({c: int(n.get(c, 0)) for c in p["classes"]})
        par_classe = pd.Series({c: min(n.get(c, 0) / e * 100, 100) for c, e in p["classes"].items()})

    return {
        "df": df,
        "recus": len(df),
        "cible": cible,
        "taux": min(len(df) / cible * 100, 100),
        "cumul": cumul,
        "par_classe": par_classe,
        "recus_classe": recus_classe,
        "fin": projeter(cumul, cible),
    }

# Calcul des resultats des questionnaires d'administration
def fusionner(df, col):
    if not col:
        return None
    cols = [c for c in df.columns if c == col or re.fullmatch(re.escape(col) + r"(_001)+", c)]
    if not cols:
        return None
    serie = df[cols].apply(lambda ligne: next((v for v in ligne if pd.notna(v)), None), axis=1)
    return serie, cols

def satisfaction(df, col):
    res = fusionner(df, col)
    if res is None:
        return None
    serie, cols = res
    vals = serie.dropna().map(niveau).dropna()
    if len(vals) == 0:
        return None
    comptes = vals.value_counts().reindex(ECHELLE).dropna()
    return comptes / len(vals) * 100, len(vals), cols

def evolution(df, col):
    res = fusionner(df, col)
    if res is None:
        return None
    vals = res[0].dropna().map(evolution_modalite).dropna()
    if len(vals) == 0:
        return None
    ordre = ["Amelioree", "Restee identique", "Degradee", "Je ne sais pas"]
    return vals.value_counts().reindex(ordre).dropna() / len(vals) * 100

def priorites(df, col):
    if not col or col not in df.columns:
        return None
    reponses = df[col].dropna().astype(str)
    if len(reponses) == 0:
        return None
    mots = reponses.str.split().explode()
    pct = (mots.value_counts().head(6) / len(reponses) * 100).sort_values()
    pct.index = [m.replace("_", " ").capitalize() for m in pct.index]
    return pct

def colonnes_notees(df, exclure=None):
    res = []
    for c in df.columns:
        if est_meta(c) or c == exclure:
            continue
        s = df[c].dropna()
        if len(s) == 0 or s.map(lambda v: isinstance(v, (list, dict))).any():
            continue
        if s.map(niveau).notna().mean() >= 0.8:
            res.append(c)
    return res

def positifs(df, colonnes):
    res = {}
    for c in colonnes:
        v = df[c].dropna().map(niveau).dropna()
        if len(v) > 0:
            res[c] = v.isin(SATISFAITS).mean() * 100
    return pd.Series(res)

# Calcul des resultats de l'evaluation des enseignants
def table_eval(df):
    lignes = []
    for c in df.columns:
        m = MODELE_EVAL.match(c)
        if not m:
            continue
        v = df[c].dropna().map(niveau_note).dropna()
        if len(v) == 0:
            continue
        lignes.append({
            "colonne": c,
            "classe": m.group(1).upper(),
            "matiere": m.group(2).replace("_", " ").capitalize(),
            "critere": m.group(3).upper(),
            "n": len(v),
            "pos": v.isin(SATISFAITS).mean() * 100,
        })
    return pd.DataFrame(lignes)

def score_moyen_par_repondant(df, colonnes):
    notes = df[colonnes].apply(
        lambda col: col.map(lambda v: SCORE.get(niveau_note(v)) if pd.notna(v) else np.nan))
    return notes.apply(pd.to_numeric, errors="coerce").mean(axis=1)

# Apercu des valeurs de chaque colonne (diagnostic)
def apercu_valeurs(df):
    lignes = []
    for c in df.columns:
        if est_meta(c):
            continue
        s = df[c].dropna()
        if len(s) == 0:
            lignes.append({"colonne": c, "reponses": 0, "valeurs frequentes": ""})
            continue
        top = s.astype(str).value_counts().head(3)
        txt = " | ".join(f"{k[:30]} ({v})" for k, v in top.items())
        lignes.append({"colonne": c, "reponses": len(s), "valeurs frequentes": txt})
    return pd.DataFrame(lignes)

# Indicateur de satisfaction par questionnaire
def indice_positif(p, r):
    df = r["df"]
    if p["type"] == "admin":
        sat = satisfaction(df, p["col_global"])
        if sat is not None:
            return sat[0][sat[0].index.isin(SATISFAITS)].sum()
        notees = colonnes_notees(df, exclure=p["col_global"])
        if notees:
            return positifs(df, notees).mean()
        return None
    tab = table_eval(df)
    if len(tab) == 0:
        return None
    return (tab["pos"] * tab["n"]).sum() / tab["n"].sum()

# Graphiques
def montrer(fig, zone=None):
    fig.tight_layout()
    (zone or st).pyplot(fig)
    plt.close(fig)

def donut(taux, couleur):
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    ax.pie([taux, 100 - taux], colors=[couleur, GRIS_CLAIR], startangle=90, counterclock=False,
           wedgeprops={"width": 0.28, "edgecolor": "white"})
    ax.text(0, 0, f"{taux:.0f} %", ha="center", va="center", fontsize=18, fontweight="bold", color=NOIR)
    return fig

def camembert(serie, titre):
    fig, ax = plt.subplots(figsize=(4, 3.4))
    teintes = [ROUGE, NOIR, GRIS, GRIS_CLAIR]
    _, _, textes = ax.pie(serie.values, labels=serie.index, colors=teintes[:len(serie)], startangle=90,
                          counterclock=False, autopct=lambda x: f"{x:.0f} %" if x > 0 else "",
                          textprops={"fontsize": 9}, wedgeprops={"edgecolor": "white"})
    for texte, teinte in zip(textes, teintes):
        texte.set_color(NOIR if teinte == GRIS_CLAIR else "white")
    ax.set_title(titre, fontsize=10)
    return fig

def donut_modalites(distribution):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    couleurs = [COULEURS_ECHELLE[m] for m in distribution.index]
    ax.pie(distribution.values, colors=couleurs, startangle=90, counterclock=False,
           wedgeprops={"width": 0.38, "edgecolor": "white"},
           autopct=lambda x: f"{x:.0f} %" if x >= 6 else "", pctdistance=0.81,
           textprops={"color": "white", "fontsize": 9})
    ax.legend(distribution.index, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, fontsize=9)
    ax.set_title("Appreciation globale", fontsize=10)
    return fig

def barres(serie, couleur, titre):
    fig, ax = plt.subplots(figsize=(5, 3))
    b = ax.barh(serie.index, serie.values, color=couleur)
    ax.bar_label(b, fmt="%.0f %%", padding=4, color=NOIR)
    ax.set_xlim(0, 115)
    ax.set_title(titre, fontsize=10)
    ax.grid(color=GRIS_CLAIR, axis="x")
    return fig

def sucettes(serie, couleur, titre):
    fig, ax = plt.subplots(figsize=(5, 3))
    y = range(len(serie))
    ax.hlines(y, 0, serie.values, color=GRIS, linewidth=2)
    ax.plot(serie.values, y, "o", color=couleur, markersize=9)
    for i, v in enumerate(serie.values):
        ax.text(v + 3, i, f"{v:.0f} %", va="center", fontsize=9)
    ax.set_yticks(list(y))
    ax.set_yticklabels(serie.index)
    ax.set_xlim(0, 115)
    ax.set_title(titre, fontsize=10)
    ax.grid(color=GRIS_CLAIR, axis="x")
    return fig

def classes_barres(resultats, noms):
    classes = list(CLASSES)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    largeur = 0.38
    for k, nom in enumerate(noms):
        pos = [i + (k - 0.5) * largeur for i in range(len(classes))]
        b = ax.bar(pos, resultats[nom]["par_classe"].values, width=largeur, color=couleur_de[nom], label=nom)
        ax.bar_label(b, fmt="%.0f %%", padding=2, fontsize=9)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.axhline(OBJECTIF, color=NOIR, linestyle="--", linewidth=1)
    ax.text(len(classes) - 0.5, OBJECTIF + 1.5, f"Objectif {OBJECTIF} %", ha="right", fontsize=8)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Taux de reponse (%)")
    ax.grid(color=GRIS_CLAIR, axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    return fig

def courbe_progression(r, couleur):
    fig, ax = plt.subplots(figsize=(8, 3.2))
    par_jour = r["cumul"].diff().fillna(r["cumul"].iloc[0])
    ax.bar(par_jour.index, par_jour.values, color=GRIS_CLAIR, label="Reponses par jour")
    ax.set_ylabel("Reponses par jour")
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(r["cumul"].index, r["cumul"] / r["cible"] * 100, color=couleur, linewidth=2.5)
    ax2.axhline(OBJECTIF, color=NOIR, linestyle="--", linewidth=1)
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Taux cumule (%)")
    ax.tick_params(axis="x", rotation=30)
    return fig

def likert(df, colonnes):
    lignes = {}
    for c in colonnes:
        v = df[c].dropna().map(niveau).dropna()
        if len(v) > 0:
            lignes[libelle(c)] = v.value_counts(normalize=True).reindex(ECHELLE).fillna(0) * 100
    tab = pd.DataFrame(lignes).T
    tab = tab.assign(pos=tab[SATISFAITS].sum(axis=1)).sort_values("pos").drop(columns="pos")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.32 * len(tab) + 1.2)))
    gauche = np.zeros(len(tab))
    for niv in ECHELLE:
        ax.barh(tab.index, tab[niv], left=gauche, color=COULEURS_ECHELLE[niv], label=niv)
        gauche += tab[niv].values
    ax.set_xlim(0, 100)
    ax.set_xlabel("%")
    ax.legend(ncol=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8)
    return fig

def carte_chaleur(tab):
    cmap = LinearSegmentedColormap.from_list("rn", [ROUGE, "#F2F2F2", NOIR])
    plancher = 40
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * tab.shape[1] + 3), max(3, 0.38 * tab.shape[0] + 1.2)))
    ax.imshow(tab.values, cmap=cmap, vmin=plancher, vmax=100, aspect="auto")
    ax.set_xticks(range(tab.shape[1]))
    ax.set_xticklabels(tab.columns)
    ax.set_yticks(range(tab.shape[0]))
    ax.set_yticklabels(tab.index)
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if v < 52 or v > 85 else NOIR)
    ax.spines[:].set_visible(False)
    return fig

def histo_densite(valeurs, couleur, xlabel, titre):
    v = np.asarray(valeurs, dtype=float)
    v = v[~np.isnan(v)]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(v, bins=12, density=True, color=GRIS_CLAIR, edgecolor="white")
    if len(v) >= 3 and v.std() > 0:
        h = 1.06 * v.std() * len(v) ** (-0.2)
        x = np.linspace(v.min() - h, v.max() + h, 200)
        dens = np.exp(-0.5 * ((x[:, None] - v[None, :]) / h) ** 2).sum(axis=1) / (len(v) * h * np.sqrt(2 * np.pi))
        ax.plot(x, dens, color=couleur, linewidth=2.5)
        ax.axvline(np.median(v), color=NOIR, linestyle="--", linewidth=1)
        ax.text(np.median(v), ax.get_ylim()[1] * 0.95, f" mediane : {np.median(v):.1f}", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Densite")
    ax.set_title(titre, fontsize=10)
    return fig

def boite_par_classe(valeurs, classes, couleur, titre):
    noms = [c for c in CLASSES if (classes == c).sum() >= 2]
    donnees = [valeurs[classes == c].dropna().values for c in noms]
    fig, ax = plt.subplots(figsize=(5, 3))
    bp = ax.boxplot(donnees, patch_artist=True, medianprops={"color": NOIR})
    for boite in bp["boxes"]:
        boite.set(facecolor=couleur, alpha=0.6, edgecolor=NOIR)
    ax.set_xticks(range(1, len(noms) + 1))
    ax.set_xticklabels(noms)
    ax.set_ylim(0, 5)
    ax.set_ylabel("Score moyen (sur 5)")
    ax.set_title(titre, fontsize=10)
    ax.grid(color=GRIS_CLAIR, axis="y")
    return fig

def courbe_heures(heures, couleur):
    comptes = heures.value_counts().reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.fill_between(comptes.index, comptes.values, color=GRIS_CLAIR)
    ax.plot(comptes.index, comptes.values, color=couleur, linewidth=2.5, marker="o", markersize=3)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Heure de la journee")
    ax.set_ylabel("Reponses")
    ax.set_title("Moment de remplissage", fontsize=10)
    return fig

# En-tete
entete = st.columns([1, 7])
if LOGO.exists():
    entete[0].image(str(LOGO), width=95)
entete[1].title("Enquetes d'evaluation du CNFSDP")
entete[1].caption(f"Annee academique 2025-2026 | Situation au {date.today().strftime('%d/%m/%Y')} | Objectif de collecte : {OBJECTIF} %")

with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=110)
    st.markdown("**Centre National de Formation en Statistique, Demographie et Planification**")
    if st.button("Actualiser les donnees"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Donnees issues de KoboToolbox, actualisees toutes les 30 minutes.")
    st.caption("Questionnaires anonymes. Seuls des resultats agreges sont affiches.")

resultats = {nom: indicateurs(p) for nom, p in enquetes.items()}
positif = {nom: indice_positif(p, resultats[nom]) for nom, p in enquetes.items()}
noms_etu = [nom for nom, p in enquetes.items() if p["classes"]]

onglets = st.tabs(["Synthese", "Progression", "Resultats", "Qualite des donnees"])

# Onglet 1 : synthese
with onglets[0]:
    colonnes = st.columns(3)
    for col, (nom, r) in zip(colonnes, resultats.items()):
        with col.container(border=True):
            st.markdown(f"**{nom}**")
            montrer(donut(r["taux"], couleur_de[nom]))
            st.caption(f"{r['recus']} reponses sur {r['cible']} attendues")
            st.markdown(statut(r["taux"]))
            st.caption(f"Date prevue pour {OBJECTIF} % : {r['fin']}")

    st.subheader("Indicateurs cles")
    synthese = pd.DataFrame([{
        "Questionnaire": nom,
        "Reponses": f"{r['recus']} / {r['cible']}",
        "Taux de reponse": f"{r['taux']:.0f} %",
        "Appreciation positive": f"{positif[nom]:.0f} %" if positif[nom] is not None else "-",
        f"Date prevue pour {OBJECTIF} %": r["fin"],
    } for nom, r in resultats.items()])
    st.dataframe(synthese, hide_index=True)
    st.caption("Appreciation positive : part des reponses Excellent, Tres bien et Bien. Resultats provisoires.")

# Onglet 2 : progression
with onglets[1]:
    st.subheader("Progression par classe")
    montrer(classes_barres(resultats, noms_etu))
    classes = list(CLASSES)
    tableau = pd.DataFrame({
        nom: [f"{resultats[nom]['recus_classe'][c]}/{CLASSES[c]} ({resultats[nom]['par_classe'][c]:.0f} %)"
              for c in classes]
        for nom in noms_etu
    }, index=classes)
    st.dataframe(tableau)
    st.download_button("Exporter le tableau (CSV)", tableau.to_csv().encode("utf-8"),
                       file_name="progression_par_classe.csv", mime="text/csv")
    for nom in noms_etu:
        retard = resultats[nom]["par_classe"].idxmin()
        st.caption(f"{nom} : classe la moins avancee, {retard} ({resultats[nom]['par_classe'][retard]:.0f} %)")

    st.subheader("Repartition des repondants par classe")
    pies = st.columns(len(noms_etu))
    for zone, nom in zip(pies, noms_etu):
        montrer(camembert(resultats[nom]["recus_classe"], nom), zone)

    st.subheader("Rythme de la collecte")
    sous = st.tabs(list(enquetes))
    for zone, (nom, r) in zip(sous, resultats.items()):
        with zone:
            st.write(f"Date prevue pour atteindre {OBJECTIF} % : **{r['fin']}**")
            montrer(courbe_progression(r, couleur_de[nom]))

# Onglet 3 : resultats
def afficher_admin(nom, p, r):
    df = r["df"]
    couleur = couleur_de[nom]
    rien = True
    sat = satisfaction(df, p["col_global"])
    if sat is not None:
        rien = False
        distribution, n, cols = sat
        m1, m2 = st.columns(2)
        m1.metric("Appreciation globale positive", f"{distribution[distribution.index.isin(SATISFAITS)].sum():.0f} %")
        m2.metric("Reponses exploitees", f"{n} sur {r['recus']}")
        if len(cols) > 1:
            st.caption("Colonnes regroupees : " + ", ".join(cols))
        g1, g2 = st.columns(2)
        montrer(donut_modalites(distribution), g1)
        evo = evolution(df, p["col_evolution"])
        if evo is not None:
            montrer(sucettes(evo.iloc[::-1], couleur, "Evolution par rapport a l'annee precedente"), g2)

    notees = colonnes_notees(df, exclure=p["col_global"])
    if notees:
        rien = False
        pos = positifs(df, notees).sort_values()
        st.metric("Satisfaction moyenne, toutes les questions notees", f"{pos.mean():.0f} %")
        forts, faibles = pos.tail(5), pos.head(5)
        forts.index = [libelle(c) for c in forts.index]
        faibles.index = [libelle(c) for c in faibles.index]
        g3, g4 = st.columns(2)
        montrer(sucettes(forts, NOIR, "Points forts (% positif)"), g3)
        montrer(sucettes(faibles.iloc[::-1], ROUGE, "Points a ameliorer (% positif)"), g4)
        with st.expander("Detail de toutes les questions"):
            montrer(likert(df, notees))

    prio = priorites(df, p["col_priorites"])
    if prio is not None:
        rien = False
        montrer(barres(prio, couleur, "Priorites d'amelioration les plus citees"))
    if rien:
        st.info("Aucun resultat exploitable pour l'instant. Consulte l'onglet Qualite des donnees.")

def afficher_eval(nom, p, r):
    df = r["df"]
    couleur = couleur_de[nom]
    tab = table_eval(df)
    if len(tab) == 0:
        st.info("Aucune question notee reconnue. Consulte l'onglet Qualite des donnees.")
        return
    classes_rep = classe_de(df, p)
    choix = st.selectbox("Classe", ["Toutes"] + list(CLASSES), key="classe_eval")
    if choix != "Toutes":
        tab = tab[tab["classe"] == choix]
        if len(tab) == 0:
            st.info("Pas encore de reponse pour cette classe.")
            return
        df = df[classes_rep == choix]
        classes_rep = classes_rep[classes_rep == choix]

    m1, m2, m3 = st.columns(3)
    m1.metric("Satisfaction positive", f"{(tab['pos'] * tab['n']).sum() / tab['n'].sum():.0f} %")
    m2.metric("Matieres evaluees", tab.groupby(["classe", "matiere"]).ngroups)
    m3.metric("Etudiants ayant repondu", len(df))

    colonnes_notes = list(tab["colonne"])
    scores = score_moyen_par_repondant(df, colonnes_notes)
    g1, g2 = st.columns(2)
    montrer(histo_densite(scores.values, couleur, "Score moyen (sur 5)", "Distribution des scores moyens"), g1)
    montrer(boite_par_classe(scores, classes_rep, couleur, "Score moyen par classe"), g2)

    vals = pd.concat([df[c].dropna().map(niveau_note) for c in colonnes_notes]).dropna()
    repartition = vals.value_counts(normalize=True).reindex(ECHELLE).dropna() * 100
    g3, g4 = st.columns(2)
    montrer(donut_modalites(repartition), g3)

    resume = tab.groupby(["classe", "matiere"]).apply(
        lambda g: pd.Series({"n": g["n"].min(), "pos": (g["pos"] * g["n"]).sum() / g["n"].sum()}),
        include_groups=False).reset_index()
    resume = resume[resume["n"] >= MIN_REPONSES]
    if len(resume) > 0:
        resume["nom"] = resume["matiere"] + " (" + resume["classe"] + ")"
        classement = resume.set_index("nom")["pos"].sort_values().tail(12)
        montrer(sucettes(classement, couleur, "Satisfaction par matiere (% positif)"), g4)

        st.markdown("**Satisfaction par critere et par matiere (% positif)**")
        sous = tab[tab["n"] >= MIN_REPONSES].copy()
        sous["nom"] = sous["matiere"] + " (" + sous["classe"] + ")"
        carte = sous.pivot_table(index="nom", columns="critere", values="pos")
        if carte.shape[0] > 0:
            montrer(carte_chaleur(carte))
    st.caption(f"Les matieres evaluees par moins de {MIN_REPONSES} etudiants ne sont pas affichees.")

with onglets[2]:
    st.warning("Resultats provisoires, calcules sur les reponses deja recues. Ils evolueront jusqu'a la fin de la collecte.")
    sous = st.tabs(list(enquetes))
    for zone, (nom, p) in zip(sous, enquetes.items()):
        with zone:
            if p["type"] == "admin":
                afficher_admin(nom, p, resultats[nom])
            else:
                afficher_eval(nom, p, resultats[nom])

# Onglet 4 : qualite des donnees
with onglets[3]:
    sous = st.tabs(list(enquetes))
    for zone, (nom, r) in zip(sous, resultats.items()):
        with zone:
            df = r["df"]
            if "start" in df.columns and "end" in df.columns:
                debut = pd.to_datetime(df["start"], utc=True, errors="coerce")
                fin = pd.to_datetime(df["end"], utc=True, errors="coerce")
                duree = (fin - debut).dt.total_seconds() / 60
                duree = duree[(duree > 0.3) & (duree < 120)]
                heures = debut.dt.tz_convert("Africa/Douala").dt.hour.dropna().astype(int)
                m1, m2 = st.columns(2)
                m1.metric("Duree mediane de remplissage", f"{duree.median():.1f} min" if len(duree) else "-")
                m2.metric("Reponses recues", r["recus"])
                g1, g2 = st.columns(2)
                if len(duree) > 0:
                    montrer(histo_densite(duree.values, couleur_de[nom], "Minutes", "Duree de remplissage"), g1)
                montrer(courbe_heures(heures, couleur_de[nom]), g2)
            with st.expander("Diagnostic : colonnes et valeurs recues"):
                st.dataframe(apercu_valeurs(df))

st.caption(f"Donnees actualisees a {datetime.now().strftime('%H:%M')} | CNFSDP, evaluation 2025-2026")
