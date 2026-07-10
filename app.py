import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import unicodedata
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.pipeline import Pipeline, FeatureUnion

warnings.filterwarnings("ignore")

# ─── Configuration ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analyse & Prévision des Pannes",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = r"X:\Maintenance\Donnees_Synthese\Base_donnees_maintenance.xlsx"

TYPE_COLORS = {
    "MECHANICAL": "#e67e22",
    "ELECTRICAL": "#2980b9",
    "OTHER": "#27ae60",
}

EQUIP_COLORS = {"EM": "#1a6fa8", "BUN": "#d97b00"}  # bleu EM, orange foncé BUN — jamais interchangeables

# Liste exhaustive des types de machines (ordre décroissant de longueur = matching glouton)
_MACHINES = [
    # 3 mots
    "PETRIN PATE FINALE", "TABLE DE BOULAGE", "PAN CONVEYOR SYS",
    "VIS DOUGH PUMP", "LEVE BAC SPONGE", "DOSAGE EAU SPONGE",
    "DEPILEUR PLAQUE", "SECURITES PORTE", "TAPIS DEPOSE",
    # 2 mots
    "EQUIMT EMBALLAGE", "METRING PUMP", "DOUGH PUMP",
    "SPRAY GLAZE", "PETRIN SPONGE", "BAC SPONGE",
    "PATE FINALE", "DE BOULAGE", "CONVEYOR SYS",
    "DOSAGE HUILE", "DOSAGE SUCRE", "RACK OFF", "PAN COOLER",
    "ALIM ELECTRIQUE", "CONVOYEUR SEEDER", "EMPILEUR PLAQUE",
    "DIVISEUSE BAGELS", "TABLE FACONNAGE",
    # Nouveaux (base BUNS / MUFFINS)
    "PAN CLEANER", "ICE WATER", "PACK STACK", "PILLO PACK",
    "DOUGH MIXEUR", "FLOUR RECLAIN", "FLOUR RECLAIM",
    "DEPANNEUR", "DEMOULEUR", "EMPILEUR", "DEPILEUR",
    "EMBALLEUSE", "CONVOYEUR", "PETRIN", "PROOFER",
    "GROUPE FROID", "DOSAGE ESTEVE", "POMPE A PATE",
    "IMPRIMANTE", "RETOURNEUR", "RETOURNEMENT",
    "LEVE BAC", "FILTRE", "TRAYWASHER", "TRAYDRYER",
    "MACHINE A LAVER", "MACHINE A SECHER", "SHIFTER", "CYCLOFILTRE",
    # 1 mot
    "DO-FLOW", "DO-PUMP", "FOUR", "PIAB", "AUTRE", "PUMP",
    "COUTEAUX", "ZIGZAG", "BOULAGE", "COOLER", "SPONGE",
    "GLAZE", "PLAQUE", "PORTE", "DEPOSE", "FINALE", "ETUVE",
    "SEEDER", "BALANCELLES", "SUCRE", "AGV",
    "ROTARY", "BAGELS", "FACONNAGE", "LAMINOIRE", "DIVISEUSE", "ROLLER",
    "STACKER", "ARMOIRE",
]

# Mots-clés d'incidents opérationnels (valeur par défaut — modifiable dans la sidebar)
_DEFAULT_EXCL_KW = [
    "BOURRAGE", r"FILMS? OVER RUN", r"\bFOR\b", "DOLLIES", "PILE",
    "OVERRUN", "PLUSIEURS", "REGLAGE", "REGLER", "ARRET", "OUBLI DE FILM",
]
_KW_VERSION = "v4"  # Incrémenter pour forcer le rechargement des défauts en session

def _excl_mask(desc_series: pd.Series, kw_list: list) -> pd.Series:
    """Retourne True pour les lignes à GARDER (description ne contient aucun mot-clé exclu)."""
    if not kw_list:
        return pd.Series(True, index=desc_series.index)
    pattern = "|".join(kw_list)  # pas d'échappement : les mots-clés sont des patterns regex
    return ~desc_series.str.contains(pattern, case=False, na=False, regex=True)


def _split_produit_machine(val: str):
    """Retourne (produit, machine) en détectant le suffixe machine connu."""
    v = str(val).strip()
    v_upper = v.upper()
    for mach in _MACHINES:
        if v_upper.endswith(mach):
            prod = v[:len(v) - len(mach)].strip()
            return (prod if prod else v), mach
    return v, "AUTRE"

# ─── Chargement & préparation ──────────────────────────────────────────────────
def _normalise(s: str) -> str:
    """Supprime les accents et met en majuscules pour la comparaison."""
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().upper()


def _localisation_to_machine(loc: str) -> str:
    """Extrait le type de machine à partir d'une LOCALISATION (correspondance partielle, insensible aux accents)."""
    v = str(loc).strip()
    v_norm = re.sub(r'\bL[12]\b', '', _normalise(v)).strip()
    for mach in _MACHINES:
        if mach in v_norm:
            return mach
    # Fallback : version raccourcie de la localisation
    return v[:45].strip()


# Mots-clés pour la classification automatique des types de panne
_ELECTRICAL_KW = [
    "ELECTR", "TRANSFO", "CAPTEUR", "VARIATEUR", "MOTEUR", "DISJONCTEUR",
    "COURT-CIRCUIT", "CABLE", "ALIMENTATION", "ONDULEUR", "CONTACTEUR",
    "INTERRUPTEUR", "BORNE", "SONDE", "ENCODEUR", "ENCODER", "SERVO",
    "24V", "230V", "400V", "THERMIQUE", "AUTOMATE", "DETECTEUR",
    "PRESSOSTAT", "THERMOSTAT", "RELAIS", "FUSIBLE", "TENSION",
    "COURANT", "PLC", "INVERTER", "FREQUENCY",
]
_MECHANICAL_KW = [
    "MECANI", "COURROIE", "ROULEMENT", "USURE", "PISTON",
    "VERIN", "RESSORT", "GRAISSE", "LUBRI", "ENGRENAGE", "CHAINE",
    "PIGNON", "ARBRE", "PALIER", "JOINT", "CASSE", "BRISE",
    "DECHIRE", "DEFORMATION", "SOUDURE", "BLOCAGE", "COINC",
    "VIBRATION", "ROUILLE", "CORROS", "RUPTURE", "FISSURE",
]


def _predict_type_keywords(desc: str) -> str:
    """Classification par mots-clés — autonome, sans fichier externe."""
    d = _normalise(str(desc))
    for kw in _ELECTRICAL_KW:
        if kw in d:
            return "ELECTRICAL"
    for kw in _MECHANICAL_KW:
        if kw in d:
            return "MECHANICAL"
    return "OTHER"


@st.cache_data(ttl=3600)  # rafraîchit toutes les heures sur le cloud
def load_data(path: str) -> pd.DataFrame:
    import io, base64, requests
    df = None

    # 1️⃣  GitHub Gist (Streamlit Cloud — données privées, sync auto)
    gh = st.secrets.get("github", {})
    if gh.get("token") and gh.get("gist_id"):
        try:
            r = requests.get(
                f"https://api.github.com/gists/{gh['gist_id']}",
                headers={"Authorization": f"token {gh['token']}"},
                timeout=10,
            )
            r.raise_for_status()
            b64 = r.json()["files"]["data.txt"]["content"]
            df = pd.read_excel(io.BytesIO(base64.b64decode(b64)), sheet_name="Base")
        except Exception:
            df = None

    # 2️⃣  Secret base64 de secours (ancien mécanisme)
    if df is None:
        sec = st.secrets.get("data", {})
        if sec.get("excel_b64"):
            df = pd.read_excel(io.BytesIO(base64.b64decode(sec["excel_b64"])), sheet_name="Base")

    # 3️⃣  Fichier local / réseau (développement)
    if df is None:
        df = pd.read_excel(path, sheet_name="Base")

    df.columns = df.columns.str.strip()

    # Date
    df["date_dt"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["date_dt"])
    df["date"] = df["date_dt"].dt.date

    # Description principale
    df["desc"] = df["MODIFICATIONS / DYSFONCTIONNEMENTS"].fillna("").str.upper().str.strip()

    # Localisation → produit (affichage) + machine (catégorie)
    df["produit"] = df["LOCALISATION"].fillna("").str.strip()
    df["machine"] = df["produit"].apply(_localisation_to_machine)

    # Équipement : MUFFINS → EM, BUNS → BUN (BATIMENT / ATELIER exclus)
    _ligne_map = {"MUFFINS": "EM", "BUNS": "BUN"}
    df["equipement"] = df["LIGNE"].fillna("").str.upper().str.strip().map(_ligne_map)
    df = df.dropna(subset=["equipement"])

    # Durée : non disponible dans la base fusionnée
    df["duree_min"] = 0

    # Type : classification par mots-clés (autonome, sans fichier externe)
    df["type"] = df["desc"].apply(_predict_type_keywords)

    # Colonnes supplémentaires — accès défensif (noms pouvant varier selon la version du fichier)
    def _col(df, *names):
        for n in names:
            if n in df.columns:
                return df[n].fillna("").astype(str).str.strip()
        return pd.Series("", index=df.index)

    df["poste"]      = _col(df, "Poste", "POSTE")
    df["technicien"] = _col(df, "Technicien 1", "TECHNICIEN 1", "Technicien1")
    df["actions"]    = _col(df, "ACTIONS", "Actions")
    df["etat"]       = _col(df, "Etat", "État", "ETAT", "etat")

    # Colonnes temporelles
    df["annee"] = df["date_dt"].dt.year
    df["mois"] = df["date_dt"].dt.to_period("M").astype(str)
    df["annee_mois"] = df["date_dt"].dt.to_period("M").astype(str)

    # Poids temporel : décroissance exponentielle, demi-vie = 3 ans
    today = pd.Timestamp.today().normalize()
    jours = (today - df["date_dt"]).dt.days.clip(lower=0)
    df["poids"] = 2 ** (-jours / (3 * 365))

    return df


@st.cache_resource
def build_classifier(df_train: pd.DataFrame):
    """Entraîne un pipeline (TF-IDF mots + caractères) + Régression Logistique sur la base labellisée."""
    sub = df_train[df_train["desc"].str.len() > 3].copy()
    # Enrichir le texte avec le type de machine : signal discriminant fort
    sub["text_feat"] = sub["desc"] + " " + sub["machine"].str.replace(" ", "_")
    X = sub["text_feat"]
    y = sub["type"]
    w = sub["poids"]

    X_train, X_test, y_train, y_test, w_train, _ = train_test_split(
        X, y, w, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                FeatureUnion([  # type: ignore[arg-type]
                    ("word", TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        max_features=15_000,
                        sublinear_tf=True,
                        min_df=2,
                    )),
                    ("char", TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=10_000,
                        sublinear_tf=True,
                        min_df=3,
                    )),
                ]),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    C=1.0,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train, clf__sample_weight=w_train)
    y_pred = pipeline.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=["MECHANICAL", "ELECTRICAL", "OTHER"])
    acc = accuracy_score(y_test, y_pred)

    return pipeline, report, cm, acc, ["MECHANICAL", "ELECTRICAL", "OTHER"]


# ─── Chargement ────────────────────────────────────────────────────────────────
df = load_data(DATA_FILE)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🔧 Paramètres")

pages = [
    "📊 Tableau de bord",
    "🏭 Analyse Machines",
    "🤖 Classification automatique",
    "📈 Prévision des pannes",
]
page = st.sidebar.radio("Navigation", pages)

# Filtres globaux
st.sidebar.markdown("---")
st.sidebar.subheader("Filtres globaux")

# CSS pour les boutons toggle EM / BUN + neutralisation du rouge Streamlit
st.markdown("""
<style>
/* Neutralise le rouge/corail de Streamlit sur tous les boutons primary */
div[data-testid="stButton"] button[kind="primary"] {
    background-color: #4a4a4a !important;
    border-color: #4a4a4a !important;
    color: white !important;
    opacity: 1 !important;
}
/* Bouton désactivé */
div[data-testid="stButton"] button[kind="secondary"] {
    opacity: 0.35 !important;
}
/* Bouton EM — bleu */
div[data-testid="stSidebarContent"] section div[data-testid="column"]:nth-child(1)
    div[data-testid="stButton"] button {
    background-color: #1a6fa8 !important;
    border-color: #1a6fa8 !important;
    color: white !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    padding: 0.55rem 0 !important;
    border-radius: 8px !important;
    white-space: pre-line !important;
}
/* Bouton BUN — orange foncé */
div[data-testid="stSidebarContent"] section div[data-testid="column"]:nth-child(2)
    div[data-testid="stButton"] button {
    background-color: #d97b00 !important;
    border-color: #d97b00 !important;
    color: white !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    padding: 0.55rem 0 !important;
    border-radius: 8px !important;
    white-space: pre-line !important;
}
/* Multiselect tags : remplace le rouge par un gris-bleu doux */
span[data-baseweb="tag"] {
    background-color: #3a6186 !important;
}
/* Couleur d'accentuation générale (sliders, focus…) */
:root {
    --primary-color: #3a6186 !important;
}
</style>
""", unsafe_allow_html=True)

if "em_active" not in st.session_state:
    st.session_state.em_active = True
if "bun_active" not in st.session_state:
    st.session_state.bun_active = True

_col_em, _col_bun = st.sidebar.columns(2)

with _col_em:
    st.markdown('<span id="btn-em"></span>', unsafe_allow_html=True)
    if st.button(
        "🔵 EM\nEnglish Muffin",
        width='stretch',
        type="primary" if st.session_state.em_active else "secondary",
        key="btn_em",
    ):
        st.session_state.em_active = not st.session_state.em_active
        st.rerun()

with _col_bun:
    st.markdown('<span id="btn-bun"></span>', unsafe_allow_html=True)
    if st.button(
        "🟠 BUN\nBun Round",
        width='stretch',
        type="primary" if st.session_state.bun_active else "secondary",
        key="btn_bun",
    ):
        st.session_state.bun_active = not st.session_state.bun_active
        st.rerun()

selected_equip = []
if st.session_state.em_active:
    selected_equip.append("EM")
if st.session_state.bun_active:
    selected_equip.append("BUN")
if not selected_equip:  # sécurité : si tout désactivé on affiche tout
    selected_equip = ["EM", "BUN"]

# Slider période au mois
months_available = sorted(df["annee_mois"].unique(), reverse=True)  # décroissant : plus récent en premier

# Défaut : premier mois de l'année en cours (index le plus élevé parmi les mois de l'année)
_annee_courante = str(pd.Timestamp.today().year)
_debut_idx = max(
    (i for i, m in enumerate(months_available) if m.startswith(_annee_courante)),
    default=len(months_available) - 1,
)

col_d, col_f = st.sidebar.columns(2)
date_debut = col_d.selectbox(
    "De",
    options=months_available,
    index=_debut_idx,
    key="date_debut",
)
date_fin = col_f.selectbox(
    "À",
    options=months_available,
    index=0,  # le plus récent par défaut
    key="date_fin",
)
# Sécurité : normaliser debut <= fin
if date_debut > date_fin:
    date_debut, date_fin = date_fin, date_debut

machines_dispo = sorted(df["machine"].unique())
selected_machines = st.sidebar.multiselect(
    "Type de machine",
    options=machines_dispo,
    default=machines_dispo,
    placeholder="Toutes les machines",
)

with st.sidebar.expander("Filtres avancés"):
    types_available = list(df["type"].unique())
    selected_types = st.multiselect(
        "Types de panne",
        options=types_available,
        default=types_available,
    )
    st.markdown("---")
    st.caption("**Incidents opérationnels à exclure**")
    # Réinitialiser le champ si les défauts ont changé (nouvelle version)
    if st.session_state.get("_excl_kw_version") != _KW_VERSION:
        st.session_state["excl_kw_v3"] = ", ".join(_DEFAULT_EXCL_KW)
        st.session_state["_excl_kw_version"] = _KW_VERSION
    excl_kw_text = st.text_input(
        "Mots-clés (séparés par des virgules)",
        key="excl_kw_v3",
        help="Les pannes dont la description contient un de ces mots sont exclues de toute l'analyse.",
    )
    excluded_kw = [k.strip() for k in excl_kw_text.split(",") if k.strip()]
    seuil_recurrence = st.slider(
        "Exclure déscriptions trop récurrentes (> N fois sur toute la base)",
        min_value=5, max_value=500, value=15, step=5,
        help="Toute description apparaissant plus de N fois dans la base est considérée opérationnelle et exclue.",
    )

# Calcul des descriptions trop récurrentes (sur la base complète)
_desc_freq = df["desc"].value_counts()
_desc_frequentes = set(_desc_freq[_desc_freq > seuil_recurrence].index)

# Appliquer les filtres
_debut_ts = pd.Period(date_debut, freq="M").to_timestamp()
_fin_ts = pd.Period(date_fin, freq="M").to_timestamp() + pd.offsets.MonthEnd(0)

mask = (
    df["date_dt"].between(_debut_ts, _fin_ts)
    & df["type"].isin(selected_types)
    & df["equipement"].isin(selected_equip)
    & df["machine"].isin(selected_machines if selected_machines else machines_dispo)
    & _excl_mask(df["desc"], excluded_kw)
    & ~df["desc"].isin(_desc_frequentes)
)
fdf = df[mask].copy()

# Base complète avec exclusions permanentes (mots-clés + récurrence) — utilisée partout sauf pour les options de filtres
df_base = df[
    _excl_mask(df["desc"], excluded_kw) &
    ~df["desc"].isin(_desc_frequentes)
].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Pannes dans la sélection", f"{len(fdf):,}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TABLEAU DE BORD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Tableau de bord":
    st.title("📊 Tableau de bord — Analyse des Pannes")
    st.caption(f"Données du {fdf['date'].min()} au {fdf['date'].max()} · {len(fdf):,} enregistrements dans la sélection · Source : Base_donnees_maintenance_FUSIONNEE")

    # KPI
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total pannes", f"{len(fdf):,}")
    col2.metric("Machines touchées", f"{fdf['machine'].nunique():,}")
    top_machine = fdf.groupby("machine")["poids"].sum().idxmax() if len(fdf) > 0 else "-"
    top_machine_count = int(fdf[fdf['machine'] == top_machine].shape[0]) if len(fdf) > 0 else 0
    col3.metric("Machine n°1 (pond.)", f"{top_machine[:20]}", f"{top_machine_count} pannes")
    em_count = (fdf['equipement'] == 'EM').sum()
    bun_count = (fdf['equipement'] == 'BUN').sum()
    col4.metric("Pannes EM", f"{em_count:,}")
    col5.metric("Pannes BUN", f"{bun_count:,}")
        # Note : la durée d'arrêt n'est pas disponible dans cette base

    # ── Ligne 1 : EM vs BUN + Top 15 types de machines
    c1, c2 = st.columns([1, 2])

    with c1:
        st.subheader("Répartition EM / BUN",
            help="Parts respectives des pannes survenues sur les lignes English Muffin (EM) et Bun (BUN).")
        equip_counts = fdf["equipement"].value_counts().reset_index()
        equip_counts.columns = ["Ligne", "Nombre"]
        fig_pie = px.pie(
            equip_counts,
            names="Ligne",
            values="Nombre",
            color="Ligne",
            color_discrete_map=EQUIP_COLORS,
            hole=0.4,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_pie, width='stretch')

    with c2:
        st.subheader("Top 15 types de machines — score pondéré",
            help="Classement des types de machines selon un score pondéré : chaque panne est multipliée par son poids temporel (décroissance exponentielle, demi-vie 3 ans). Les pannes récentes comptent donc plus qu'une panne ancienne.")
        top15_m = (
            fdf.groupby(["machine", "equipement"])["poids"]
            .sum()
            .reset_index()
            .rename(columns={"poids": "score", "equipement": "Ligne"})
        )
        # Garder le top 15 machines (somme toutes lignes)
        top15_names = (
            top15_m.groupby("machine")["score"].sum()
            .nlargest(15).index
        )
        top15_m = top15_m[top15_m["machine"].isin(top15_names)]
        # Ordonner par score total
        order = top15_m.groupby("machine")["score"].sum().sort_values().index.tolist()
        top15_m["machine"] = pd.Categorical(top15_m["machine"], categories=order, ordered=True)
        fig_top = px.bar(
            top15_m.sort_values("machine"),
            x="score",
            y="machine",
            orientation="h",
            color="Ligne",
            color_discrete_map=EQUIP_COLORS,
            barmode="stack",
            labels={"score": "Score pondéré", "machine": "Type de machine"},
        )
        fig_top.update_layout(showlegend=False, margin=dict(t=10, b=10), height=420)
        st.plotly_chart(fig_top, width='stretch')

    # ── Ligne 2 : Top produits + Top durées par machine
    c3, c4 = st.columns([2, 1])

    with c3:
        st.subheader("Top 10 localisations les plus touchées (pondéré)",
            help="Localisations (machines/zones) ayant accumulé le score pondéré de pannes le plus élevé. Les pannes récentes pèsent davantage dans ce classement.")
        top10_p = (
            fdf.groupby(["produit", "equipement"])["poids"]
            .sum()
            .reset_index()
            .rename(columns={"poids": "score", "equipement": "Ligne"})
        )
        top10_names = (
            top10_p.groupby("produit")["score"].sum()
            .nlargest(10).index
        )
        top10_p = top10_p[top10_p["produit"].isin(top10_names)]
        order_p = top10_p.groupby("produit")["score"].sum().sort_values().index.tolist()
        top10_p["produit"] = pd.Categorical(top10_p["produit"], categories=order_p, ordered=True)
        fig_prod = px.bar(
            top10_p.sort_values("produit"),
            x="score",
            y="produit",
            orientation="h",
            color="Ligne",
            color_discrete_map=EQUIP_COLORS,
            barmode="stack",
            labels={"score": "Score pondéré", "produit": "Localisation"},
        )
        fig_prod.update_layout(showlegend=False, margin=dict(t=10, b=10), height=380)
        st.plotly_chart(fig_prod, width='stretch')

    with c4:
        st.subheader("Répartition par type de panne (Top 10 machines)",
            help="Répartition MECHANICAL / ELECTRICAL / OTHER pour les 10 machines les plus en panne.")
        _type_top10 = fdf.groupby("machine").size().nlargest(10).index
        _type_data = (
            fdf[fdf["machine"].isin(_type_top10)]
            .groupby(["machine", "type"]).size()
            .reset_index(name="count")
        )
        _order_t = _type_data.groupby("machine")["count"].sum().sort_values().index.tolist()
        _type_data["machine"] = pd.Categorical(_type_data["machine"], categories=_order_t, ordered=True)
        fig_dur = px.bar(
            _type_data.sort_values("machine"),
            x="count",
            y="machine",
            orientation="h",
            color="type",
            color_discrete_map=TYPE_COLORS,
            barmode="stack",
            labels={"count": "Pannes", "machine": "Machine", "type": "Type"},
        )
        fig_dur.update_layout(showlegend=True, margin=dict(t=10, b=10))
        st.plotly_chart(fig_dur, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE MACHINES & PRODUITS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏭 Analyse Machines":
    st.title("🏭 Analyse Machines & Produits")

    tab_mach, tab_prod = st.tabs(["⚙️ Par type de machine", "📍 Par localisation"])

    # ── ONGLET MACHINES ─────────────────────────────────────────────────────
    with tab_mach:
        top_n = st.slider("Nombre de types de machines à afficher", 5, 30, 15)
        top_mach_data = (
            fdf.groupby(["machine", "equipement"])
            .agg(nb_pannes=("machine", "count"), duree_totale=("duree_min", "sum"))
            .reset_index()
            .rename(columns={"equipement": "Ligne"})
        )
        top_names = (
            top_mach_data.groupby("machine")["nb_pannes"].sum()
            .nlargest(top_n).index
        )
        top_mach_data = top_mach_data[top_mach_data["machine"].isin(top_names)]
        order_tm = top_mach_data.groupby("machine")["nb_pannes"].sum().sort_values().index.tolist()
        top_mach_data["machine"] = pd.Categorical(top_mach_data["machine"], categories=order_tm, ordered=True)
        fig_tm = px.bar(
            top_mach_data.sort_values("machine"),
            x="nb_pannes", y="machine", orientation="h",
            color="Ligne",
            color_discrete_map=EQUIP_COLORS,
            barmode="stack",
            labels={"nb_pannes": "Pannes", "machine": "Machine", "Ligne": "Ligne"},
        )
        fig_tm.update_layout(showlegend=False, height=max(350, top_n * 25), margin=dict(t=10))
        st.plotly_chart(fig_tm, width='stretch')

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Répartition EM / BUN par machine (Top 10)",
                help="Pour chacun des 10 types de machines les plus en panne, répartition des pannes entre les deux lignes EM et BUN.")
            top10m = fdf.groupby("machine").size().nlargest(10).index
            pivot_em = (
                fdf[fdf["machine"].isin(top10m)]
                .groupby(["machine", "equipement"]).size()
                .reset_index(name="count")
                .rename(columns={"equipement": "Ligne"})
            )
            fig_pm = px.bar(
                pivot_em, x="machine", y="count", color="Ligne",
                color_discrete_map=EQUIP_COLORS,
                labels={"machine": "Machine", "count": "Pannes"},
                barmode="group",
            )
            fig_pm.update_layout(showlegend=False, xaxis_tickangle=-30, margin=dict(b=120))
            st.plotly_chart(fig_pm, width='stretch')

        with c2:
            st.subheader("Évolution annuelle EM vs BUN — Top 3 machines",
                help="Nombre de pannes par année pour les 3 machines les plus touchées, séparé par ligne (EM = bleu, BUN = rouge).")
            # Historique complet (pas de filtre date) — EM/BUN + machine uniquement
            _df_ann = df_base[
                df_base["equipement"].isin(selected_equip) &
                df_base["machine"].isin(selected_machines if selected_machines else machines_dispo)
            ]
            top3m = _df_ann.groupby("machine").size().nlargest(3).index
            ann_m = (
                _df_ann[_df_ann["machine"].isin(top3m)]
                .groupby(["annee", "machine", "equipement"]).size()
                .reset_index(name="count")
                .rename(columns={"equipement": "Ligne"})
            )
            ann_m["série"] = ann_m["machine"] + " (" + ann_m["Ligne"] + ")"
            fig_am = px.line(
                ann_m, x="annee", y="count", color="Ligne",
                line_dash="machine",
                color_discrete_map=EQUIP_COLORS,
                markers=True,
                labels={"annee": "Année", "count": "Pannes", "machine": "Machine"},
            )
            st.plotly_chart(fig_am, width='stretch')

        st.markdown("---")
        st.subheader("🔍 Détail par type de machine")
        sel_mach = st.selectbox("Choisir une machine", sorted(fdf["machine"].unique()))
        mach_df = fdf[fdf["machine"] == sel_mach].sort_values("date", ascending=False)
        c1, c2, c3 = st.columns(3)
        c1.metric("Pannes totales", len(mach_df))
        c2.metric("Durée totale (min)", f"{mach_df['duree_min'].sum():.0f}")
        c3.metric("Produits concernés", mach_df["produit"].nunique())
        st.dataframe(
            mach_df[["date", "produit", "equipement", "type", "poste", "technicien", "desc"]]
            .rename(columns={"date": "Date", "produit": "Localisation", "equipement": "Équipement",
                             "type": "Type", "poste": "Poste", "technicien": "Technicien", "desc": "Description"})
            .reset_index(drop=True),
            width='stretch', height=300,
        )

    # ── ONGLET PRODUITS ────────────────────────────────────────────────────
    with tab_prod:
        top_n2 = st.slider("Nombre de localisations à afficher", 5, 50, 20, key="top_n2")
        top_prods = (
            fdf.groupby(["produit", "equipement"])
            .agg(nb_pannes=("produit", "count"), duree_totale=("duree_min", "sum"))
            .reset_index()
            .rename(columns={"equipement": "Ligne"})
        )
        top_prod_names = (
            top_prods.groupby("produit")["nb_pannes"].sum()
            .nlargest(top_n2).index
        )
        top_prods = top_prods[top_prods["produit"].isin(top_prod_names)]
        order_tp = top_prods.groupby("produit")["nb_pannes"].sum().sort_values().index.tolist()
        top_prods["produit"] = pd.Categorical(top_prods["produit"], categories=order_tp, ordered=True)
        fig_tp = px.bar(
            top_prods.sort_values("produit"),
            x="nb_pannes", y="produit", orientation="h",
            color="Ligne",
            color_discrete_map=EQUIP_COLORS,
            barmode="stack",
            labels={"nb_pannes": "Pannes", "produit": "Localisation"},
        )
        fig_tp.update_layout(showlegend=False, height=max(350, top_n2 * 22), margin=dict(t=10))
        st.plotly_chart(fig_tp, width='stretch')

        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Machines les plus fréquentes par localisation (Top 10)",
                help="Pour les 10 localisations les plus en panne, détail des types de machines impliquées.")
            top10p = fdf.groupby("produit").size().nlargest(10).index
            pivot_mp = (
                fdf[fdf["produit"].isin(top10p)]
                .groupby(["produit", "machine", "equipement"]).size()
                .reset_index(name="count")
                .rename(columns={"equipement": "Ligne"})
            )
            # Colorier par ligne EM/BUN, distinguer machines par opacité
            fig_mp = px.bar(
                pivot_mp, x="produit", y="count", color="Ligne",
                pattern_shape="machine",
                color_discrete_map=EQUIP_COLORS,
                labels={"produit": "Produit", "count": "Pannes", "machine": "Machine"},
                barmode="stack",
            )
            fig_mp.update_layout(showlegend=False, xaxis_tickangle=-30, margin=dict(b=140))
            st.plotly_chart(fig_mp, width='stretch')
        with c2:
            st.subheader("Top 10 localisations par type de panne",
                help="Répartition MECHANICAL / ELECTRICAL / OTHER pour les 10 localisations les plus en panne.")
            _top10_lp = fdf.groupby("produit").size().nlargest(10).index
            _type_lp = (
                fdf[fdf["produit"].isin(_top10_lp)]
                .groupby(["produit", "type"]).size()
                .reset_index(name="count")
            )
            fig_dp = px.bar(
                _type_lp, x="produit", y="count",
                color="type",
                color_discrete_map=TYPE_COLORS,
                barmode="stack",
                labels={"count": "Pannes", "produit": "Localisation", "type": "Type"},
            )
            fig_dp.update_layout(showlegend=True, xaxis_tickangle=-30, margin=dict(b=140, t=10))
            st.plotly_chart(fig_dp, width='stretch')

        st.markdown("---")
        st.subheader("🔍 Détail par localisation")
        sel_prod = st.selectbox("Choisir une localisation", sorted(fdf["produit"].unique()))
        prod_df = fdf[fdf["produit"] == sel_prod].sort_values("date", ascending=False)
        c1, c2, c3 = st.columns(3)
        c1.metric("Pannes totales", len(prod_df))
        c2.metric("Type majoritaire", prod_df["type"].mode().iloc[0] if len(prod_df) > 0 else "-")
        c3.metric("Postes concernés", prod_df["poste"].nunique())
        st.dataframe(
            prod_df[["date", "machine", "equipement", "type", "poste", "technicien", "desc"]]
            .rename(columns={"date": "Date", "machine": "Catégorie machine", "equipement": "Équipement",
                             "type": "Type", "poste": "Poste", "technicien": "Technicien", "desc": "Description"})
            .reset_index(drop=True),
            width='stretch', height=300,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CLASSIFICATION AUTOMATIQUE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Classification automatique":
    st.title("🤖 Classification automatique des causes")
    st.info(
        "Un modèle **TF-IDF + Régression Logistique** est entraîné sur les descriptions "
        "des pannes pour prédire automatiquement le type (MECHANICAL / ELECTRICAL / OTHER)."
    )

    with st.spinner("Entraînement du modèle en cours…"):
        pipeline, report, cm, acc, labels = build_classifier(df_base)

    st.success(f"✅ Modèle entraîné — Précision globale : **{acc:.1%}**")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Matrice de confusion",
            help="Croise les types réels (lignes) et prédits (colonnes) par le modèle. La diagonale = bonnes prédictions. Hors diagonale = erreurs de classification.")
        fig_cm = px.imshow(
            cm,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            labels={"x": "Prédit", "y": "Réel", "color": "Pannes"},
            aspect="auto",
        )
        fig_cm.update_layout(margin=dict(t=30))
        st.plotly_chart(fig_cm, width='stretch')

    with c2:
        st.subheader("Rapport de classification",
            help="Précision : taux de vrais positifs parmi les prédictions. Rappel : taux de vrais positifs parmi les réels. F1-score : moyenne harmonique des deux. Support : nombre d'exemples de test par classe.")
        rows = []
        for label in labels:
            r = report.get(label, {})
            rows.append(
                {
                    "Classe": label,
                    "Précision": f"{r.get('precision', 0):.2%}",
                    "Rappel": f"{r.get('recall', 0):.2%}",
                    "F1-score": f"{r.get('f1-score', 0):.2%}",
                    "Support": int(r.get("support", 0)),
                }
            )
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

        st.markdown("---")
        # Importance des mots par classe
        st.subheader("Mots les plus discriminants",
            help="Mots et bigrammes ayant le poids le plus fort dans la décision du modèle pour chaque classe. Un score élevé = ce mot est fortement associé à ce type de panne.")
        tfidf = pipeline.named_steps["tfidf"]
        clf = pipeline.named_steps["clf"]
        all_feature_names = tfidf.get_feature_names_out()
        # Garder uniquement les features de type « mot » (préfixe "word__")
        word_mask = [n.startswith("word__") for n in all_feature_names]
        feature_names = [n[7:] for n in all_feature_names if n.startswith("word__")]  # strip "word__"

        tab_labels = st.tabs(labels)
        for i, (tab, lbl) in enumerate(zip(tab_labels, labels)):
            with tab:
                if hasattr(clf, "coef_"):
                    coef = clf.coef_[i][word_mask]
                    top_idx = coef.argsort()[-15:][::-1]
                    words_df = pd.DataFrame(
                        {"Mot / Bigramme": [feature_names[j] for j in top_idx], "Score": coef[top_idx]}
                    )
                    fig_words = px.bar(
                        words_df.sort_values("Score"),
                        x="Score",
                        y="Mot / Bigramme",
                        orientation="h",
                        color="Score",
                        color_continuous_scale="Oranges",
                    )
                    fig_words.update_layout(showlegend=False, margin=dict(t=10))
                    st.plotly_chart(fig_words, width='stretch')

    st.markdown("---")
    st.subheader("🔬 Tester la classification en direct")
    _col_tdesc, _col_tmach = st.columns([3, 1])
    with _col_tdesc:
        user_text = st.text_area(
            "Entrez une description de panne :",
            placeholder="Ex: REMPLACEMENT MOTEUR CONVOYEUR / COURT-CIRCUIT ARMOIRE ELECTRIQUE…",
            height=80,
        )
    with _col_tmach:
        user_machine = st.selectbox(
            "Type de machine",
            options=sorted(df["machine"].unique()),
            key="test_machine_sel",
        )
    if st.button("Classifier", type="primary") and user_text.strip():
        _input = user_text.upper() + " " + user_machine.replace(" ", "_")
        probs = pipeline.predict_proba([_input])[0]
        pred_label = labels[probs.argmax()]

        st.markdown(f"### Résultat : **{pred_label}**")
        prob_df = pd.DataFrame({"Type": labels, "Probabilité (%)": (probs * 100).round(1)})
        fig_prob = px.bar(
            prob_df,
            x="Type",
            y="Probabilité (%)",
            color="Type",
            color_discrete_map=TYPE_COLORS,
            text_auto=True,
        )
        fig_prob.update_layout(showlegend=False, yaxis_range=[0, 100])
        st.plotly_chart(fig_prob, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PRÉVISION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Prévision des pannes":
    st.title("📈 Prévision des pannes par machine")

    try:
        import numpy as np

        # ── Sélection de la machine et paramètres
        col_sel, col_hor = st.columns([3, 1])
        with col_sel:
            # Historique complet filtré par EM/BUN et machine (exclusions déjà appliquées dans df_base)
            _df_prev = df_base[
                df_base["equipement"].isin(selected_equip) &
                df_base["machine"].isin(selected_machines if selected_machines else machines_dispo)
            ]
            me_counts = (
                _df_prev.groupby(["produit", "equipement"]).size().reset_index(name="n")
            )
            # Compter les mois distincts pour garantir 12 mois d'historique
            me_months = (
                _df_prev.groupby(["produit", "equipement"])["annee_mois"]
                .nunique()
                .reset_index(name="n_mois")
            )
            me_counts = me_counts.merge(me_months, on=["produit", "equipement"])
            me_counts = me_counts[me_counts["n_mois"] >= 12].sort_values("n", ascending=False)
            me_counts["label"] = me_counts["produit"] + " (" + me_counts["equipement"] + ")"
            label_to_pair = dict(zip(
                me_counts["label"],
                zip(me_counts["produit"], me_counts["equipement"])
            ))
            labels_dispo = me_counts["label"].tolist()
            if not labels_dispo:
                st.warning("Aucune machine n'a suffisamment d'historique (≥ 12 mois) avec les filtres actuels.")
                st.stop()
            sel_machines = st.multiselect(
                "Machines à prévoir (top pannes par défaut)",
                options=labels_dispo,
                default=labels_dispo[:5],
                max_selections=10,
            )
        with col_hor:
            horizon = st.slider("Horizon (mois)", 3, 24, 12)

        if not sel_machines:
            st.info("Sélectionnez au moins une machine.")
            st.stop()

        # ── Séries sur l'historique complet filtré par EM/BUN + machine (sans date)
        all_months = sorted(_df_prev["annee_mois"].unique())

        sel_produit_names = list({label_to_pair[l][0] for l in sel_machines})
        sel_equips = list({label_to_pair[l][1] for l in sel_machines})
        monthly_machine = (
            _df_prev[
                _df_prev["produit"].isin(sel_produit_names) &
                _df_prev["equipement"].isin(sel_equips)
            ]
            .groupby(["annee_mois", "produit", "equipement"])
            .size()
            .reset_index(name="count")
        )

        fig = go.Figure()
        forecast_rows = []
        palette = px.colors.qualitative.Plotly

        def _prevision_saisonniere(series: np.ndarray, horizon: int, alpha: float = 0.3) -> np.ndarray:
            """
            Tendance + saisonnalité mensuelle (12 mois).
            - Tendance : lissage exponentiel pondéré (EWMA) sur les valeurs récentes
            - Saisonnalité : ratio moyen de chaque mois calendaire vs moyenne globale
            - Résidu ajusté par la tendance récente
            """
            n = len(series)

            # ── Tendance : régression linéaire pondérée (récent > ancien)
            t = np.arange(n)
            # Facteur normalisé : dernier mois pèse ×1000 le premier, quelle que soit la longueur
            w = np.exp(np.log(1000) * t / max(n - 1, 1))
            w /= w.sum()
            t_mean = np.average(t, weights=w)
            s_mean = np.average(series, weights=w)
            slope = np.sum(w * (t - t_mean) * (series - s_mean)) / np.sum(w * (t - t_mean) ** 2)
            intercept = s_mean - slope * t_mean
            trend = intercept + slope * t

            # ── Saisonnalité : décomposition sur 12 mois
            detrended = series - trend
            seasonal = np.zeros(12)
            counts_s = np.zeros(12)
            for i, v in enumerate(detrended):
                m = i % 12
                seasonal[m] += v
                counts_s[m] += 1
            seasonal = seasonal / np.maximum(counts_s, 1)
            # Centrer la saisonnalité
            seasonal -= seasonal.mean()

            # ── Prévision
            future = np.zeros(horizon)
            last_t = n - 1
            for j in range(horizon):
                t_fut = last_t + j + 1
                m_fut = (n + j) % 12
                future[j] = (intercept + slope * t_fut) + seasonal[m_fut]

            return np.clip(np.round(future), 0, None)

        for i, label in enumerate(sel_machines):
            produit_val, equip = label_to_pair[label]
            # Remplir les mois manquants avec 0
            sub = (
                pd.DataFrame({"annee_mois": all_months})
                .merge(
                    monthly_machine[
                        (monthly_machine["produit"] == produit_val) &
                        (monthly_machine["equipement"] == equip)
                    ][["annee_mois", "count"]],
                    on="annee_mois", how="left"
                )
                .fillna(0)
                .sort_values("annee_mois")
                .reset_index(drop=True)
            )

            series = sub["count"].values.astype(float)
            if len(series) < 12:
                continue

            future_pred = _prevision_saisonniere(series, horizon)

            # Génération des labels de mois futurs (robuste pandas 3.x)
            last_ts = pd.Period(sub["annee_mois"].iloc[-1], freq="M").to_timestamp()
            future_periods = [
                (last_ts + pd.DateOffset(months=j + 1)).strftime("%Y-%m")
                for j in range(horizon)
            ]

            color = palette[i % len(palette)]
            short_name = label[:35]

            # Prévision avec zone d'incertitude (±15 %) — calculé avant le trait de liaison
            upper_vals = [float(v) for v in (future_pred * 1.15).round()]
            lower_vals = [float(v) for v in (future_pred * 0.85).round()]
            pred_vals  = [float(v) for v in future_pred]

            # Filtrer l'affichage historique sur la période sélectionnée
            sub_display = sub[
                (sub["annee_mois"] >= date_debut) &
                (sub["annee_mois"] <= date_fin)
            ]

            # Historique (fenêtre filtrée)
            fig.add_trace(go.Scatter(
                x=sub_display["annee_mois"],
                y=sub_display["count"],
                mode="lines",
                name=f"{short_name}",
                legendgroup=label,
                line=dict(color=color, width=2),
            ))

            # Trait de liaison réel → prévision (pointillé discret)
            if len(sub_display) > 0:
                link_x = [sub_display["annee_mois"].iloc[-1], future_periods[0]]
                link_y = [float(sub_display["count"].iloc[-1]), pred_vals[0]]
                fig.add_trace(go.Scatter(
                    x=link_x,
                    y=link_y,
                    mode="lines",
                    line=dict(color=color, dash="dot", width=1),
                    showlegend=False,
                    legendgroup=label,
                    hoverinfo="skip",
                ))

            fig.add_trace(go.Scatter(
                x=future_periods + future_periods[::-1],
                y=upper_vals + lower_vals[::-1],
                fill="toself",
                fillcolor=color,
                opacity=0.12,
                line=dict(width=0),
                showlegend=False,
                legendgroup=label,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=future_periods,
                y=pred_vals,
                mode="lines+markers",
                name=f"{short_name} (prév.)",
                legendgroup=label,
                line=dict(color=color, dash="dot", width=2),
                marker=dict(symbol="circle-open", size=6),
            ))

            for period, val in zip(future_periods, pred_vals):
                forecast_rows.append({"Mois": period, "Machine": label, "Pannes prévues": int(val)})

        # Ligne de séparation : dernier mois de la période affichée (= date_fin)
        sep_x = date_fin
        y_max = max(
            (sub["count"].max() for sub in [
                pd.DataFrame({"annee_mois": all_months}).merge(
                    monthly_machine[
                        (monthly_machine["produit"] == label_to_pair[lbl][0]) &
                        (monthly_machine["equipement"] == label_to_pair[lbl][1])
                    ][["annee_mois","count"]],
                    on="annee_mois", how="left"
                ).fillna(0)
                for lbl in sel_machines
            ]),
            default=10
        )
        fig.add_trace(go.Scatter(
            x=[sep_x, sep_x],
            y=[0, float(y_max) * 1.1],
            mode="lines",
            line=dict(color="white", dash="dash", width=1),
            opacity=0.4,
            showlegend=False,
            hoverinfo="skip",
            name="_sep",
        ))

        fig.update_layout(
            xaxis_title="Mois",
            yaxis_title="Nombre de pannes",
            legend_title="Machine",
            hovermode="x unified",
            height=520,
        )
        st.plotly_chart(fig, width='stretch')

        if forecast_rows:
            st.subheader("Tableau des prévisions",
                help="Nombre de pannes prévues par mois et par machine sur l'horizon sélectionné. Calcul basé sur une régression linéaire pondérée (mois récents = plus de poids).")
            fc_df = (
                pd.DataFrame(forecast_rows)
                .pivot(index="Mois", columns="Machine", values="Pannes prévues")
            )
            fc_df["TOTAL"] = fc_df.sum(axis=1)
            st.dataframe(fc_df.style.format("{:.0f}"), width='stretch')

            # Machine la plus à risque sur l'horizon
            risk = (
                pd.DataFrame(forecast_rows)
                .groupby("Machine")["Pannes prévues"]
                .sum()
                .sort_values(ascending=False)
            )
            st.subheader("🚨 Machines les plus à risque sur l'horizon",
                help="Cumul des pannes prévues sur tout l'horizon par machine. Permet de prioriser les actions de maintenance préventive.")
            fig_risk = px.bar(
                risk.reset_index(),
                x="Machine",
                y="Pannes prévues",
                color="Pannes prévues",
                color_continuous_scale="Reds",
                text_auto=True,
            )
            fig_risk.update_layout(showlegend=False, xaxis_tickangle=-30)
            st.plotly_chart(fig_risk, width='stretch')

            # ── Top 5 des prévisions ──────────────────────────────────────────
            st.subheader("🏆 Top 5 — Machines les plus critiques sur l'horizon",
                help="Classement des 5 machines avec le plus grand nombre de pannes prévues cumulées sur l'horizon sélectionné.")
            top5 = risk.head(5).reset_index()
            top5.index = top5.index + 1  # numérotation 1 → 5
            top5.columns = ["Machine", "Pannes prévues (total)"]
            medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
            cols_top5 = st.columns(min(len(top5), 5))
            for col_t, (_, row), medal in zip(cols_top5, top5.iterrows(), medals):
                col_t.metric(
                    label=f"{medal} {row['Machine']}",
                    value=f"{int(row['Pannes prévues (total)'])} pannes",
                )

        # ── Section MTBF ─────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔧 Risque MTBF — Probabilité de panne imminente",
            help="Basé sur le temps moyen entre pannes (MTBF) et le temps écoulé depuis la dernière panne. "
                 "Modèle exponentiel : P(panne) = 1 − e^(−t/MTBF). "
                 "Plus le temps écoulé dépasse le MTBF, plus la probabilité est élevée.")

        today_ts = pd.Timestamp.today().normalize()
        mtbf_rows = []

        for lbl in sel_machines:
            produit_val, equip = label_to_pair[lbl]
            sub_hist = _df_prev[
                (_df_prev["produit"] == produit_val) &
                (_df_prev["equipement"] == equip) &
                (_df_prev["date_dt"] <= _fin_ts)
            ].sort_values("date_dt")

            if len(sub_hist) < 2:
                continue

            dates_sorted = sub_hist["date_dt"].drop_duplicates().sort_values()
            intervals_days = dates_sorted.diff().dropna().dt.days
            if len(intervals_days) == 0 or intervals_days.mean() == 0:
                continue

            mtbf_days = float(intervals_days.mean())
            last_failure = dates_sorted.iloc[-1]
            days_since = int((today_ts - last_failure).days)

            # Probabilité de panne : modèle exponentiel (mémoire sans état)
            # R(t) = exp(-t/MTBF)  →  P(panne) = 1 - R(t)
            failure_prob = (1 - np.exp(-days_since / mtbf_days)) * 100

            if failure_prob < 40:
                risk_label = "🟢 Faible"
            elif failure_prob < 70:
                risk_label = "🟡 Modéré"
            else:
                risk_label = "🔴 Élevé"

            mtbf_rows.append({
                "Machine": lbl,
                "MTBF moyen (j)": round(mtbf_days),
                "Dernière panne": last_failure.date(),
                "Jours écoulés": days_since,
                "Prob. panne (%)": round(failure_prob, 1),
                "Risque": risk_label,
            })

        if mtbf_rows:
            mtbf_df = pd.DataFrame(mtbf_rows).sort_values("Prob. panne (%)", ascending=False)

            fig_mtbf = px.bar(
                mtbf_df,
                x="Machine",
                y="Prob. panne (%)",
                color="Prob. panne (%)",
                color_continuous_scale=["#27ae60", "#f39c12", "#e74c3c"],
                range_color=[0, 100],
                text="Prob. panne (%)",
                labels={"Prob. panne (%)": "Probabilité (%)"},
            )
            fig_mtbf.add_hline(
                y=70, line_dash="dash", line_color="#e74c3c", opacity=0.6,
                annotation_text="Risque élevé (70 %)", annotation_position="top left",
            )
            fig_mtbf.add_hline(
                y=40, line_dash="dash", line_color="#f39c12", opacity=0.6,
                annotation_text="Risque modéré (40 %)", annotation_position="top left",
            )
            fig_mtbf.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_mtbf.update_layout(
                showlegend=False,
                yaxis_range=[0, 115],
                xaxis_tickangle=-30,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_mtbf, width='stretch')

            st.dataframe(
                mtbf_df[["Machine", "MTBF moyen (j)", "Dernière panne", "Jours écoulés", "Prob. panne (%)", "Risque"]],
                width='stretch',
                hide_index=True,
            )
            st.caption(
                "**Lecture** : si les jours écoulés ≈ MTBF, la probabilité est ~63 %. "
                "Si les jours écoulés = 2 × MTBF, elle atteint ~86 %. "
                "Une machine récemment réparée repart toujours à 0 %."
            )

            # ── Détail par machine ────────────────────────────────────────
            st.markdown("##### 🔍 Détail des pannes probables")
            # Pré-sélectionner la machine au risque le plus élevé
            _mach_options = mtbf_df["Machine"].tolist()
            sel_detail = st.selectbox(
                "Choisir une machine pour voir les types et causes les plus probables",
                options=_mach_options,
                index=0,
                key="mtbf_detail_sel",
            )
            _det_machine, _det_equip = label_to_pair[sel_detail]
            _det_hist = _df_prev[
                (_df_prev["machine"] == _det_machine) &
                (_df_prev["equipement"] == _det_equip) &
                (_df_prev["date_dt"] <= _fin_ts)
            ]

            if len(_det_hist) > 0:
                _c1, _c2 = st.columns([1, 2])

                with _c1:
                    st.markdown("**Répartition par type**")
                    _type_counts = (
                        _det_hist["type"].value_counts()
                        .reset_index()
                        .rename(columns={"index": "Type", "type": "Nombre"})
                    )
                    _type_counts.columns = ["Type", "Nombre"]
                    _type_counts["Probabilité (%)"] = (
                        _type_counts["Nombre"] / _type_counts["Nombre"].sum() * 100
                    ).round(1)
                    fig_det_type = px.pie(
                        _type_counts,
                        names="Type",
                        values="Nombre",
                        color="Type",
                        color_discrete_map=TYPE_COLORS,
                        hole=0.4,
                    )
                    fig_det_type.update_traces(textinfo="percent+label")
                    fig_det_type.update_layout(showlegend=False, margin=dict(t=10, b=10))
                    st.plotly_chart(fig_det_type, width='stretch')

                with _c2:
                    st.markdown("**Top 10 causes les plus fréquentes**")
                    _desc_counts = (
                        _det_hist[_det_hist["desc"].str.len() > 3]
                        .groupby(["desc", "type"])
                        .size()
                        .reset_index(name="Occurrences")
                        .sort_values("Occurrences", ascending=False)
                        .head(10)
                        .rename(columns={"desc": "Description", "type": "Type"})
                        .reset_index(drop=True)
                    )
                    st.dataframe(_desc_counts, width='stretch', hide_index=True)

        else:
            st.info("Pas assez d'historique pour calculer le MTBF sur les machines sélectionnées (minimum 2 pannes).")

    except Exception as e:
        st.error(f"Erreur lors de la prévision : {e}")
