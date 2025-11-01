
import io
from datetime import date, datetime, timedelta
from typing import List, Set, Tuple

import pandas as pd
import streamlit as st

from scheduler_fifo import SchedulerFIFO

import json


WEEKDAY_LABELS = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']


PHASES_FROM_OFFSETS = lambda offsets: ["new"] + [f"rev{n}" for n in sorted({o for o in offsets if o != 0})]


# --- CONFIG IMPORT STATE (one-shot) ---
if "cfg_import_pending" not in st.session_state:
    st.session_state.cfg_import_pending = False
if "cfg_bytes" not in st.session_state:
    st.session_state.cfg_bytes = None






def dumps_skips_after(skips_after: dict[str, date]) -> str:
    """Représentation texte (pour copier/coller)."""
    lines = [f"{eid} ; {d.isoformat()}" for eid, d in sorted(skips_after.items(), key=lambda kv: (kv[1], kv[0]))]
    return "\n".join(lines)

def parse_skips_after(text: str) -> dict[str, date]:
    out = {}
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = [p.strip() for p in ln.split(";")]
        if len(parts) != 2:
            continue
        eid, dstr = parts
        # normalise " + revX" -> "(revX)"
        if " + rev" in eid and " (rev" not in eid:
            eid = eid.replace(" + rev", " (rev")
            if not eid.endswith(")"):
                eid += ")"
        try:
            dval = datetime.strptime(dstr, "%Y-%m-%d").date()
        except Exception:
            continue
        out[eid] = dval
    return out




def config_to_json(*, start_date, offsets, caps, blackouts_text, skips_after, subject_ids):
    cfg = {
        "start_date": start_date.isoformat(),
        "offsets": list(offsets),
        "caps": list(caps) if isinstance(caps, (list, tuple)) else caps,
        "blackouts_text": blackouts_text or "",
        "skips_after": {eid: d.isoformat() for eid, d in (skips_after or {}).items()},
        "subjects": list(subject_ids or []),
        "review_order": "short-first",
    }
    import json
    return json.dumps(cfg, ensure_ascii=False, indent=2)


def json_to_config(payload: str) -> dict:
    """Parse une config JSON et retourne un dict python brut (dates converties)."""
    raw = json.loads(payload)
    cfg = {}
    cfg["start_date"] = datetime.strptime(raw["start_date"], "%Y-%m-%d").date()
    cfg["offsets"] = list(raw.get("offsets", [0,3,7,30]))
    caps = raw.get("caps", 4)
    cfg["caps"] = tuple(caps) if isinstance(caps, (list,tuple)) else int(caps)
    cfg["blackouts_text"] = raw.get("blackouts_text","")
    skips_raw = raw.get("skips_after", {})
    cfg["skips_after"] = {eid: datetime.strptime(d, "%Y-%m-%d").date() for eid, d in skips_raw.items()}
    cfg["subjects"] = list(raw.get("subjects", []))
    cfg["review_order"] = raw.get("review_order","short-first")
    return cfg









def parse_offsets(text: str) -> Tuple[int, ...]:
    try:
        vals = sorted({int(x) for x in text.replace(',', ' ').split()})
        if 0 not in vals:
            vals = (0,) + tuple(v for v in vals if v != 0)
        return tuple(vals)
    except Exception:
        return (0, 3, 7, 30)

def parse_blackouts(text: str, start: date, horizon_days: int = 900) -> Set[date]:
    # Accept single dates YYYY-MM-DD or ranges 'YYYY-MM-DD..YYYY-MM-DD'
    out: Set[date] = set()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith('#')]
    for ln in lines:
        if '..' in ln:
            a, b = [x.strip() for x in ln.split('..', 1)]
            try:
                da = datetime.strptime(a, '%Y-%m-%d').date()
                db = datetime.strptime(b, '%Y-%m-%d').date()
                if db < da:
                    da, db = db, da
                cur = da
                while cur <= db:
                    out.add(cur)
                    cur += timedelta(days=1)
            except Exception:
                continue
        else:
            try:
                d = datetime.strptime(ln, '%Y-%m-%d').date()
                out.add(d)
            except Exception:
                continue
    end = start + timedelta(days=horizon_days)
    return {d for d in out if start <= d <= end}


def load_subjects(uploaded_file) -> List[str]:
    try:
        df = pd.read_csv(uploaded_file)
        if 'subject_id' in df.columns:
            return df['subject_id'].astype(str).tolist()
        return df.iloc[:, 0].astype(str).tolist()
    except Exception:
        return []

def subjects_table(subject_ids: List[str]) -> pd.DataFrame:
    return pd.DataFrame({'subject_id': subject_ids})

def plan_to_dataframe(plan) -> pd.DataFrame:
    rows = []
    for day in plan:
        for oc in day.items:
            rows.append({'Date': day.date, 'Subject': oc.sid, 'Phase': oc.phase})

    df = pd.DataFrame(rows).sort_values(['Date', 'Phase', 'Subject']).reset_index(drop=True)
    return df

def make_week_grid(df_sched: pd.DataFrame, week_start: date, max_lines_per_cell: int = 12) -> pd.DataFrame:
    days = [week_start + timedelta(days=i) for i in range(7)]
    cols = [f"{WEEKDAY_LABELS[i]} {d.strftime('%d/%m')}" for i, d in enumerate(days)]
    cells = {c: [] for c in cols}
    per_day = {d: [] for d in days}
    for _, row in df_sched.iterrows():
        d = row['Date']
        if d in per_day:
            per_day[d].append(f"{row['Subject']} ({row['Phase']})")
    max_rows = min(max((len(per_day[d]) for d in days), default=0), max_lines_per_cell)
    for r in range(max_rows):
        for i, d in enumerate(days):
            items = per_day[d]
            cells[cols[i]].append(items[r] if r < len(items) else '')
    return pd.DataFrame(cells)

def month_bounds(year: int, month: int) -> Tuple[date, date]:
    first = date(year, month, 1)
    start = first - timedelta(days=(first.weekday() % 7))
    if month == 12:
        last = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    end = last + timedelta(days=(6 - last.weekday()))
    return start, end

def make_month_grid(df_sched: pd.DataFrame, year: int, month: int, max_lines_per_cell: int = 6) -> pd.DataFrame:
    start, end = month_bounds(year, month)
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)
    rows = []
    for i in range(0, len(days), 7):
        week = days[i:i+7]
        row = {}
        for j, d in enumerate(week):
            label = f"{WEEKDAY_LABELS[j]} {d.strftime('%d/%m')}"
            items = df_sched.loc[df_sched['Date'] == d, :]
            lines = [f"{r['Subject']} ({r['Phase']})" for _, r in items.iterrows()]
            if len(lines) > max_lines_per_cell:
                more = len(lines) - max_lines_per_cell
                lines = lines[:max_lines_per_cell] + [f"+{more}…"]
            row[label] = "\n".join(lines)
        rows.append(row)
    return pd.DataFrame(rows)

def export_csv(df_sched: pd.DataFrame) -> str:
    buf = io.StringIO()
    df = df_sched.copy()
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    df.to_csv(buf, index=False)
    return buf.getvalue()

def export_ics(plan, calendar_name='Revisions', all_day=True) -> str:
    buf = []
    buf.append('BEGIN:VCALENDAR')
    buf.append('VERSION:2.0')
    buf.append(f'X-WR-CALNAME:{calendar_name}')
    buf.append('PRODID:-//medicocalendar//scheduler//EN')
    for day in plan:
        for oc in day.items:
            uid = f"{day.date.isoformat()}_{oc.sid}_{oc.phase}@medicocalendar"
            title = f"{oc.sid} ({oc.phase})"
            buf.append('BEGIN:VEVENT')
            if all_day:
                dt = day.date.strftime("%Y%m%d")
                buf.append(f"DTSTART;VALUE=DATE:{dt}")
                buf.append(f"DTEND;VALUE=DATE:{(day.date + timedelta(days=1)).strftime('%Y%m%d')}")
            else:
                dt = day.date.strftime('%Y%m%d')
                buf.append(f'DTSTART;VALUE=DATE:{dt}')
                buf.append(f'DTEND;VALUE=DATE:{dt}')
            buf.append(f'SUMMARY:{title}')
            buf.append(f'UID:{uid}')
            buf.append('END:VEVENT')
    buf.append('END:VCALENDAR')
    return "\r\n".join(buf)

st.set_page_config(page_title='Calendrier de révisions (J0/J+N)', layout='wide')
st.title('🗓️ Calendrier de révisions — J0/J+N (journée entière)')
st.caption("Planificateur déterministe : révisions d'abord, puis nouvelles matières jusqu'au cap. Politique Ancrage pour les skips.")

with st.sidebar:



    # --- STATE INIT (global, une seule fois) ---
    if "skips_after" not in st.session_state:
        st.session_state.skips_after = {}  # dict: "<sid> (revX)" -> date
    if "blackouts_text" not in st.session_state:
        st.session_state.blackouts_text = ""



    st.header('Paramètres')
    start_date = st.date_input('Date de départ', value=date.today())
    offsets_text = st.text_input('Offsets (jours, ex: 0 3 7 30)', value='0 3 7 30')
    offsets = parse_offsets(offsets_text)

    cap_mode = st.radio('Capacité', ['Globale', 'Par jour de semaine'], horizontal=True)
    if cap_mode == 'Globale':
        cap_global = st.number_input('Cap/jour (tous jours)', min_value=0, max_value=20, value=4, step=1)
        caps = (cap_global,)*7
    else:
        c_lun = st.number_input('Lun', 0, 20, 4); c_mar = st.number_input('Mar', 0, 20, 4)
        c_mer = st.number_input('Mer', 0, 20, 4); c_jeu = st.number_input('Jeu', 0, 20, 4)
        c_ven = st.number_input('Ven', 0, 20, 4); c_sam = st.number_input('Sam', 0, 20, 2)
        c_dim = st.number_input('Dim', 0, 20, 0)
        caps = (c_lun, c_mar, c_mer, c_jeu, c_ven, c_sam, c_dim)


    st.markdown('**Jours off / vacances**')
    blackouts_text = st.text_area('Dates ou plages (YYYY-MM-DD ou YYYY-MM-DD..YYYY-MM-DD), une par ligne', height=120, placeholder='2025-12-24..2026-01-02\n2026-02-15')




    # --- Skips (avec date) : SAISIE MANUELLE ---
    st.markdown("### Skips (avec date)")

    skip_eid = st.text_input(
        "ID du bloc (copie depuis le calendrier, format: <subject_id> (revX) ou (new))",
        placeholder="Ex: UE4/Cardio/ICaigue (rev3)",
        key="skip_eid_text",
    )
    skip_date = st.date_input("Replanifier **pas avant**", value=date.today(), key="skip_date")

    def _normalize_eid(eid: str) -> str:
        eid = (eid or "").strip()
        # autoriser l'ancien format " + revX"
        if " + rev" in eid and " (rev" not in eid:
            eid = eid.replace(" + rev", " (rev")
            if not eid.endswith(")"):
                eid = eid + ")"
        return eid

    col_add, col_del, col_clear = st.columns([1,1,1])
    with col_add:
        if st.button("➕ Ajouter / Mettre à jour", key="btn_skip_add"):
            eid = _normalize_eid(skip_eid)
            if " (" not in eid or not eid.endswith(")"):
                st.warning("Format attendu : <subject_id> (revX)  ex: UE4/Cardio/ICaigue (rev3)")
            else:
                st.session_state.skips_after[eid] = skip_date
                st.success(f"Skip ajouté: {eid} → {skip_date.isoformat()}")
                st.rerun()



    with col_del:
        if st.button("🗑️ Supprimer cet ID", key="btn_skip_del"):
            eid = _normalize_eid(skip_eid)
            st.session_state.skips_after.pop(eid, None)
            st.rerun()



    with col_clear:
        if st.button("🧹 Vider tous les skips", key="btn_skip_clear"):
            st.session_state.skips_after = {}
            st.session_state.skips_after.clear()
            st.rerun()

    # Import multi-lignes (accepte "(revX)" et l'ancien " + revX")
    with st.expander("Importer des skips (texte)"):
        txt = st.text_area(
            "Format: <subject_id> (revX) ; YYYY-MM-DD  — un par ligne",
            height=120,
            placeholder="UE4/Cardio/ICaigue (rev3) ; 2025-11-03\nUE6/Pharmaco/ATB (new) ; 2025-11-10",
            key="skip_import_area",
        )
        if st.button("Importer ces skips", key="btn_import_skips"):
            st.session_state.skips_after.update(parse_skips_after(txt))
            st.success("Skips importés.")
            st.rerun()













    st.markdown('---')
    st.subheader('Sujets (CSV)')
    uploaded = st.file_uploader('Importer un CSV (colonnes: subject_id[,title])', type=['csv'], accept_multiple_files=False)
    if uploaded is None:
        st.info('Aucun CSV importé. Utilise la liste d\'exemple (subjects_50.csv).')
        default_subjects = pd.read_csv('subjects_50.csv')['subject_id'].tolist()
        subject_ids = default_subjects
    else:
        subject_ids = load_subjects(uploaded) or pd.read_csv('subjects_50.csv')['subject_id'].tolist()












    # Liste des skips actuels
    current_skips = st.session_state.get("skips_after", {})
    if current_skips:
        st.caption("Skips enregistrés :")
        for eid, dval in sorted(current_skips.items(), key=lambda kv: (kv[1], kv[0])):
            col1, col2 = st.columns([5,1])
            with col1:
                st.write(f"- **{eid}** → {dval.isoformat()}")
            with col2:
                if st.button("🗑️", key=f"del_{eid}"):
                    st.session_state.skips_after.pop(eid, None)
                    st.rerun()
    else:
        st.info("Aucun skip défini.")






    st.markdown("### Configuration (sauvegarde/chargement)")

    # --- Export JSON  ---
    cfg_json = config_to_json(
        start_date=start_date,
        offsets=offsets,
        caps=caps,
        blackouts_text=st.session_state.get("blackouts_text", blackouts_text),
        skips_after=st.session_state.get("skips_after", {}),
        subject_ids=subject_ids,
    )
    st.download_button(
    "💾 Exporter la configuration",
    data=cfg_json,
    file_name="revisions_config.json",
    mime="application/json",
    key="btn_export_cfg",
    )

    # --- Import JSON (anti-boucle) ---
    cfg_file = st.file_uploader("📂 Importer une configuration (.json)", type=["json"], key="cfg_upl")

    # Étape 1 : l'utilisateur choisit un fichier puis clique sur "Charger"
    if cfg_file is not None and not st.session_state.cfg_import_pending and st.session_state.cfg_bytes is None:
        st.info("Fichier prêt. Cliquez « Charger cette configuration » pour l’appliquer.")
        if st.button("Charger cette configuration", key="btn_cfg_load"):
            st.session_state.cfg_bytes = cfg_file.read()
            st.session_state.cfg_import_pending = True
            st.rerun()

    # Étape 2 : au rerun, on consomme le flag et on hydrate l'état UNE SEULE FOIS
    if st.session_state.cfg_import_pending:
        try:
            payload = (st.session_state.cfg_bytes or b"").decode("utf-8")
            cfg = json_to_config(payload)

            # Hydrate l’état (skips/jours off bruts)
            st.session_state.skips_after    = cfg["skips_after"]            # {"UE4/... (revX)": date}
            st.session_state.blackouts_text = cfg["blackouts_text"]

            # Paramètres principaux
            start_date = cfg["start_date"]
            offsets    = tuple(cfg["offsets"])
            caps       = cfg["caps"] if isinstance(cfg["caps"], int) else tuple(cfg["caps"])
            subjects   = cfg["subjects"] or st.session_state.get("subject_ids", [])

            # Répercuter dans l'état central (utilisé par l’export et le moteur)
            st.session_state["start_date"]  = start_date
            st.session_state["offsets"]     = list(offsets)
            st.session_state["caps"]        = caps
            st.session_state["subject_ids"] = subjects

            st.success("Configuration chargée.")
        except Exception as e:
            st.error(f"Échec du chargement : {e}")
        finally:
            # Consommer le flag et vider le buffer pour éviter toute boucle
            st.session_state.cfg_import_pending = False
            st.session_state.cfg_bytes = None
            # IMPORTANT: ne PAS appeler st.rerun() ici (on vient déjà d’en faire un)

















blackouts = parse_blackouts(blackouts_text, start_date)

skips_after = st.session_state.skips_after  # dict {"<sid> + <phase>": date}
sched = SchedulerFIFO(
    subjects=subject_ids,
    offsets=offsets,
    cap=caps,
    start_date=start_date,
    blackouts=parse_blackouts(st.session_state.blackouts_text, start_date),
    skips_after=st.session_state.skips_after,   # <<< ICI
)
plan = sched.plan()
df_sched = plan_to_dataframe(plan)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('Sujets', len(subject_ids))
with col2:
    st.metric('Jour de départ', start_date.strftime('%Y-%m-%d'))
with col3:
    st.metric('Dernier jour planifié', df_sched['Date'].max().strftime('%Y-%m-%d'))
with col4:
    cap_mean = sum(caps) / 7
    st.metric('Cap moyen', f'{cap_mean:.2f}')





def month_bounds(year: int, month: int):
    """Retourne (start, end) : du lundi avant/du 1er au dimanche après/du dernier jour du mois."""
    first = date(year, month, 1)
    start = first - timedelta(days=(first.weekday() % 7))  # lundi de la semaine du 1er
    last  = (date(year+1,1,1) - timedelta(days=1)) if month == 12 else (date(year, month+1, 1) - timedelta(days=1))
    end   = last + timedelta(days=(6 - last.weekday()))    # dimanche de la dernière semaine affichée
    return start, end

def day_list(df_sched: pd.DataFrame, d: date, max_lines: int = 6, sort_reviews_first: bool = True) -> str:
    """Retourne une liste à puces pour le jour d, éventuellement triée rev→new, coupée avec +x si trop longue."""
    items = df_sched[df_sched["Date"] == d]
    if sort_reviews_first:
        items = items.assign(_ord=items["Phase"].map(lambda p: 0 if p != "new" else 1)) \
                     .sort_values(["_ord", "Subject"]).drop(columns="_ord")
    lines = [f"- {r.Subject} ({r.Phase})" for _, r in items.iterrows()]
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"… +{len(items) - max_lines}"]
    return "\n".join(lines) if lines else "_—_"
# ====================================================



# NAV
# --- INIT STATE (unique) ---
if "day_picker" not in st.session_state:
    st.session_state.day_picker = start_date
if "week_picker" not in st.session_state:
    st.session_state.week_picker = start_date - timedelta(days=start_date.weekday())  # lundi
if "month_picker" not in st.session_state:
    st.session_state.month_picker = start_date.replace(day=1)


view = st.radio("Vue", ["Jour", "Semaine", "Mois"], horizontal=True)

def nav_buttons(left_label, right_label):
    colL, colM, colR = st.columns([1,4,1])
    with colL:
        return st.button(left_label), colM, st.button(right_label)



if view == "Jour":
    # NAV
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("← Jour -1", key="btn_prev_day"):
            st.session_state.day_picker -= timedelta(days=1)
            st.rerun()
    with c2:
        if st.button("Jour +1 →", key="btn_next_day"):
            st.session_state.day_picker += timedelta(days=1)
            st.rerun()

    # Sélecteur (source de vérité = session_state)
    dsel = st.date_input("Jour", key="day_picker")

    # Données du jour (tri: révisions d’abord, puis new)
    day_df = df_sched[df_sched["Date"] == dsel].copy()
    if day_df.empty:
        st.info("Aucun bloc ce jour.")
    else:
        day_df = day_df.assign(_ord=day_df["Phase"].map(lambda p: 0 if p != "new" else 1)) \
                       .sort_values(["_ord", "Subject"]).drop(columns="_ord")
        st.write(f"**{dsel.strftime('%A %d %B %Y')}** — {len(day_df)} blocs")
        # rendu lisible sous forme de puces
        bullets = "\n".join([f"- {r.Subject} ({r.Phase})" for _, r in day_df.iterrows()])
        st.markdown(bullets)
        # (option) table compacte :
        # st.dataframe(day_df[["Subject","Phase"]], width="stretch")



elif view == "Semaine":
    # NAV
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("← Semaine -1", key="btn_prev_week"):
            st.session_state.week_picker -= timedelta(days=7)
            st.rerun()
    with c2:
        if st.button("Semaine +1 →", key="btn_next_week"):
            st.session_state.week_picker += timedelta(days=7)
            st.rerun()

    # Sélecteur (clé = source de vérité). On force lundi pour cohérence.
    monday = st.session_state.week_picker - timedelta(days=st.session_state.week_picker.weekday())
    st.session_state.week_picker = monday
    wsel = st.date_input("Semaine (lundi)", key="week_picker")
    week_start = st.session_state.week_picker  # déjà lundi

    st.subheader(f"Semaine du {week_start.strftime('%d/%m/%Y')}")
    cols = st.columns(7)
    for i in range(7):
        d = week_start + timedelta(days=i)
        with cols[i]:
            st.markdown(f"**{WEEKDAY_LABELS[i]} {d.strftime('%d/%m')}**")
            # utilise le helper global day_list(df_sched, d, ...)
            st.markdown(day_list(df_sched, d, max_lines=20))





elif view == "Mois":
    # init état
    if "month_picker" not in st.session_state:
        st.session_state.month_picker = start_date.replace(day=1)

    # NAV
    colL, colR = st.columns([1, 1])
    with colL:
        if st.button("← Mois -1", key="btn_prev_month"):
            cur = st.session_state.month_picker
            prev_month = (cur.replace(day=1) - timedelta(days=1)).replace(day=1)
            st.session_state.month_picker = prev_month
            st.rerun()
    with colR:
        if st.button("Mois +1 →", key="btn_next_month"):
            cur = st.session_state.month_picker
            next_month = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
            st.session_state.month_picker = next_month
            st.rerun()

    # sélecteur (source de vérité = session_state)
    msel = st.date_input("Mois (choisir une date du mois)", key="month_picker")
    month_ref = st.session_state.month_picker.replace(day=1)

    # grille du mois
    start, end = month_bounds(month_ref.year, month_ref.month)
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)

    st.subheader(f"Mois de {month_ref.strftime('%B %Y')}")
    for i in range(0, len(days), 7):
        cols = st.columns(7)
        for j, d in enumerate(days[i:i+7]):
            in_month = (d.month == month_ref.month)
            title = f"**{WEEKDAY_LABELS[j]} {d.day:02d}/{d.month:02d}**"
            with cols[j]:
                if not in_month:
                    st.markdown(f"<div style='opacity:0.5'>{title}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(title)
                st.markdown(day_list(df_sched, d, max_lines=6))












st.markdown('---')

csv_text = export_csv(df_sched)
st.download_button('⬇️ Télécharger CSV', csv_text, file_name='revisions.csv', mime='text/csv')

ics_text = export_ics(plan, calendar_name='Révisions – Médecine', all_day=True)
st.download_button('📅 Télécharger ICS', ics_text, file_name='revisions.ics', mime='text/calendar')

with st.expander('Aperçu des sujets (IDs)'):
    st.dataframe(subjects_table(subject_ids).head(50), width='stretch')
