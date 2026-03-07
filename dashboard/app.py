"""
dashboard/app.py
----------------
Streamlit operator dashboard.

Run:  streamlit run dashboard/app.py

Tabs
----
  📊 Overview      — live counts + LOST alerts + REAPPEARANCE alerts
  🟢 Active        — currently tracked, grouped by camera
  🔴 Lost          — registry (never deleted), operator actions
  🔍 Search        — by ID / time range / camera
  👤 Person Detail — full event timeline + sighting log + crop gallery
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import time
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from PIL import Image

from database.identity_store import IdentityStore

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Cam ReID  |  Operator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.status-active   { color: #48bb78; font-weight: bold; }
.status-lost     { color: #fc8181; font-weight: bold; }
.status-resolved { color: #90cdf4; font-weight: bold; }
.reappear-box {
    background: #1a4731;
    border-left: 4px solid #48bb78;
    padding: 10px 16px;
    border-radius: 4px;
    margin: 6px 0;
}
.lost-box {
    background: #3d1515;
    border-left: 4px solid #fc8181;
    padding: 10px 16px;
    border-radius: 4px;
    margin: 6px 0;
}
.event-row { padding: 4px 0; border-bottom: 1px solid #2d3748; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

DB_PATH = os.environ.get("REID_DB_PATH", "database/identities.db")

@st.cache_resource
def get_store():
    return IdentityStore(
        db_path=DB_PATH,
        lost_threshold_secs=float(os.environ.get("REID_LOST_THRESHOLD", "120")),
        similarity_threshold=float(os.environ.get("REID_SIM_THRESHOLD", "0.60")),
    )


def fmt_time(ts):
    if not ts: return "—"
    return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")


def time_ago(ts):
    if not ts: return "—"
    d = time.time() - float(ts)
    if d < 60:   return f"{int(d)}s ago"
    if d < 3600: return f"{int(d/60)}m ago"
    if d < 86400:return f"{int(d/3600)}h ago"
    return f"{int(d/86400)}d ago"


def load_crop(path):
    if not path: return None
    p = Path(path)
    if not p.exists(): return None
    try:    return Image.open(p).convert("RGB")
    except: return None


EVENT_ICONS = {
    "first_seen":   "🆕",
    "lost":         "🔴",
    "reappeared":   "🔄",
    "resolved":     "✅",
    "reactivated":  "🔵",
    "note":         "📝",
}

def event_icon(etype): return EVENT_ICONS.get(etype, "•")

def status_badge(s):
    icons = {"active": "🟢", "lost": "🔴", "resolved": "🔵"}
    return f"{icons.get(s,'⚪')} {s.upper()}"


# ── Sidebar ───────────────────────────────────────────────────────────────────

store = get_store()

with st.sidebar:
    st.title("🎯 ReID Operator")
    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)
    if auto_refresh:
        st.markdown('<meta http-equiv="refresh" content="5">', unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔍 Check Lost Persons Now"):
        promoted = store.promote_lost()
        if promoted:
            st.warning(f"Promoted: {', '.join(promoted)}")
        else:
            st.success("No new lost persons.")

    st.markdown("---")
    stats = store.stats()
    st.metric("Active",        stats["active"])
    st.metric("Lost",          stats["lost"])
    st.metric("Resolved",      stats["resolved"])
    st.metric("Reappearances", stats["reappearances"])
    st.caption(f"DB: `{DB_PATH}`")


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_ov, tab_active, tab_lost, tab_search, tab_detail = st.tabs([
    "📊 Overview",
    "🟢 Active",
    "🔴 Lost Registry",
    "🔍 Search",
    "👤 Person Detail",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 · OVERVIEW
# ══════════════════════════════════════════════════════════════════════

with tab_ov:
    store.promote_lost()
    stats = store.stats()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🟢 Active",        stats["active"])
    c2.metric("🔴 Lost",          stats["lost"],
              delta=f"+{stats['lost']}" if stats["lost"] else None,
              delta_color="inverse")
    c3.metric("🔵 Resolved",      stats["resolved"])
    c4.metric("📍 Sightings",     stats["sightings"])
    c5.metric("🔄 Reappearances", stats["reappearances"])

    st.markdown("---")

    # ── Reappearance alerts (last 10 min) ─────────────────────────────
    reappearances = store.get_recent_reappearances(since_seconds=600)
    if reappearances:
        st.subheader(f"🔄 Recent Reappearances  ({len(reappearances)})")
        st.caption("These persons were marked LOST and have now been re-detected — ID automatically restored.")
        for ev in reappearances:
            col_img, col_info = st.columns([1, 6])
            with col_img:
                img = load_crop(ev.get("best_crop_path"))
                if img: st.image(img, width=70)
            with col_info:
                st.markdown(
                    f"<div class='reappear-box'>"
                    f"<b>{ev['global_id']}</b>  was LOST → "
                    f"🔄 <b>Reappeared on Camera {ev['camera_id']}</b>"
                    f"  at {fmt_time(ev['occurred_at'])}"
                    f"  ({time_ago(ev['occurred_at'])})<br>"
                    f"<small>{ev.get('detail','')}</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    st.markdown("---")

    # ── Lost person alerts ────────────────────────────────────────────
    lost = store.get_all(status="lost")
    if lost:
        st.subheader(f"⚠️ Lost Person Alerts  ({len(lost)})")
        for p in lost:
            col_img, col_info, col_act = st.columns([1, 4, 2])
            with col_img:
                img = load_crop(p["best_crop_path"])
                if img: st.image(img, width=70)
                else: st.markdown("🚫")
            with col_info:
                st.markdown(
                    f"<div class='lost-box'>"
                    f"<b>{p['global_id']}</b>  —  "
                    f"Last seen {time_ago(p['last_seen_at'])} "
                    f"on Camera {p['last_camera_id']}<br>"
                    f"<small>First seen: {fmt_time(p['first_seen_at'])}</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with col_act:
                if st.button(f"✅ Resolve", key=f"ov_res_{p['global_id']}"):
                    store.resolve(p["global_id"], "Resolved via overview")
                    st.rerun()
    else:
        st.success("✅ No lost persons.")

    # ── Recent activity ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Recent Activity (last 30 min)")
    recent = store.search_by_time(since=time.time() - 1800)
    recent = [r for r in recent if r]
    if recent:
        df = pd.DataFrame([{
            "ID":         p["global_id"],
            "Status":     p["status"],
            "Last Camera":f"Cam {p['last_camera_id']}",
            "Last Seen":  time_ago(p["last_seen_at"]),
            "First Seen": fmt_time(p["first_seen_at"]),
        } for p in recent])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No activity in the last 30 minutes.")


# ══════════════════════════════════════════════════════════════════════
# TAB 2 · ACTIVE
# ══════════════════════════════════════════════════════════════════════

with tab_active:
    st.header("🟢 Active Persons")
    active = store.get_all(status="active")

    if not active:
        st.info("No active persons.")
    else:
        by_cam = {}
        for p in active:
            by_cam.setdefault(p["last_camera_id"], []).append(p)

        for cam_id in sorted(by_cam.keys()):
            persons = by_cam[cam_id]
            st.subheader(f"Camera {cam_id}  —  {len(persons)} persons")
            cols = st.columns(min(len(persons), 5))
            for i, p in enumerate(persons):
                with cols[i % 5]:
                    img = load_crop(p["best_crop_path"])
                    if img: st.image(img, caption=p["global_id"], width=100)
                    else:   st.markdown(f"**{p['global_id']}**")
                    st.caption(time_ago(p["last_seen_at"]))

        st.markdown("---")
        st.dataframe(pd.DataFrame([{
            "ID":         p["global_id"],
            "Last Camera":f"Cam {p['last_camera_id']}",
            "Last Seen":  time_ago(p["last_seen_at"]),
            "First Seen": fmt_time(p["first_seen_at"]),
        } for p in active]), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 · LOST REGISTRY
# ══════════════════════════════════════════════════════════════════════

with tab_lost:
    st.header("🔴 Lost Persons Registry")
    st.info(
        "Lost persons are **never automatically deleted**. "
        "They stay here until you resolve the case — even if the person "
        "reappears (in which case they move back to Active automatically "
        "and appear in Reappearance alerts on the Overview tab)."
    )

    lost = store.get_all(status="lost")
    if not lost:
        st.success("✅ Registry is empty.")
    else:
        st.warning(f"⚠️  {len(lost)} person(s) missing")

        for p in lost:
            events = store.get_events(p["global_id"])
            with st.expander(
                f"🔴  {p['global_id']}  —  missing since "
                f"{fmt_time(p['last_seen_at'])}  ({time_ago(p['last_seen_at'])})",
                expanded=True,
            ):
                col_img, col_info = st.columns([1, 3])
                with col_img:
                    img = load_crop(p["best_crop_path"])
                    if img: st.image(img, width=130, caption="Last known appearance")
                    else:   st.markdown("🚫 No image")

                with col_info:
                    st.markdown(f"**ID:** `{p['global_id']}`")
                    st.markdown(f"**First seen:** {fmt_time(p['first_seen_at'])}")
                    st.markdown(
                        f"**Last seen:** {fmt_time(p['last_seen_at'])}"
                        f"  ({time_ago(p['last_seen_at'])})"
                        f"  on Camera {p['last_camera_id']}"
                    )
                    sightings = store.get_sightings(p["global_id"])
                    cams = sorted(set(s["camera_id"] for s in sightings))
                    st.markdown(f"**Cameras visited:** {', '.join(f'Cam {c}' for c in cams)}")
                    st.markdown(f"**Total sightings:** {len(sightings)}")
                    if p.get("notes"):
                        st.info(f"📝 {p['notes']}")

                # Event timeline
                if events:
                    st.markdown("**Event Timeline:**")
                    for ev in events:
                        icon = event_icon(ev["event_type"])
                        cam_str = f"  Cam {ev['camera_id']}" if ev["camera_id"] is not None else ""
                        st.markdown(
                            f"`{fmt_time(ev['occurred_at'])}`  "
                            f"{icon} **{ev['event_type'].upper()}**"
                            f"{cam_str}  — {ev['detail']}",
                        )

                # Operator actions
                st.markdown("---")
                a1, a2, a3 = st.columns(3)
                with a1:
                    note = st.text_input("Add note", key=f"note_{p['global_id']}")
                    if st.button("💬 Save Note", key=f"savenote_{p['global_id']}"):
                        if note:
                            store.add_note(p["global_id"], note)
                            st.success("Note saved.")
                            st.rerun()
                with a2:
                    if st.button("✅ Resolve (Found)", key=f"res_{p['global_id']}", type="primary"):
                        store.resolve(p["global_id"], note or "Resolved via dashboard")
                        st.success(f"{p['global_id']} resolved.")
                        st.rerun()
                with a3:
                    if st.button("🔄 Reactivate", key=f"react_{p['global_id']}"):
                        store.reactivate(p["global_id"])
                        st.info(f"{p['global_id']} reactivated.")
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════
# TAB 4 · SEARCH
# ══════════════════════════════════════════════════════════════════════

with tab_search:
    st.header("🔍 Search")

    mode = st.radio("Search by", ["Global ID", "Time Range", "Camera"], horizontal=True)
    results = []

    if mode == "Global ID":
        gid = st.text_input("Global ID (e.g. GID-0001)")
        if gid:
            p = store.get_person(gid.strip().upper())
            results = [p] if p else []
            if not p: st.warning("Not found.")

    elif mode == "Time Range":
        c1, c2 = st.columns(2)
        with c1:
            d_from = st.date_input("From", (datetime.now()-timedelta(hours=1)).date())
            t_from = st.time_input("Time from")
        with c2:
            d_to = st.date_input("To", datetime.now().date())
            t_to = st.time_input("Time to", datetime.now().time())
        status_f = st.selectbox("Status", ["all","active","lost","resolved"])
        if st.button("Search"):
            since = datetime.combine(d_from, t_from).timestamp()
            until = datetime.combine(d_to,   t_to  ).timestamp()
            found = store.search_by_time(since, until)
            if status_f != "all":
                found = [p for p in found if p and p["status"] == status_f]
            results = [p for p in found if p]

    elif mode == "Camera":
        cam  = st.number_input("Camera ID", min_value=0, value=0, step=1)
        hrs  = st.slider("Look back (hours)", 1, 72, 1)
        if st.button("Search Camera"):
            results = [
                r for r in store.search_by_time(
                    since=time.time() - hrs*3600,
                    camera_id=int(cam),
                )
                if r
            ]

    if results:
        st.success(f"{len(results)} result(s)")
        for p in results:
            ci, ii = st.columns([1, 5])
            with ci:
                img = load_crop(p.get("best_crop_path"))
                if img: st.image(img, width=70)
            with ii:
                cams = sorted(set(
                    s["camera_id"] for s in p.get("sightings", [])
                )) or [p["last_camera_id"]]
                st.markdown(
                    f"**{p['global_id']}** {status_badge(p['status'])}"
                    f"  |  Cameras: {', '.join(f'Cam {c}' for c in cams)}"
                    f"  |  Last seen: {time_ago(p['last_seen_at'])}"
                )
        st.dataframe(pd.DataFrame([{
            "ID":         p["global_id"],
            "Status":     p["status"],
            "Last Camera":f"Cam {p['last_camera_id']}",
            "Last Seen":  fmt_time(p["last_seen_at"]),
            "First Seen": fmt_time(p["first_seen_at"]),
        } for p in results]), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 5 · PERSON DETAIL
# ══════════════════════════════════════════════════════════════════════

with tab_detail:
    st.header("👤 Person Detail")

    all_persons = store.get_all()
    if not all_persons:
        st.info("No persons in DB yet. Run the tracker first.")
    else:
        selected = st.selectbox(
            "Select person",
            [p["global_id"] for p in all_persons],
            format_func=lambda g: f"{g}  ({next((p['status'] for p in all_persons if p['global_id']==g), '')})"
        )

        if selected:
            p = store.get_person(selected)
            if p:
                # Header
                ci, mi = st.columns([1, 3])
                with ci:
                    img = load_crop(p.get("best_crop_path"))
                    if img: st.image(img, width=160)
                    else:   st.markdown("🚫 No crop")
                with mi:
                    st.markdown(f"## {p['global_id']}")
                    st.markdown(f"**Status:** {status_badge(p['status'])}")
                    st.markdown(f"**First seen:** {fmt_time(p['first_seen_at'])}")
                    st.markdown(
                        f"**Last seen:** {fmt_time(p['last_seen_at'])}"
                        f"  ({time_ago(p['last_seen_at'])})"
                        f"  on Camera {p['last_camera_id']}"
                    )
                    if p.get("resolved_at"):
                        st.markdown(f"**Resolved:** {fmt_time(p['resolved_at'])}")
                    if p.get("notes"):
                        st.info(f"📝 {p['notes']}")

                st.markdown("---")

                # ── Full Event Timeline ────────────────────────────────
                events = p.get("events", [])
                if events:
                    st.subheader("📋 Full Event Timeline")
                    st.caption("Complete lifecycle: first detection → lost → reappearances → resolution")
                    for ev in events:
                        icon = event_icon(ev["event_type"])
                        cam_str = f"  •  Camera {ev['camera_id']}" if ev["camera_id"] is not None else ""
                        colour = {
                            "first_seen":  "#48bb78",
                            "lost":        "#fc8181",
                            "reappeared":  "#68d391",
                            "resolved":    "#90cdf4",
                            "reactivated": "#63b3ed",
                            "note":        "#ecc94b",
                        }.get(ev["event_type"], "#a0aec0")
                        st.markdown(
                            f"<div style='padding:6px 0; border-bottom:1px solid #2d3748'>"
                            f"<span style='color:gray;font-size:0.8em'>{fmt_time(ev['occurred_at'])}</span>"
                            f"&nbsp;&nbsp;"
                            f"{icon} <span style='color:{colour};font-weight:bold'>"
                            f"{ev['event_type'].upper()}</span>"
                            f"<span style='color:gray'>{cam_str}</span>"
                            f"&nbsp;&nbsp;{ev['detail']}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                st.markdown("---")

                # ── Camera Journey ─────────────────────────────────────
                sightings = p.get("sightings", [])
                if sightings:
                    st.subheader("📍 Camera Journey")
                    cam_data = {}
                    for s in sightings:
                        c = s["camera_id"]
                        cam_data.setdefault(c, []).append(s["seen_at"])
                    st.dataframe(pd.DataFrame([{
                        "Camera":      f"Cam {c}",
                        "First Visit": fmt_time(min(ts)),
                        "Last Visit":  fmt_time(max(ts)),
                        "Sightings":   len(ts),
                    } for c, ts in sorted(cam_data.items())]),
                    use_container_width=True)

                    # Crop gallery
                    st.subheader("🖼 Appearance Gallery")
                    crop_paths = [s["crop_path"] for s in sightings if s.get("crop_path")][:12]
                    if crop_paths:
                        gcols = st.columns(6)
                        for ci2, cp in enumerate(crop_paths):
                            img = load_crop(cp)
                            if img:
                                sv = next((s for s in sightings if s.get("crop_path")==cp), None)
                                cap = f"Cam {sv['camera_id']}" if sv else ""
                                with gcols[ci2 % 6]:
                                    st.image(img, width=90, caption=cap)

                    # Sighting log
                    st.subheader("Full Sighting Log")
                    st.dataframe(pd.DataFrame([{
                        "Camera": f"Cam {s['camera_id']}",
                        "Frame":  s["frame_idx"],
                        "Time":   fmt_time(s["seen_at"]),
                        "Conf":   f"{s['conf']:.2f}" if s["conf"] else "—",
                    } for s in sightings]), use_container_width=True, height=280)

                # ── Operator actions ───────────────────────────────────
                st.markdown("---")
                st.subheader("Operator Actions")
                a1, a2, a3 = st.columns(3)
                with a1:
                    note_in = st.text_area("Add note", key="det_note")
                    if st.button("💬 Save", key="det_save"):
                        if note_in:
                            store.add_note(selected, note_in)
                            st.success("Saved.")
                            st.rerun()
                with a2:
                    if p["status"] == "lost":
                        if st.button("✅ Resolve", type="primary", key="det_resolve"):
                            store.resolve(selected, note_in or "Resolved via detail view")
                            st.success("Resolved.")
                            st.rerun()
                    elif p["status"] == "resolved":
                        if st.button("🔄 Reactivate", key="det_react"):
                            store.reactivate(selected)
                            st.info("Reactivated.")
                            st.rerun()
                with a3:
                    st.markdown(f"### {status_badge(p['status'])}")
