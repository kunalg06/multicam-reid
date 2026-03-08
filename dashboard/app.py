import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import json
import pandas as pd
import streamlit as st
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
/* Stop Streamlit's own spinner from flickering on fragment reruns */
div[data-testid="stSpinner"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── Shared helpers ────────────────────────────────────────────────────────────

DB_PATH = os.environ.get("REID_DB_PATH", "database/identities.db")


@st.cache_resource
def get_store() -> IdentityStore:
    """Single shared store instance — created once, lives for the session."""
    return IdentityStore(
        db_path=DB_PATH,
        lost_threshold_secs=float(os.environ.get("REID_LOST_THRESHOLD", "120")),
        similarity_threshold=float(os.environ.get("REID_SIM_THRESHOLD", "0.60")),
    )


def fmt_time(ts) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")


def time_ago(ts) -> str:
    if not ts:
        return "—"
    d = time.time() - float(ts)
    if d < 60:    return f"{int(d)}s ago"
    if d < 3600:  return f"{int(d / 60)}m ago"
    if d < 86400: return f"{int(d / 3600)}h ago"
    return f"{int(d / 86400)}d ago"


def load_crop(path) -> "Image.Image | None":
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return Image.open(p).convert("RGB")
    except Exception:
        return None


EVENT_ICONS = {
    "first_seen":  "🆕",
    "lost":        "🔴",
    "reappeared":  "🔄",
    "resolved":    "✅",
    "reactivated": "🔵",
    "note":        "📝",
}
EVENT_COLOURS = {
    "first_seen":  "#48bb78",
    "lost":        "#fc8181",
    "reappeared":  "#68d391",
    "resolved":    "#90cdf4",
    "reactivated": "#63b3ed",
    "note":        "#ecc94b",
}


def status_badge(s: str) -> str:
    return {"active": "🟢", "lost": "🔴", "resolved": "🔵"}.get(s, "⚪") + f" {s.upper()}"


# ── Sidebar ───────────────────────────────────────────────────────────────────

store = get_store()

with st.sidebar:
    st.title("🎯 ReID Operator")
    st.markdown("---")

    # ── Manual refresh (fragment only, never a page reload) ───────────
    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        # This sets a session-state flag that the live fragment checks
        st.session_state["_refresh_requested"] = True

    st.caption("Refreshes only the stats & alerts — your inputs stay intact.")
    st.markdown("---")

    # ── Lost-check trigger ────────────────────────────────────────────
    if st.button("🔍 Check Lost Persons Now", use_container_width=True):
        promoted = store.promote_lost()
        if promoted:
            st.warning(f"Promoted: {', '.join(promoted)}")
        else:
            st.success("No new lost persons.")

    st.markdown("---")

    # ── Static sidebar stats (only updates on manual refresh) ─────────
    stats = store.stats()
    st.metric("🟢 Active",        stats["active"])
    st.metric("🔴 Lost",          stats["lost"])
    st.metric("🔵 Resolved",      stats["resolved"])
    st.metric("🔄 Reappearances", stats["reappearances"])
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
# OVERVIEW TAB
# Live stats section is a fragment — reruns independently.
# Operator alert buttons are INSIDE the fragment but trigger store
# actions then call st.rerun() which only reruns the fragment itself.
# ══════════════════════════════════════════════════════════════════════

with tab_ov:

    @st.fragment
    def overview_live():
        """
        This function is a Streamlit fragment.
        When st.rerun() is called inside here, ONLY this fragment reruns.
        The rest of the page (Lost tab inputs, Search tab, etc.) is untouched.
        """
        store.promote_lost()
        stats = store.stats()

        # ── Metrics row ───────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("🟢 Active",        stats["active"])
        c2.metric("🔴 Lost",          stats["lost"],
                  delta=f"+{stats['lost']}" if stats["lost"] else None,
                  delta_color="inverse")
        c3.metric("🔵 Resolved",      stats["resolved"])
        c4.metric("📍 Sightings",     stats["sightings"])
        c5.metric("🔄 Reappearances", stats["reappearances"])

        # ── Reappearance alerts ───────────────────────────────────────
        reappearances = store.get_recent_reappearances(since_seconds=600)
        if reappearances:
            st.markdown("---")
            st.subheader(f"🔄 Recent Reappearances  ({len(reappearances)})")
            st.caption("These persons were LOST and have been re-detected — ID automatically restored.")
            for ev in reappearances:
                col_img, col_info = st.columns([1, 7])
                with col_img:
                    img = load_crop(ev.get("best_crop_path"))
                    if img:
                        st.image(img, width=65)
                with col_info:
                    st.markdown(
                        f"<div class='reappear-box'>"
                        f"<b>{ev['global_id']}</b> was LOST → "
                        f"🔄 <b>Reappeared on Camera {ev['camera_id']}</b>"
                        f" at {fmt_time(ev['occurred_at'])}"
                        f" ({time_ago(ev['occurred_at'])})<br>"
                        f"<small>{ev.get('detail', '')}</small>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # ── Lost alerts with quick-resolve buttons ────────────────────
        lost = store.get_all(status="lost")
        if lost:
            st.markdown("---")
            st.subheader(f"⚠️ Lost Person Alerts  ({len(lost)})")
            for p in lost:
                col_img, col_info, col_act = st.columns([1, 5, 2])
                with col_img:
                    img = load_crop(p["best_crop_path"])
                    if img:
                        st.image(img, width=65)
                    else:
                        st.markdown("🚫")
                with col_info:
                    st.markdown(
                        f"<div class='lost-box'>"
                        f"<b>{p['global_id']}</b> — "
                        f"Last seen {time_ago(p['last_seen_at'])} "
                        f"on Camera {p['last_camera_id']}<br>"
                        f"<small>First seen: {fmt_time(p['first_seen_at'])}</small>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with col_act:
                    # Button inside fragment → only fragment reruns after click
                    if st.button("✅ Resolve", key=f"ov_res_{p['global_id']}"):
                        store.resolve(p["global_id"], "Resolved via overview")
                        st.rerun()   # reruns fragment only, not whole page ✅
        else:
            st.markdown("---")
            st.success("✅ No lost persons.")

        # ── Recent activity table ─────────────────────────────────────
        recent = [r for r in store.search_by_time(since=time.time() - 1800) if r]
        if recent:
            st.markdown("---")
            st.subheader("Recent Activity (last 30 min)")
            st.dataframe(pd.DataFrame([{
                "ID":          p["global_id"],
                "Status":      p["status"],
                "Last Camera": f"Cam {p['last_camera_id']}",
                "Last Seen":   time_ago(p["last_seen_at"]),
                "First Seen":  fmt_time(p["first_seen_at"]),
            } for p in recent]), use_container_width=True)

        # ── Handle manual refresh button from sidebar ─────────────────
        # If sidebar button was pressed, rerun this fragment once then clear flag
        if st.session_state.get("_refresh_requested"):
            st.session_state["_refresh_requested"] = False
            st.rerun()   # fragment rerun only

    overview_live()


# ══════════════════════════════════════════════════════════════════════
# ACTIVE TAB
# Simple read — no fragment needed, no refresh interaction required.
# ══════════════════════════════════════════════════════════════════════

with tab_active:
    st.header("🟢 Active Persons")

    if st.button("🔄 Refresh Active", key="refresh_active"):
        pass   # just reruns this tab's render on next script run

    active = store.get_all(status="active")

    if not active:
        st.info("No active persons being tracked.")
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
                    if img:
                        st.image(img, caption=p["global_id"], width=100)
                    else:
                        st.markdown(f"**{p['global_id']}**\n🚫")
                    st.caption(time_ago(p["last_seen_at"]))

        st.markdown("---")
        st.dataframe(pd.DataFrame([{
            "ID":          p["global_id"],
            "Last Camera": f"Cam {p['last_camera_id']}",
            "Last Seen":   time_ago(p["last_seen_at"]),
            "First Seen":  fmt_time(p["first_seen_at"]),
        } for p in active]), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# LOST REGISTRY TAB
# All operator inputs (text fields, buttons) are OUTSIDE any fragment.
# They will never be wiped by a data refresh.
# ══════════════════════════════════════════════════════════════════════

with tab_lost:
    st.header("🔴 Lost Persons Registry")
    st.info(
        "Lost persons are **never automatically deleted**. "
        "They remain here until you explicitly resolve the case. "
        "If they reappear on any camera, their ID is automatically restored "
        "and a Reappearance alert appears on the Overview tab."
    )

    if st.button("🔄 Refresh List", key="refresh_lost"):
        pass   # triggers a script rerun which re-reads lost list

    lost = store.get_all(status="lost")

    if not lost:
        st.success("✅ Lost persons registry is empty.")
    else:
        st.warning(f"⚠️  {len(lost)} person(s) currently LOST")

        for p in lost:
            events = store.get_events(p["global_id"])

            with st.expander(
                f"🔴  {p['global_id']}  —  missing since "
                f"{fmt_time(p['last_seen_at'])}  ({time_ago(p['last_seen_at'])})",
                expanded=False,   # collapsed by default so list is readable
            ):
                col_img, col_info = st.columns([1, 3])
                with col_img:
                    img = load_crop(p["best_crop_path"])
                    if img:
                        st.image(img, width=130, caption="Last known appearance")
                    else:
                        st.markdown("🚫 No image")

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
                    st.markdown(
                        f"**Cameras visited:** {', '.join(f'Cam {c}' for c in cams)}"
                    )
                    st.markdown(f"**Total sightings:** {len(sightings)}")
                    if p.get("notes"):
                        st.info(f"📝 {p['notes']}")

                # Event timeline
                if events:
                    st.markdown("**Event Timeline:**")
                    for ev in events:
                        icon   = EVENT_ICONS.get(ev["event_type"], "•")
                        colour = EVENT_COLOURS.get(ev["event_type"], "#a0aec0")
                        cam_s  = (
                            f"  •  Camera {ev['camera_id']}"
                            if ev["camera_id"] is not None else ""
                        )
                        st.markdown(
                            f"<div style='padding:5px 0;"
                            f"border-bottom:1px solid #2d3748'>"
                            f"<span style='color:gray;font-size:0.8em'>"
                            f"{fmt_time(ev['occurred_at'])}</span>&nbsp;&nbsp;"
                            f"{icon} <span style='color:{colour};"
                            f"font-weight:bold'>{ev['event_type'].upper()}</span>"
                            f"<span style='color:gray'>{cam_s}</span>"
                            f"&nbsp;&nbsp;{ev['detail']}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                st.markdown("---")

                # ── Operator action inputs ─────────────────────────────
                # These use unique keys per person so they are stable
                # across reruns and are never wiped by data refreshes.
                note_key = f"lost_note_{p['global_id']}"
                note_text = st.text_input(
                    "Add note",
                    key=note_key,
                    placeholder="e.g. Confirmed found at Gate 3",
                )

                act1, act2, act3 = st.columns(3)

                with act1:
                    if st.button(
                        "💬 Save Note",
                        key=f"save_note_{p['global_id']}",
                        disabled=not note_text,
                    ):
                        store.add_note(p["global_id"], note_text)
                        # Clear the input after save using session state
                        st.session_state[note_key] = ""
                        st.success("Note saved.")

                with act2:
                    if st.button(
                        "✅ Resolve (Found)",
                        key=f"resolve_{p['global_id']}",
                        type="primary",
                    ):
                        store.resolve(
                            p["global_id"],
                            note_text or "Resolved via dashboard",
                        )
                        st.success(f"✅ {p['global_id']} resolved.")
                        st.rerun()   # refresh this tab's list

                with act3:
                    if st.button(
                        "🔄 Reactivate",
                        key=f"reactivate_{p['global_id']}",
                        help="Use if this was marked lost by mistake",
                    ):
                        store.reactivate(p["global_id"])
                        st.info(f"{p['global_id']} reactivated.")
                        st.rerun()

                # Sighting log
                if sightings:
                    st.markdown("**Recent sightings:**")
                    st.dataframe(
                        pd.DataFrame([{
                            "Camera": f"Cam {s['camera_id']}",
                            "Frame":  s["frame_idx"],
                            "Time":   fmt_time(s["seen_at"]),
                            "Conf":   f"{s['conf']:.2f}" if s["conf"] else "—",
                        } for s in sightings[:20]]),
                        use_container_width=True,
                        height=200,
                    )


# ══════════════════════════════════════════════════════════════════════
# SEARCH TAB
# ══════════════════════════════════════════════════════════════════════

with tab_search:
    st.header("🔍 Search")

    mode = st.radio(
        "Search by", ["Global ID", "Time Range", "Camera"],
        horizontal=True,
    )
    results = []

    if mode == "Global ID":
        gid_input = st.text_input("Global ID (e.g. GID-0001)")
        if gid_input:
            p = store.get_person(gid_input.strip().upper())
            results = [p] if p else []
            if not p:
                st.warning("No person found.")

    elif mode == "Time Range":
        c1, c2 = st.columns(2)
        with c1:
            d_from = st.date_input("From", (datetime.now() - timedelta(hours=1)).date())
            t_from = st.time_input("Time from")
        with c2:
            d_to = st.date_input("To", datetime.now().date())
            t_to = st.time_input("Time to", datetime.now().time())
        status_filter = st.selectbox("Status filter", ["all", "active", "lost", "resolved"])
        if st.button("Search", key="search_time"):
            since = datetime.combine(d_from, t_from).timestamp()
            until = datetime.combine(d_to,   t_to).timestamp()
            found = store.search_by_time(since, until)
            if status_filter != "all":
                found = [p for p in found if p and p["status"] == status_filter]
            results = [p for p in found if p]

    elif mode == "Camera":
        cam_id = st.number_input("Camera ID", min_value=0, value=0, step=1)
        hrs    = st.slider("Look back (hours)", 1, 72, 1)
        if st.button("Search Camera", key="search_cam"):
            results = [
                r for r in store.search_by_time(
                    since=time.time() - hrs * 3600,
                    camera_id=int(cam_id),
                )
                if r
            ]

    if results:
        st.success(f"{len(results)} result(s)")
        for p in results:
            col_i, col_t = st.columns([1, 5])
            with col_i:
                img = load_crop(p.get("best_crop_path"))
                if img:
                    st.image(img, width=70)
            with col_t:
                cams = sorted(set(
                    s["camera_id"] for s in p.get("sightings", [])
                )) or [p["last_camera_id"]]
                st.markdown(
                    f"**{p['global_id']}** {status_badge(p['status'])}"
                    f"  |  Cameras: {', '.join(f'Cam {c}' for c in cams)}"
                    f"  |  Last seen: {time_ago(p['last_seen_at'])}"
                )
        st.dataframe(pd.DataFrame([{
            "ID":          p["global_id"],
            "Status":      p["status"],
            "Last Camera": f"Cam {p['last_camera_id']}",
            "Last Seen":   fmt_time(p["last_seen_at"]),
            "First Seen":  fmt_time(p["first_seen_at"]),
        } for p in results]), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# PERSON DETAIL TAB
# ══════════════════════════════════════════════════════════════════════

with tab_detail:
    st.header("👤 Person Detail")

    all_persons = store.get_all()
    if not all_persons:
        st.info("No persons in database yet. Run the tracker first.")
    else:
        selected = st.selectbox(
            "Select person",
            [p["global_id"] for p in all_persons],
            format_func=lambda g: (
                f"{g}  "
                f"({next((p['status'] for p in all_persons if p['global_id'] == g), '')})"
            ),
        )

        if selected:
            p = store.get_person(selected)
            if p:
                # ── Header ────────────────────────────────────────────
                col_i, col_m = st.columns([1, 3])
                with col_i:
                    img = load_crop(p.get("best_crop_path"))
                    if img:
                        st.image(img, width=160)
                    else:
                        st.markdown("🚫 No crop")
                with col_m:
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

                # ── Event timeline ────────────────────────────────────
                events = p.get("events", [])
                if events:
                    st.subheader("📋 Full Event Timeline")
                    st.caption(
                        "Complete lifecycle: first detection → lost → "
                        "reappearances → operator actions → resolution"
                    )
                    for ev in events:
                        icon   = EVENT_ICONS.get(ev["event_type"], "•")
                        colour = EVENT_COLOURS.get(ev["event_type"], "#a0aec0")
                        cam_s  = (
                            f"  •  Camera {ev['camera_id']}"
                            if ev["camera_id"] is not None else ""
                        )
                        st.markdown(
                            f"<div style='padding:6px 0;"
                            f"border-bottom:1px solid #2d3748'>"
                            f"<span style='color:gray;font-size:0.8em'>"
                            f"{fmt_time(ev['occurred_at'])}</span>&nbsp;&nbsp;"
                            f"{icon} <span style='color:{colour};"
                            f"font-weight:bold'>{ev['event_type'].upper()}</span>"
                            f"<span style='color:gray'>{cam_s}</span>"
                            f"&nbsp;&nbsp;{ev['detail']}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                st.markdown("---")

                # ── Camera journey ────────────────────────────────────
                sightings = p.get("sightings", [])
                if sightings:
                    st.subheader("📍 Camera Journey")
                    cam_data = {}
                    for s in sightings:
                        cam_data.setdefault(s["camera_id"], []).append(s["seen_at"])
                    st.dataframe(pd.DataFrame([{
                        "Camera":      f"Cam {c}",
                        "First Visit": fmt_time(min(ts)),
                        "Last Visit":  fmt_time(max(ts)),
                        "Sightings":   len(ts),
                    } for c, ts in sorted(cam_data.items())]),
                    use_container_width=True)

                    # Crop gallery
                    st.subheader("🖼 Appearance Gallery")
                    crop_paths = [
                        s["crop_path"] for s in sightings if s.get("crop_path")
                    ][:12]
                    if crop_paths:
                        gcols = st.columns(6)
                        for ci, cp in enumerate(crop_paths):
                            img = load_crop(cp)
                            if img:
                                sv = next(
                                    (s for s in sightings if s.get("crop_path") == cp),
                                    None,
                                )
                                with gcols[ci % 6]:
                                    st.image(
                                        img, width=90,
                                        caption=f"Cam {sv['camera_id']}" if sv else "",
                                    )

                    # Full sighting log
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
                    note_in = st.text_area(
                        "Add note",
                        key=f"detail_note_{selected}",   # stable key per person
                        placeholder="Enter a note...",
                    )
                    if st.button(
                        "💬 Save Note",
                        key=f"detail_save_{selected}",
                        disabled=not note_in,
                    ):
                        store.add_note(selected, note_in)
                        st.success("Saved.")

                with a2:
                    if p["status"] == "lost":
                        if st.button(
                            "✅ Resolve",
                            type="primary",
                            key=f"detail_resolve_{selected}",
                        ):
                            store.resolve(selected, note_in or "Resolved via detail view")
                            st.success("Resolved.")
                            st.rerun()
                    elif p["status"] == "resolved":
                        if st.button(
                            "🔄 Reactivate",
                            key=f"detail_reactivate_{selected}",
                        ):
                            store.reactivate(selected)
                            st.info("Reactivated.")
                            st.rerun()

                with a3:
                    st.markdown("**Current status:**")
                    st.markdown(f"### {status_badge(p['status'])}")