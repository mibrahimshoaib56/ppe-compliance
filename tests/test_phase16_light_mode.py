from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "app.py")


def test_landing_page_renders_dark_mode_by_default():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    # Landing page is active by default
    assert at.session_state.in_app is False
    assert at.session_state.theme == "Dark Theme"

    # Verify Enter button is present
    enter_buttons = [b for b in at.button if "ENTER SAFETY SYSTEM" in b.label]
    assert len(enter_buttons) == 1

    # Verify theme toggle button is present on landing page
    landing_theme_btn = [b for b in at.button if b.key == "landing_theme_toggle"]
    assert len(landing_theme_btn) == 1
    assert landing_theme_btn[0].label == "Light Mode"


def test_landing_page_toggle_to_light_mode():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    # Click Light Mode toggle
    landing_theme_btn = [b for b in at.button if b.key == "landing_theme_toggle"][0]
    landing_theme_btn.click().run()
    assert not at.exception

    assert at.session_state.theme == "Light Theme"
    # Button label should now propose Dark Mode
    updated_btn = [b for b in at.button if b.key == "landing_theme_toggle"][0]
    assert updated_btn.label == "Dark Mode"


def test_enter_app_and_navbar_theme_toggle():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception

    # Enter the operational application
    enter_btn = [b for b in at.button if "ENTER SAFETY SYSTEM" in b.label][0]
    enter_btn.click().run()
    assert not at.exception

    assert at.session_state.in_app is True
    assert at.session_state.current_page == "Dashboard"

    # Verify top navbar theme toggle
    top_theme_btn = [b for b in at.button if b.key == "top_theme_toggle"]
    assert len(top_theme_btn) == 1
    assert top_theme_btn[0].label == "Light Mode"

    # Switch to Light Theme from navbar
    top_theme_btn[0].click().run()
    assert not at.exception
    assert at.session_state.theme == "Light Theme"
    assert at.session_state.in_app is True
    assert at.session_state.current_page == "Dashboard"

    # Switch back to Dark Theme
    top_theme_btn_back = [b for b in at.button if b.key == "top_theme_toggle"][0]
    assert top_theme_btn_back.label == "Dark Mode"
    top_theme_btn_back.click().run()
    assert not at.exception
    assert at.session_state.theme == "Dark Theme"


def test_css_tokens_dark_vs_light():
    from app import inject_custom_css
    import streamlit as st

    # We can inspect the CSS definition logic directly from app.py source
    with open("app.py", "r", encoding="utf-8") as f:
        src = f.read()

    # Verify Dark Mode tokens are strictly intact
    assert "--bg-main: #000000;" in src
    assert "--bg-sidebar: #0D0D0D;" in src
    assert "--surface-card: #151515;" in src
    assert "--accent-primary: #A3E635;" in src
    assert "--text-primary: #F1F5F9;" in src
    assert "--text-secondary: #94A3B8;"

    # Verify Light Mode tokens match Phase 16 specification
    assert "--bg-main: #F4F6F8;" in src
    assert "--bg-sidebar: #E9EEF2;" in src
    assert "--surface-card: #FFFFFF;" in src
    assert "--surface-elevated: #F4F6F8;" in src
    assert "--border-color: #DCE3E9;" in src
    assert "--accent-primary: #65A30D;" in src
    assert "--text-primary: #17212B;" in src
    assert "--text-secondary: #64748B;" in src
    assert "--color-safe: #16A34A;" in src
    assert "--color-violation: #DC2626;" in src
    assert "--color-warning: #D97706;" in src

    # Verify Light Mode active nav indicator
    assert "background-color: #DCECC2 !important;" in src
    assert "border-left: 3px solid var(--accent-primary) !important;" in src
