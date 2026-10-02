"""Streamlit integration checks exercise every view and both model paths."""

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize("page", ["Overview", "Data Quality", "EDA", "ML Readiness", "Report"])
def test_demo_views_render_without_exceptions(page):
    app = AppTest.from_file("app/streamlit_app.py", default_timeout=30).run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    assert app.title[0].value == page
    assert app.session_state["profile"].rows == 612


@pytest.mark.parametrize("task", ["Classification · churn", "Regression · annual value"])
def test_model_lab_training_and_report(task):
    app = AppTest.from_file("app/streamlit_app.py", default_timeout=60).run()
    app.sidebar.selectbox[0].set_value(task).run()
    app.sidebar.radio[0].set_value("Model Lab").run()
    app.button[0].click().run()
    assert not app.exception
    assert len(app.session_state["model_run"].results) == 4
    app.sidebar.radio[0].set_value("Report").run()
    assert not app.exception


def test_upload_empty_state_and_target_change_clear_results():
    app = AppTest.from_file("app/streamlit_app.py", default_timeout=60).run()
    app.sidebar.radio[0].set_value("Model Lab").run()
    app.button[0].click().run()
    assert "model_run" in app.session_state
    app.sidebar.selectbox[1].set_value("No target selected").run()
    assert "model_run" not in app.session_state
    app.sidebar.radio[1].set_value("Upload CSV").run()
    assert not app.exception
    assert any("Upload a CSV" in message.value for message in app.info)
