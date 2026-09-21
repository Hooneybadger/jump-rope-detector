import app.pdf as pdf_module


def test_pdf_report_contains_only_measurement_record_sections():
    from datetime import UTC, datetime

    content = pdf_module.workout_pdf(
        workout_id=7,
        user_name="점퍼",
        mode_name="번갈아뛰기",
        count=112,
        duration=60,
        status_name="완료",
        started_at=datetime(2026, 9, 21, tzinfo=UTC),
    )

    assert content.startswith(b"%PDF-1.7")
    assert b"/FontFile2" in content
    assert content.count(b" BT /F1 ") == 16
    assert not hasattr(pdf_module, "estimate_calories")
    assert not hasattr(pdf_module, "POSTURE_TIPS")
