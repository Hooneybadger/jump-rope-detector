import struct
import zlib

import pytest

from app import calories


def signup(client, username="jumper"):
    response = client.post("/api/auth/signup", json={
        "username": username, "email": f"{username}@example.com", "display_name": "점퍼",
        "password": "correct-horse-battery-staple",
    })
    assert response.status_code == 201
    client.headers["X-CSRF-Token"] = response.json()["csrfToken"]
    return response.json()


def tiny_png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"\x00\xd9\x62\x2a")) + chunk(b"IEND", b"")


def test_new_member_is_asked_for_a_profile_until_they_answer(client):
    user = signup(client)
    assert user["profile"]["status"] == "pending"
    assert user["profile"]["recommendation"] is None
    skipped = client.post("/api/auth/body-profile/skip")
    assert skipped.json()["profile"]["status"] == "skipped"
    assert client.get("/api/auth/me").json()["profile"]["status"] == "skipped"


def test_saving_a_profile_unlocks_a_recommended_duration(client):
    signup(client)
    response = client.put("/api/auth/body-profile", json={"sex": "female", "age": 34, "height_cm": 158.04, "weight_kg": 52.5})
    assert response.status_code == 200
    profile = response.json()["profile"]
    assert profile == {
        "status": "completed", "sex": "female", "age": 34, "heightCm": 158.0, "weightKg": 52.5, "bmi": 21.0,
        "recommendation": {"seconds": 643, "reason": profile["recommendation"]["reason"]},
    }
    # Skipping later never discards a completed profile.
    assert client.post("/api/auth/body-profile/skip").json()["profile"]["status"] == "completed"


@pytest.mark.parametrize("payload", [
    {"age": 5}, {"age": 34.5}, {"height_cm": 300}, {"weight_kg": 10}, {"sex": "other"}, {"birth_date": "1990-01-01"},
])
def test_profile_rejects_out_of_range_or_unknown_fields(client, payload):
    signup(client)
    assert client.put("/api/auth/body-profile", json=payload).status_code == 422


def test_profile_changes_need_the_csrf_token(client):
    signup(client)
    client.headers.pop("X-CSRF-Token")
    assert client.put("/api/auth/body-profile", json={"age": 30}).status_code == 403
    assert client.post("/api/auth/body-profile/skip").status_code == 403


def test_admin_user_list_never_exposes_body_profiles(admin_client):
    rows = admin_client.get("/api/admin/users").json()
    assert rows and all("profile" not in row and "avatarUrl" not in row for row in rows)


def test_profile_photo_upload_replace_and_delete(client):
    signup(client)
    image = tiny_png()
    response = client.put("/api/auth/avatar", content=image, headers={"Content-Type": "image/png"})
    assert response.status_code == 200
    assert response.json()["avatarUrl"].startswith("/api/auth/avatar?v=")
    stored = client.get("/api/auth/avatar")
    assert stored.content == image and stored.headers["content-type"] == "image/png"
    assert client.delete("/api/auth/avatar").json()["avatarUrl"] is None
    assert client.get("/api/auth/avatar").status_code == 404


def test_profile_photo_rejects_non_images_and_large_files(client):
    signup(client)
    assert client.put("/api/auth/avatar", content=b"<svg onload=alert(1)>", headers={"Content-Type": "image/png"}).status_code == 415
    large = b"\xff\xd8\xff" + b"0" * (512 * 1024)
    assert client.put("/api/auth/avatar", content=large, headers={"Content-Type": "image/jpeg"}).status_code == 413
    assert client.put("/api/auth/avatar", content=b"", headers={"Content-Type": "image/jpeg"}).status_code == 400


@pytest.mark.parametrize(("profile", "seconds"), [
    (calories.BodyProfile(age=10), 3600),
    (calories.BodyProfile(age=17, height_cm=170, weight_kg=90), 3600),
    (calories.BodyProfile(age=18, height_cm=170, weight_kg=90), 3600),  # Korean guidelines: youth is 6-18
    (calories.BodyProfile(age=19), 643),  # 75 min/week vigorous / 7
    (calories.BodyProfile(age=34, height_cm=158, weight_kg=52), 643),
    (calories.BodyProfile(age=41, height_cm=172, weight_kg=86), 1286),  # BMI 29.1: 150 min/week / 7
    (calories.BodyProfile(age=64, height_cm=160, weight_kg=66), 1286),
    (calories.BodyProfile(sex="female"), 643),
    (calories.BodyProfile(age=70), 643),
    (calories.BodyProfile(age=70, height_cm=155, weight_kg=62), 857),  # 65+ upper bound is 100 min/week
])
def test_recommended_time_follows_korean_guidelines_by_age_and_bmi(profile, seconds):
    assert calories.recommendation(profile).seconds == seconds


def test_recommended_time_does_not_depend_on_sex():
    for body in ({"age": 30, "height_cm": 170, "weight_kg": 65}, {"age": 12}, {"age": 70, "height_cm": 160, "weight_kg": 70}):
        male = calories.recommendation(calories.BodyProfile(sex="male", **body))
        female = calories.recommendation(calories.BodyProfile(sex="female", **body))
        assert male.seconds == female.seconds


@pytest.mark.parametrize(("seconds", "label"), [(100, "1분 40초"), (1800, "30분"), (45, "45초"), (643, "10분 43초"), (0, "0초")])
def test_duration_label_uses_minutes_and_seconds(seconds, label):
    assert calories.duration_label(seconds) == label


def test_no_recommendation_without_profile_data():
    assert calories.recommendation(None) is None
    assert calories.recommendation(calories.BodyProfile()) is None


def test_adult_calories_use_own_body_mass():
    # 11.8 MET x 3.5 x 80 kg / 200 = 16.52 kcal per minute
    estimate = calories.estimate("basic", 110, 60, profile=calories.BodyProfile(age=30, weight_kg=80))
    assert estimate.personal and estimate.basis.code == "15551"
    assert estimate.kcal == pytest.approx(16.52)


@pytest.mark.parametrize(("age", "mety"), [(6, 6.9), (9, 6.9), (10, 7.1), (12, 7.1), (13, 7.2), (15, 7.2), (16, 7.4), (18, 7.4)])
def test_youth_jump_rope_uses_nccor_smoothed_mety(age, mety):
    # NCCOR Youth Compendium, 10260X Jump Rope, smoothed values (the default for energy estimates).
    assert calories.youth_basis(age).met == mety


def test_youth_calories_use_youth_compendium_and_schofield_bmr():
    profile = calories.BodyProfile(sex="male", age=10, weight_kg=35)
    estimate = calories.estimate("alternating", 110, 60, profile=profile)
    # METy 7.1 x (17.686 x 35 + 658.2) kcal/day / 1440
    assert estimate.basis.code == "10260X" and estimate.basis.met == 7.1
    assert estimate.kcal == pytest.approx(7.1 * (17.686 * 35 + 658.2) / 1440)
    assert estimate.net_kcal == pytest.approx(6.1 * (17.686 * 35 + 658.2) / 1440)
    girl = calories.estimate("basic", 110, 60, profile=calories.BodyProfile(sex="female", age=10, weight_kg=35))
    assert girl.kcal < estimate.kcal


def test_youth_without_sex_averages_both_equations():
    unknown = calories.schofield_bmr(None, 8, 25)
    assert unknown == pytest.approx((calories.schofield_bmr("male", 8, 25) + calories.schofield_bmr("female", 8, 25)) / 2)


def test_youth_without_weight_still_uses_the_youth_compendium():
    estimate = calories.estimate("basic", 100, 60, profile=calories.BodyProfile(sex="female", age=14))
    assert estimate.basis.youth and estimate.weight_kg == calories.YOUTH_REFERENCE_WEIGHT_KG
    assert not estimate.personal


def test_harris_benedict_matches_the_compendium_coefficients():
    # 30-year-old, 175 cm, 70 kg man: 66.4730 + 13.7516 x 70 + 5.0033 x 175 - 6.7550 x 30
    assert calories.harris_benedict_rmr("male", 30, 175, 70) == pytest.approx(1702.0, abs=0.1)
    assert calories.harris_benedict_rmr("female", 30, 160, 55) == pytest.approx(1336.7, abs=0.1)
    both = calories.harris_benedict_rmr(None, 30, 170, 60)
    assert both == pytest.approx((calories.harris_benedict_rmr("male", 30, 170, 60) + calories.harris_benedict_rmr("female", 30, 170, 60)) / 2)


def test_net_calories_subtract_the_users_own_resting_energy():
    man = calories.estimate("basic", 1100, 600, profile=calories.BodyProfile("male", 25, 180, 70))
    woman = calories.estimate("basic", 1100, 600, profile=calories.BodyProfile("female", 25, 180, 70))
    older = calories.estimate("basic", 1100, 600, profile=calories.BodyProfile("male", 75, 180, 70))
    # The oxygen cost of jumping scales with body mass, so the gross total is the same...
    assert man.kcal == woman.kcal == older.kcal == pytest.approx(11.8 * 3.5 * 70 / 200 * 10)
    # ...while a lower resting rate (women, older adults) leaves more of it to the exercise itself.
    assert man.resting_method == "harris-benedict"
    assert man.net_kcal == pytest.approx(man.kcal - calories.harris_benedict_rmr("male", 25, 180, 70) / 1440 * 10)
    assert man.net_kcal < woman.net_kcal and man.net_kcal < older.net_kcal


def test_net_calories_fall_back_to_one_met_without_age_and_height():
    estimate = calories.estimate("basic", 110, 60, profile=calories.BodyProfile(weight_kg=80))
    assert estimate.resting_method == "standard"
    assert estimate.net_kcal == pytest.approx((11.8 - 1) * 3.5 * 80 / 200)


def test_older_adults_get_met60_intensity_without_changing_energy():
    estimate = calories.estimate("basic", 110, 60, profile=calories.BodyProfile(age=70, weight_kg=60))
    assert estimate.kcal == pytest.approx(11.8 * 3.5 * 60 / 200)
    assert estimate.met60 == pytest.approx(11.8 * 3.5 / 2.7)
    assert calories.estimate("basic", 110, 60, profile=calories.BodyProfile(age=59, weight_kg=60)).met60 is None
