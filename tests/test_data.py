from PIL import Image

from src.data import find_images, validate_image


def test_find_images_filters_and_sorts(tmp_path):
    (tmp_path / "b.png").write_bytes(b"not an image")
    (tmp_path / "a.JPG").write_bytes(b"not an image")
    (tmp_path / "notes.txt").write_text("skip")

    assert [p.name for p in find_images(tmp_path)] == ["a.JPG", "b.png"]


def test_validate_image_accepts_real_image_and_rejects_invalid(tmp_path):
    good = tmp_path / "good.png"
    Image.new("RGB", (3, 3), color="green").save(good)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")

    assert validate_image(good)
    assert not validate_image(bad)

