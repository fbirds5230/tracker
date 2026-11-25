from urllib.error import HTTPError

import pytest

from scripts import pytorch_projects


def links(*names):
    return "".join(f'<a href="{name}/">{name}</a>\n' for name in names)


def test_selects_public_channels_without_mistaking_packages_or_legacy_builds_for_them():
    html = links(
        "torch", "numpy", "cpu", "cpu-cxx11-abi", "cu128", "rocm6.2.4", "rocm-preview", "xpu",
        "cpu-pypi-pkg", "cu117-pypi-cudnn", "cu121-pypi-cudnn", "cu121-full", "cu124-full", "cu126-full", "cu128-full",
        "nightly", "test", "variant",
    )
    assert pytorch_projects.channel_urls(html) == [
        pytorch_projects.ROOT + name + "/"
        for name in ("cpu", "cpu-cxx11-abi", "cu128", "rocm-preview", "rocm6.2.4", "xpu")
    ]


def test_collects_the_canonical_union_from_every_selected_channel():
    root = pytorch_projects.ROOT
    pages = {
        root: links("cpu", "cu128", "xpu"),
        root + "cpu/": links("NumPy", "typing_extensions"),
        root + "cu128/": links("numpy", "Pillow"),
        root + "xpu/": links("typing.extensions", "torch"),
    }
    seen = set()

    def fetch(url):
        seen.add(url)
        return pages[url]

    names, channels = pytorch_projects.collect(fetch)
    assert names == ["numpy", "pillow", "torch", "typing-extensions"]
    assert set(channels) == set(pages) - {root}
    assert seen == set(pages)


@pytest.mark.parametrize("failure", [HTTPError("https://download.pytorch.org/whl/cu128/", 403, "Forbidden", {}, None), ValueError("bad index")])
def test_a_selected_channel_error_aborts_the_snapshot(failure):
    def fetch(url):
        if url == pytorch_projects.ROOT:
            return links("cpu", "cu128")
        if url.endswith("/cu128/"):
            raise failure
        return links("numpy")

    with pytest.raises(type(failure)):
        pytorch_projects.collect(fetch)


@pytest.mark.parametrize("html", ["<h1>Unavailable</h1>", '<a href="../outside/">outside</a>', '<a href="https://example.invalid/numpy/">numpy</a>'])
def test_a_selected_channel_without_valid_project_links_is_an_error(html):
    with pytest.raises(ValueError):
        pytorch_projects.project_names(html, pytorch_projects.ROOT + "cpu/")


def test_a_root_without_channels_is_an_error():
    with pytest.raises(ValueError):
        pytorch_projects.channel_urls(links("torch", "numpy"))
