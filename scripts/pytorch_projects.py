"""Print driver.py's static PyTorch project snapshot; see README's "PyTorch extra index snapshot".

Only this maintenance script reads the public indexes. The browser never runs it. A failed request or an
unreadable selected index aborts before anything is printed, so a partial fetch cannot replace the snapshot.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
import re
from urllib.parse import urljoin, urlsplit
from urllib.request import urlopen

ROOT = "https://download.pytorch.org/whl/"
# Published CPU/CUDA/ROCm/XPU channels, including historical versions and ROCm preview. The old cpu-cxx11-abi
# page is still readable, so include it too. Legacy -full/-pypi-* build directories have no readable project
# listing at the root's links; their exclusion and the upstream publisher's rules are documented in README.
CHANNEL = re.compile(r"(?:cpu(?:-cxx11-abi)?|cu[0-9]+|rocm(?:[0-9]+(?:\.[0-9]+)*|-preview)|xpu)")


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs.extend(value for key, value in attrs if key == "href" and value)


def hrefs(html):
    parser = Links()
    parser.feed(html)
    parser.close()
    return parser.hrefs


def channel_urls(html):
    # The root mixes channel directories with package names. Follow only the channel-shaped relative links.
    names = {link[:-1] for link in hrefs(html) if link.endswith("/") and CHANNEL.fullmatch(link[:-1])}
    if not names:
        raise ValueError(f"{ROOT}: no public channel links")
    return [ROOT + name + "/" for name in sorted(names)]


def project_names(html, url):
    index = urlsplit(url)
    names = set()
    for link in hrefs(html):
        target = urlsplit(urljoin(url, link))
        if (target.scheme, target.netloc) != (index.scheme, index.netloc) or target.query or target.fragment:
            raise ValueError(f"{url}: unexpected project link {link!r}")
        if not target.path.startswith(index.path):
            raise ValueError(f"{url}: project link leaves the index: {link!r}")
        name = target.path[len(index.path):].rstrip("/")
        if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", name):
            raise ValueError(f"{url}: invalid project name in {link!r}")
        names.add(re.sub(r"[-_.]+", "-", name).lower())
    if not names:
        raise ValueError(f"{url}: no project links")
    return names


def read_html(url):
    with urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def collect(fetch=read_html):
    channels = channel_urls(fetch(ROOT))
    with ThreadPoolExecutor(max_workers=6) as pool:
        pages = pool.map(lambda url: project_names(fetch(url), url), channels)
        names = set().union(*pages)
    return sorted(names), channels


def main():
    names, channels = collect()
    date = datetime.now(timezone.utc).date()
    print(f"# Union of {len(channels)} public PyTorch channel indexes, retrieved {date}.")
    print("# Regenerate with: python3 scripts/pytorch_projects.py (see README for scope and exclusions).")
    print("PYTORCH_PROJECTS = (")
    for name in names:
        print(f'    "{name}",')
    print(")")


if __name__ == "__main__":
    main()
