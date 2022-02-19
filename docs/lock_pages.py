import glob
import gzip
import json
import re
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from argparse import ArgumentParser

locked_pages = {
    Path('examples/superfast-supertrend/streaming-locked'):
        'Superfast SuperTrend - Streaming',
    Path('examples/superfast-supertrend/multithreading-locked'):
        'Superfast SuperTrend - Multithreading',
    Path('examples/superfast-supertrend/pipelines-locked'):
        'Superfast SuperTrend - Pipelines'
}


def generate_uuid():
    return str(uuid.uuid4())


def namespace(element):
    m = re.match(r'\{(.*)\}', element.tag)
    return m.group(1) if m else ''


parser = ArgumentParser()
parser.add_argument("--renew", action='store_true')
args = parser.parse_args()

if __name__ == "__main__":
    if args.renew:
        while True:
            locked_uuid = {page_path: generate_uuid() for page_path in locked_pages}
            if len(set(locked_uuid.values())) == len(locked_pages):
                break
    else:
        locked_uuid = {}
        with open('../locked-pages.md', 'r') as f:
            lines = f.read().split('\n')
        for line in lines:
            for page_path in locked_pages:
                if str(page_path) in line:
                    locked_uuid[page_path] = line[-37:-1]
                    break

    tree = ET.parse('./site/sitemap.xml')
    root = tree.getroot()
    ET.register_namespace('', namespace(root))
    for page_path in locked_pages:
        for child in list(root):
            for subchild in list(child):
                if str(page_path) in subchild.text:
                    root.remove(child)
    tree.write('./site/sitemap.xml', encoding='utf-8', xml_declaration=True)
    with gzip.open("./site/sitemap.xml.gz", 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=True)

    with open("./site/search/search_index.json", 'r') as f:
        dct = json.loads(f.read())
    new_docs = []
    for doc in dct['docs']:
        if 'locked/' not in doc['location']:
            new_docs.append(doc)
    dct['docs'] = new_docs
    with open("./site/search/search_index.json", 'w') as f:
        f.write(json.dumps(dct))

    for p in list(glob.iglob('./site/**/*-locked', recursive=True)):
        p_path = Path(p)
        found_path = False
        for page_path in locked_pages:
            if './site/' + str(page_path) == p:
                new_path = p.replace(
                    str(page_path),
                    f'{page_path}-{locked_uuid[page_path]}'
                )
                with open(p_path / 'index.html', 'r') as f:
                    s = f.read()
                s = s.replace(
                    '<head>',
                    '<head><meta name="robots" content="noindex">'
                )
                s = s.replace(
                    str(page_path),
                    f'{page_path}-{locked_uuid[page_path]}'
                )
                with open(p_path / 'index.html', 'w') as f:
                    f.write(s)
                p_path.rename(new_path)
                found_path = True
                break
        if not found_path:
            raise ValueError(f"Locked page '{p}' missing")

    if args.renew:
        links = []
        for page_path, page_title in locked_pages.items():
            links.append('* [{}](https://vectorbt.pro/{}-{})'.format(
                page_title,
                str(page_path),
                locked_uuid[page_path]
            ))
        with open('../locked-pages.md', 'w') as f:
            f.write('\n'.join(links))
