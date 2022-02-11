import json
from pathlib import Path

import requests

with open('../.github_gist_token', 'r') as f:
    GITHUB_TOKEN = f.read()

locked_notebooks = {
    Path('../notebooks/BasicRSI.ipynb'): 'Basic RSI strategy',
    Path('../notebooks/SuperTrend.ipynb'): 'Superfast SuperTrend'
}

gist_urls = {}


def get_gists():
    url = 'https://api.github.com/gists?since=2022-01-01T00:00:00Z'
    payload = {}
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {GITHUB_TOKEN}'
    }
    return requests.get(url, data=payload, headers=headers)


def delete_gist(gist_id):
    url = 'https://api.github.com/gists/' + gist_id
    payload = {}
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {GITHUB_TOKEN}'
    }
    return requests.delete(url, data=payload, headers=headers)


def create_gist(file_name, content, description):
    url = 'https://api.github.com/gists'
    payload = json.dumps({
        'files': {
            file_name: {
                'content': content
            }
        },
        'description': description,
        'public': False
    })
    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'Authorization': f'token {GITHUB_TOKEN}'
    }
    return requests.post(url, data=payload, headers=headers)


if __name__ == "__main__":
    gists = get_gists().json()

    for notebook_path, notebook_title in locked_notebooks.items():
        with open(notebook_path, 'r') as f:
            content = f.read()

        for gist in gists:
            if notebook_path.name in gist['files']:
                print(f"Deleting the gist for '{notebook_path.name}'...")
                res = delete_gist(gist['id'])
                print(res.status_code)

        print(f"Creating a new gist for '{notebook_path.name}'...")
        res = create_gist(notebook_path.name, content, notebook_title)
        gist_urls[notebook_path] = res.json()['url']
        print(res.status_code, gist_urls[notebook_path])

    links = []
    for notebook_path, url in gist_urls.items():
        links.append('* [{}](https://nbviewer.org/gist/polakowo/{})'.format(
            locked_notebooks[notebook_path],
            url.split('/')[-1]
        ))
    with open('../locked-notebooks.md', 'w') as f:
        f.write('\n'.join(links))
