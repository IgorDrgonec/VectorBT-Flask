import json
from pathlib import Path
from argparse import ArgumentParser

import requests

with open("../.github_gist_token", "r") as f:
    GITHUB_TOKEN = f.read()

locked_notebooks = {
    Path("../notebooks/BasicRSI.ipynb"): "Basic RSI strategy",
    Path("../notebooks/SuperTrend.ipynb"): "SuperFast SuperTrend",
    Path("../notebooks/MTFAnalysis.ipynb"): "MTF analysis",
    Path("../notebooks/PortfolioOptimization.ipynb"): "Portfolio optimization",
    Path("../notebooks/SignalDevelopment.ipynb"): "Signal development",
}


def get_gists():
    url = "https://api.github.com/gists?since=2022-01-01T00:00:00Z&per_page=100"
    payload = {}
    headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {GITHUB_TOKEN}"}
    return requests.get(url, data=payload, headers=headers)


def delete_gist(gist_id):
    url = "https://api.github.com/gists/" + gist_id
    payload = {}
    headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {GITHUB_TOKEN}"}
    return requests.delete(url, data=payload, headers=headers)


def create_gist(file_name, content, description):
    url = "https://api.github.com/gists"
    payload = json.dumps({"files": {file_name: {"content": content}}, "description": description, "public": False})
    headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {GITHUB_TOKEN}"}
    return requests.post(url, data=payload, headers=headers)


def update_gist(gist_id, file_name, content, description):
    url = "https://api.github.com/gists/" + gist_id
    payload = json.dumps({"files": {file_name: {"content": content}}, "description": description, "public": False})
    headers = {"Accept": "application/vnd.github.v3+json", "Authorization": f"token {GITHUB_TOKEN}"}
    return requests.patch(url, data=payload, headers=headers)


parser = ArgumentParser()
parser.add_argument("--renew", action="store_true")
args = parser.parse_args()


if __name__ == "__main__":
    gists = get_gists().json()
    gist_urls = {}

    for notebook_path, notebook_title in locked_notebooks.items():
        with open(notebook_path, "r") as f:
            content = f.read()

        if args.renew:
            for gist in gists:
                if notebook_path.name in gist["files"]:
                    print(f"Deleting the gist for '{notebook_path.name}'...")
                    res = delete_gist(gist["id"])
                    print(res.status_code)
                    break

            print(f"Creating a new gist for '{notebook_path.name}'...")
            res = create_gist(notebook_path.name, content, notebook_title)
            gist_urls[notebook_path] = res.json()["url"]
            print(res.status_code, gist_urls[notebook_path])
        else:
            found_gist = False
            for gist in gists:
                if notebook_path.name in gist["files"]:
                    print(f"Updating the gist for '{notebook_path.name}'...")
                    res = update_gist(gist["id"], notebook_path.name, content, notebook_title)
                    print(res.status_code)
                    found_gist = True
                    break
            if not found_gist:
                print(f"Couldn't find the gist for '{notebook_path.name}'!")

    if args.renew:
        links = []
        for notebook_path, url in gist_urls.items():
            links.append(
                "* [{}](https://nbviewer.org/gist/polakowo/{})".format(
                    locked_notebooks[notebook_path], url.split("/")[-1]
                )
            )
        with open("../locked-notebooks.md", "w") as f:
            f.write("\n".join(links))
