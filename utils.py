import aiohttp
import aiofiles
import os
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("TOKEN")

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}


def get_blob_uri(signed_url: str) -> str:
    parsed = urlparse(signed_url)
    parts = parsed.path.strip('/').split('/')
    container = parts[0]
    directory_path = '/'.join(parts[1:-1])
    storage_account = parsed.netloc.split('.')[0]
    return f"wasbs://{container}@{storage_account}.blob.core.windows.net/{directory_path}/"


async def stage_artifact(session):
    url = "https://iit-ropar.truefoundry.cloud/api/ml/v1/artifact-versions/stage"
    payload = {
        "manifest": {
            "name": "agricare-artifact",
            "ml_repo": "agricare-storage",
            "source": {"type": "truefoundry"},
            "type": "artifact-version",
            "metadata": {}
        }
    }
    async with session.post(url, json=payload) as res:
        res.raise_for_status()
        return (await res.json())["id"]


async def get_signed_url(session, artifact_id, operation, file):
    url = "https://iit-ropar.truefoundry.cloud/api/ml/v1/artifact-versions/signed-urls"
    payload = {
        "id": artifact_id,
        "paths": [file],
        "operation": operation
    }
    async with session.post(url, json=payload) as res:
        res.raise_for_status()
        return (await res.json())["data"][0]["signed_url"]


async def upload_file(signed_url, file):
    headers = {"x-ms-blob-type": "BlockBlob"}
    async with aiohttp.ClientSession() as session:
        async with aiofiles.open(file, "rb") as f:
            data = await f.read()
        async with session.put(signed_url, headers=headers, data=data) as res:
            res.raise_for_status()


async def finalize_artifact(session, artifact_id, blob_uri):
    url = "https://iit-ropar.truefoundry.cloud/api/ml/v1/artifact-versions"
    payload = {
        "manifest": {
            "name": "agricare-artifact",
            "type": "artifact-version",
            "source": {
                "type": "truefoundry",
                "uri": blob_uri
            },
            "metadata": {},
            "step": 0,
            "ml_repo": "agricare-storage"
        }
    }
    async with session.put(url, json=payload) as res:
        res.raise_for_status()
        return await res.json()


async def main(file):
    async with aiohttp.ClientSession(headers=headers) as session:
        artifact_id = await stage_artifact(session)
        write_url = await get_signed_url(session, artifact_id, "WRITE", file)
        await upload_file(write_url, file)
        blob_uri = get_blob_uri(write_url)
        await finalize_artifact(session, artifact_id, blob_uri)
        read_url = await get_signed_url(session, artifact_id, "READ")
        return read_url
