import json
import requests
import pandas as pd
import deepl


def get_deepl_client(key_path="/Users/sofiaromero/Documents/DeepL.txt"):
    """Load DeepL API key and return (client, key) tuple."""
    key = open(key_path).read()
    return deepl.DeepLClient(key), key


def generate_glossaries(dict_path_l, s_lang_l, t_lang_l, auth_key):
    """Create DeepL glossaries from CSV files and return the API responses."""
    url = "https://api-free.deepl.com/v3/glossaries"
    headers = {
        "Authorization": f"DeepL-Auth-Key {auth_key}",
        "User-Agent": "YourApp/1.2.3",
        "Content-Type": "application/json",
    }
    glossaries = []
    for i, d in enumerate(dict_path_l):
        payload = {
            "name": f"glossary{i}",
            "dictionaries": [{
                "source_lang": s_lang_l[i],
                "target_lang": t_lang_l[i],
                "entries": open(d, "r").read(),
                "entries_format": "csv",
            }],
        }
        resp = requests.post(url, headers=headers, json=payload)
        print(resp.status_code, resp.json())
        glossaries.append(resp.json())
    return glossaries


def get_glossary_ids(auth_key):
    """Return list of all glossary IDs registered under the given DeepL API key."""
    url = "https://api-free.deepl.com/v3/glossaries"
    headers = {
        "Authorization": f"DeepL-Auth-Key {auth_key}",
        "User-Agent": "YourApp/1.2.3",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    print(resp.status_code)
    return [x.get("glossary_id") for x in resp.json()["glossaries"]]


def translate_to(text_l, s_lang, t_lang, deepl_client, glossary_id=None):
    """Translate a list of strings from s_lang to t_lang using DeepL."""
    res = []
    for text in text_l:
        if glossary_id:
            res.append(deepl_client.translate_text(text, target_lang=t_lang, source_lang=s_lang, glossary=glossary_id).text)
        else:
            res.append(deepl_client.translate_text(text, target_lang=t_lang, source_lang=s_lang).text)
    return res


def translate_to_df(txts_json, file_to, deepl_client, s_lang="DE", t_lang="EN-GB", glossary_id=None):
    """Load a JSON file of texts, translate them, and save results to CSV and JSON.

    The JSON can be a dict of {id: str} or {id: [str, ...]}.
    Saves to <file_to>.csv and <file_to>.json.
    """
    with open(txts_json, "r") as f:
        txts = json.load(f)

    if isinstance(list(txts.values())[0], str):
        res_df = pd.DataFrame(
            [(key, txts[key]) for key in txts],
            columns=["id", "text"],
        )
    else:
        res_df = pd.DataFrame(
            [(key, var) for key, L in txts.items() for var in L],
            columns=["id", "text"],
        )
    res_df = res_df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    res_df["translation"] = translate_to(res_df["text"].tolist(), s_lang, t_lang, deepl_client, glossary_id)
    res_df.to_csv(f"{file_to}.csv", index=False)
    res_df.to_json(f"{file_to}.json", orient="records")
    return res_df
