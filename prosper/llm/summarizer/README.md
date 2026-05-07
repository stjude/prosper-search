# patho-toy-summarizer

Minimal deterministic CLI that summarizes a fixed set of 5 neighbor pathology cases.

## Install (editable)
```bash
pip install -e .
```

## Usage
```bash
patho_summarize sample.json
```
Prints a single cautious sentence (<=25 words) describing:
- Dominant (mode) diagnosis if any
- Mean similarity or probability and range (auto-detected)
- Count of outliers (diagnosis != mode)

Wording chooses "supports" if mean similarity >0.75 (or probability >=0.5) else "suggests".

### Optional LLM Rewrite
Install extra dependency:
```bash
pip install -e .[llm]
```

#### Option A — Azure AI Foundry (recommended)
```bash
export AZURE_OPENAI_ENDPOINT=https://<your-azure-resource>.services.ai.azure.com
export AZURE_OPENAI_API_KEY=<your-azure-api-key>
export AZURE_OPENAI_DEPLOYMENT=o3-mini   # model name available on your endpoint
patho_summarize sample.json --llm
```

> **Note:** The endpoint should be the base Azure AI Services URL (without
> `/api/projects/...`).  The tool appends `/openai/v1/` automatically and
> uses the standard OpenAI client under the hood.

#### Option B — Standard OpenAI
```bash
export OPENAI_API_KEY=sk-...
export OPENAI_ORG_ID=org-...   # optional, if required
patho_summarize sample.json --llm
```

When `--llm` is used the tool attempts an LLM narrative (≤200 words). If successful it prints only that narrative. If the call fails or yields no usable text, it prints an error (if available) and then the deterministic summary.

Environment variables:
| Variable | Purpose | Default |
|----------|---------|---------|
| **Azure AI Foundry** | | |
| `AZURE_OPENAI_ENDPOINT` | Azure AI Services base URL (enables Azure mode) | unset |
| `AZURE_OPENAI_API_KEY` | API key for Azure (falls back to `OPENAI_API_KEY`) | (required for Azure) |
| `AZURE_OPENAI_DEPLOYMENT` | Model name available on your Foundry endpoint | `o3-mini` |
| **Standard OpenAI** | | |
| `OPENAI_API_KEY` | Auth for OpenAI | (required for real calls) |
| `PATHO_LLM_MODEL` | Override model name | `gpt-4o-mini` |
| `OPENAI_ORG_ID` / `PATHO_OPENAI_ORG` | Organization ID for OpenAI client | unset |
| **General** | | |
| `PATHO_LLM_FAKE` | If set (any value) returns a local synthetic summary (no network) | unset |

On errors (e.g. quota, auth, library missing) you will see an `ERROR:` line rather than silent fallback.

## JSON Schema
Two supported input schemas:

1. Minimal (original):
```jsonc
{
  "current_case": {"site": "string|null", "age": "string|null", "sex": "string|null"},
  "neighbors": [
    {"case_id":"str","similarity":0.0,"probability":0.0,"diagnosis":"str|null","site":"str|null","age":"string|null","sex":"string|null"}
    // exactly 5 neighbors
  ]
}
```

2. Enriched `neighbors.json` (produced by search pipeline):
```jsonc
{
  "schema_version": 1,
  "current_case": {
    "slide_id": "str",
    "age": "string|null",
    "sex": "string|null",
    "site": "string|null",
    "tumor_group": "string|null",
    "evaluation": "string|null",
    "sample_type": "string|null",
    "treatment": "string|null",
    "discrepant": "Y|N|null",
    "diagnosis": "string|null",
    "met": "Primary|Metastatic|Data not available|null",
    "ped_adult": "Pediatric|Adult|null"
  },
  "neighbors": [
    {
      "case_id": "str",
      "similarity": 0.0,
      "probability": 0.0,
      "age": "string|null",
      "sex": "string|null",
      "site": "string|null",
      "tumor_group": "string|null",
      "evaluation": "string|null",
      "sample_type": "string|null",
      "treatment": "string|null",
      "discrepant": "Y|N|null",
      "diagnosis": "string|null",
      "met": "Primary|Metastatic|Data not available|null",
      "ped_adult": "Pediatric|Adult|null"
    }
    // exactly 5 neighbors
  ]
}
```

Use `--neighbors-json` flag with enriched schema to enable richer LLM narrative.

## Testing
```bash
pytest -q
```

## Example Deterministic Output (fallback)
```
Neighbors supports Adenocarcinoma, similarity mean 0.79 (range 0.76-0.82, lower is better), 1 outlier(s)
```

## Example LLM Narrative
```
Neighbors support Adenocarcinoma (votes: {"Adenocarcinoma":4,"Squamous Cell":1}). Similarity mean 0.79 range 0.72-0.86 (lower is better). Current case: site=lung, age=55, sex=F. Discrepant neighbor: N4 (Squamous Cell, similarity 0.76) diverges from dominant pattern. Overall profile supports Adenocarcinoma acknowledging one discrepant case.
```

### Enriched Input Example
```
patho_summarize output/neighbors.json --llm --neighbors-json
```
Produces a narrative incorporating demographics (age, sex, site), tumor_group consensus, discrepant flags, and metastatic vs primary distinctions.

## Notes
- Purely local deterministic logic; no network calls.
- Deterministic summary still enforces original ≤25 word constraint; LLM narrative allows up to 200 words and truncates if exceeded.
# comet-llm
