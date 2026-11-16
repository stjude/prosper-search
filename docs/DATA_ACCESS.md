# Data access

This repository contains code only. It contains no whole-slide images,
mosaics, embeddings, labels, case-level metadata, crosswalks, reader-study
records, LLM payloads or model weights. The Apache-2.0 licence covers the code.
It grants no rights to any dataset.

## Datasets used in the manuscript

| Dataset | Role in the paper | Access |
|---|---|---|
| COMET (St. Jude) | Internal reference library and query set: 2,778 slides from 1,876 patients (14 tumor groups and non-neoplastic tissue) in the all-tissue benchmark, out of 2,921 WSIs | Images and case-level metadata are not publicly available because of patient privacy and institutional governance restrictions. Access may be considered subject to institutional approval and applicable data-use agreements; requests go to the corresponding author (abbas.shirinifard@stjude.org). The COMET resource can be explored at https://comet.stjude.org, and PROSPER slide search runs in the St. Jude PeCan portal (https://pecan.stjude.cloud/histology). |
| STV (Solid Tumor Validation dataset, St. Jude) | Additional query collection: 696 eligible slides from 350 patients, with no patients shared with COMET | Restricted on the same terms as COMET |
| KidRare | Additional query collection (1,581 eligible queries) | Publicly discoverable at https://huggingface.co/datasets/Firehdx233/KidRare. Use requires compliance with the provider's access conditions, noncommercial terms and redistribution restrictions. |
| CCDI-MCI | Additional query collection (524 verified FFPE queries) | Available through the source named in the manuscript's CCDI-MCI citation, subject to its access requirements |
| Reader-study records | Reader-study endpoints (Fig. 8, Supplementary Fig. S14, Supplementary Data 6) | Not released. Reader identities, session records and the summaries shown to readers are personal or case-level data. |

Aggregate results and cohort accounting are in Supplementary Data 1–6 of the
manuscript. Source images and restricted case-level crosswalks are not
redistributed.

## Foundation-model weights

Pretrained encoders such as Virchow2, UNI2-h, Prov-GigaPath, H-optimus-1,
CONCH v1.5/TITAN and PRISM are obtained from their providers under each
provider's licence, often after accepting terms on Hugging Face.
`scripts/setup_models.py` caches the models that the package CLI supports. This
repository does not redistribute any weights.

## Package data files

`data/README.md` describes the data files expected by the package demo and
pre-computed search, such as the slide-embedding matrix, patch embeddings,
thumbnails and the sample slide. These files are not public downloads. They
are released only through St. Jude data-governance channels.
`scripts/download_data.py` works only against an authorized source configured
for your site.

## Language-model services

The LLM summarization step calls an Azure OpenAI or OpenAI endpoint that you
configure through environment variables (see the root README). No credentials
or endpoints are included. Sending case metadata to an external service needs
your institution's approval.
