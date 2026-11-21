# Apache-2.0
"""What easydetect-hub says about each dataset it carries: name, licence,
official page, the credit its licence asks for, and the paper to cite.

Public datasets whose licences allow sharing models trained on them; each
keeps its own terms (some pictures stay under third-party terms, such as
Flickr's), shown beside its models. Each licence was read on the dataset's
official page (Nov 2026).
"""

from __future__ import annotations

DATASETS = {
    "coco": {
        "title": "COCO 2017",
        "license": "CC BY 4.0 (annotations); pictures under the Flickr terms",
        "source": "https://cocodataset.org",
        "attribution": "COCO 2017 annotations, CC BY 4.0 (the COCO Consortium); pictures "
                       "from Flickr under its terms of use.",
        "citation": "T.-Y. Lin et al. Microsoft COCO: Common Objects in Context. ECCV 2014. "
                    "arXiv:1405.0312.",
        "note": "640 px: easydetect's ed-s (from D-FINE's Apache-2.0 COCO weights). 320 px: "
                "the same trained again at 320 with 100 queries; its 8-bit file is "
                "quantization-aware (trained in 8 bits), as the Pibo robot runs it.",
    },
    "openimages": {
        "title": "Open Images V7",
        "license": "CC BY 4.0 (boxes, Google LLC); pictures listed as CC BY 2.0",
        "source": "https://storage.googleapis.com/openimages/web/index.html",
        "attribution": "Open Images V7 by Google LLC, boxes under CC BY 4.0; pictures listed "
                       "by their owners as CC BY 2.0.",
        "citation": "A. Kuznetsova et al. The Open Images Dataset V4. IJCV 2020. "
                    "arXiv:1811.00982.",
        "note": "Not every object is boxed: classes known to be unboxed in a picture "
                "(groups, parent classes, present but not boxed) are neither right nor "
                "wrong when scoring.",
    },
    "construction-ppe": {
        "title": "Construction-PPE",
        "license": "CC BY 4.0",
        "source": "https://docs.ultralytics.com/datasets/detect/construction-ppe/",
        "attribution": "Construction-PPE dataset by Ultralytics, CC BY 4.0.",
        "citation": "Ultralytics. Construction-PPE dataset. "
                    "docs.ultralytics.com/datasets/detect/construction-ppe.",
    },
    "lvis": {
        "title": "LVIS v1",
        "license": "CC BY 4.0 (annotations); pictures from COCO under the Flickr terms",
        "source": "https://www.lvisdataset.org",
        "attribution": "LVIS v1 annotations, CC BY 4.0; pictures from COCO (Flickr terms of use).",
        "citation": "A. Gupta, P. Dollár, R. Girshick. LVIS: A Dataset for Large Vocabulary "
                    "Instance Segmentation. CVPR 2019. arXiv:1908.03195.",
        "note": "Scored COCO's way, not with LVIS's federated AP: a class LVIS marks as "
                "not exhaustively boxed in a picture is neither right nor wrong there.",
    },
    "taco": {
        "title": "TACO (Trash Annotations in Context)",
        "license": "CC BY 4.0 (annotations); pictures CC BY 4.0 or ODbL, as listed per picture",
        "source": "http://tacodataset.org",
        "attribution": "TACO by P. Proença and P. Simões, CC BY 4.0; some pictures "
                       "ODbL (c) OpenLitterMap & Contributors.",
        "citation": "P. F. Proença, P. Simões. TACO: Trash Annotations in Context for "
                    "Litter Detection. arXiv:2003.06975, 2020.",
    },
    "doclaynet": {
        "title": "DocLayNet",
        "license": "CDLA-Permissive-1.0",
        "source": "https://github.com/DS4SD/DocLayNet",
        "attribution": "DocLayNet by IBM Research, CDLA-Permissive-1.0.",
        "citation": "B. Pfitzmann et al. DocLayNet: A Large Human-Annotated Dataset for "
                    "Document-Layout Analysis. KDD 2022. arXiv:2206.01062.",
    },
    "gwhd": {
        "title": "Global Wheat Head Dataset 2021",
        "license": "CC BY 4.0",
        "source": "https://zenodo.org/records/5092309",
        "attribution": "Global Wheat Head Dataset 2021, CC BY 4.0.",
        "citation": "E. David et al. Global Wheat Head Dataset 2021: More Diversity to "
                    "Improve the Benchmarking of Wheat Head Localization Methods. "
                    "Plant Phenomics 2021. arXiv:2105.07660.",
    },
    "caltech-camera-traps": {
        "title": "Caltech Camera Traps (CCT-20)",
        "license": "CDLA-Permissive-1.0",
        "source": "https://lila.science/datasets/caltech-camera-traps",
        "attribution": "Caltech Camera Traps (LILA BC), CDLA-Permissive-1.0.",
        "citation": "S. Beery, G. Van Horn, P. Perona. Recognition in Terra Incognita. "
                    "ECCV 2018. arXiv:1807.04975.",
    },
    "plantdoc": {
        "title": "PlantDoc",
        "license": "CC BY 4.0",
        "source": "https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset",
        "attribution": "PlantDoc by D. Singh et al., CC BY 4.0.",
        "citation": "D. Singh et al. PlantDoc: A Dataset for Visual Plant Disease Detection. "
                    "CoDS-COMAD 2020. arXiv:1911.10317.",
    },
    "bccd": {
        "title": "BCCD (blood cells)",
        "license": "MIT",
        "source": "https://github.com/Shenggan/BCCD_Dataset",
        "attribution": "BCCD Dataset, MIT licence.",
        "citation": "BCCD Dataset. github.com/Shenggan/BCCD_Dataset.",
    },
    "fashionpedia": {
        "title": "Fashionpedia",
        "license": "CC BY 4.0 (annotations); only pictures under CC BY 2.0, public domain or "
                   "free stock licences (Unsplash, Pexels, freestocks, Burst)",
        "source": "https://fashionpedia.github.io/home/",
        "attribution": "Fashionpedia annotations, CC BY 4.0 (the iMaterialist Fashion group); "
                       "pictures under the licence listed with each.",
        "citation": "M. Jia et al. Fashionpedia: Ontology, Segmentation, and an Attribute "
                    "Localization Dataset. ECCV 2020. arXiv:2004.12276.",
        "note": "Trained only on the pictures whose own licence allows any use: those under "
                "non-commercial, no-derivatives, share-alike or unknown terms were left out.",
    },
    "ap10k": {
        "title": "AP-10K (animal pose)",
        "license": "CC BY 4.0",
        "source": "https://github.com/AlexTheBad/AP-10K",
        "attribution": "AP-10K by H. Yu et al., CC BY 4.0.",
        "citation": "H. Yu et al. AP-10K: A Benchmark for Animal Pose Estimation in the Wild. "
                    "NeurIPS 2021 Datasets and Benchmarks. arXiv:2108.12617.",
    },
    "izembek": {
        "title": "Izembek Lagoon Waterfowl",
        "license": "CC0 1.0",
        "source": "https://lila.science/datasets/izembek-lagoon-waterfowl/",
        "attribution": "Izembek Lagoon waterfowl surveys, U.S. Geological Survey (LILA BC), "
                       "CC0 1.0.",
        "citation": "E. L. Weiser et al. Aerial photo imagery from fall waterfowl surveys, "
                    "Izembek Lagoon, Alaska, 2017-2019. U.S. Geological Survey data release, "
                    "2022. doi:10.5066/P9UHP1LE.",
        "note": "The survey pictures (8688 x 5792) are cut into 640 px squares, so a bird "
                "keeps its size: run the model on squares of a big aerial picture too.",
    },
    "fathomnet": {
        "title": "FathomNet (CC0 / CC BY part)",
        "license": "CC0 1.0 or CC BY 4.0, as listed with each picture and box",
        "source": "https://fathomnet.org",
        "attribution": "FathomNet (MBARI and contributors); pictures and boxes under CC0 1.0 "
                       "or CC BY 4.0 as listed with each.",
        "citation": "K. Katija et al. FathomNet: A global image database for enabling "
                    "artificial intelligence in the ocean. Scientific Reports 12, 2022. "
                    "arXiv:2109.14646.",
        "note": "Only pictures that are CC0 or CC BY 4.0 with every box CC0 or CC BY 4.0, "
                "and the 60 animals boxed most among them.",
    },
}

RF100 = {
    "license": "CC BY 4.0",
    "citation": "F. Ciaglia et al. Roboflow 100: A Rich, Multi-Domain Object Detection "
                "Benchmark. arXiv:2211.13523, 2022.",
}


def lookup(key: str) -> dict:
    """The entry for ``key``; ``rf100-<name>`` for any Roboflow 100 dataset."""
    if key in DATASETS:
        return dict(DATASETS[key])
    if key.startswith("rf100-"):
        name = key[len("rf100-"):]
        return {"title": f"Roboflow 100: {name}", "license": RF100["license"],
                "source": f"https://huggingface.co/datasets/Francesco/{name}",
                "attribution": f"{name} from Roboflow 100 (Roboflow Universe), CC BY 4.0.",
                "citation": RF100["citation"]}
    raise KeyError(f"{key}: not a dataset easydetect-hub carries "
                   f"(known: {', '.join(DATASETS)}, rf100-<name>)")


#: how easydetect-hub's front page groups them, in this order
CATEGORIES = {
    "General objects": "coco openimages lvis",
    "Everyday objects": "fashionpedia chess-pieces-mjzgj coins-1apki currency-v4f8j "
                        "furniture-ngpea marbles "
                        "poker-cards-cxcvz soda-bottles uno-deck valentines-chocolate "
                        "washroom-rf1fa wine-labels pills-sxdht lettuce-pallets",
    "People, safety and traffic": "construction-ppe construction-safety-gsnvb mask-wearing-608pr "
                                  "hand-gestures-jps7z sign-language-sokdr people-in-paintings "
                                  "soccer-players-5fuqs thermal-dogs-and-people-x6ejw smoke-uvylj "
                                  "road-signs-6ih4y road-traffic vehicles-q0x2v street-work "
                                  "truck-movement excavators-czvg9",
    "Animals and environment": "ap10k caltech-camera-traps animals-ij5d2 bees-jt5in insects-mytwu "
                               "pests-2xlvx thermal-cheetah-my4dp trail-camera shark-teeth-5atku "
                               "cloud-types taco",
    "Agriculture and plants": "gwhd plantdoc apples-fvpl5 cotton-20xz5 cotton-plant-disease "
                              "grass-weeds leaf-disease-nsdsr peanuts-sd4kf weed-crop-aerial",
    "Underwater": "fathomnet aquarium-qlnqy coral-lwptl underwater-objects-5v7p8 "
                  "underwater-pipes-4ng4t peixos-fish fish-market-ggjso",
    "Aerial and earth": "izembek aerial-cows aerial-pool aerial-spheres solar-panels-taxvb "
                        "sedimentary-features-9eosf",
    "Medical and microscopy": "bccd abdomen-mri acl-x-ray axial-mri bacteria-ptywi "
                              "bone-fracture-7fylg brain-tumor-m2pbp cavity-rs0uf cells-uyemf "
                              "gynecology-mri liver-disease mitosis-gjs3g parasites-1s07h phages "
                              "stomata-cells x-ray-rheumatology",
    "Documents and screens": "doclaynet activity-diagrams-qdobr document-parts signatures-xc8up "
                             "paper-parts tweeter-posts tweeter-profile number-ops digits-t2eg6",
    "Industry and electronics": "4-fold-defect asbestos cable-damage cables-nl42k cell-towers "
                                "corrosion-bi3q3 gauge-u2lwv printed-circuit-board "
                                "secondary-chains wall-damage circuit-elements circuit-voltages "
                                "radio-signal",
    "Games and video": "apex-videogame csgo-videogame farcry6-videogame "
                       "halo-infinite-angel-videogame team-fight-tactics avatar-recognition-nuexe "
                       "robomasters-285km",
}


def category(key: str) -> str:
    name = key[len("rf100-"):] if key.startswith("rf100-") else key
    return next((c for c, names in CATEGORIES.items() if name in names.split()), "Other")
