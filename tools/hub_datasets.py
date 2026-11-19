# Apache-2.0
"""What easydetect-hub says about each dataset it carries: name, licence,
official page, the credit its licence asks for, and the paper to cite.

Only datasets whose terms allow commercial use and sharing trained models are
listed. Each licence was read on the dataset's official page (Nov 2026).
"""

from __future__ import annotations

DATASETS = {
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
    "General objects": "openimages lvis",
    "Everyday objects": "chess-pieces-mjzgj coins-1apki currency-v4f8j furniture-ngpea marbles "
                        "poker-cards-cxcvz soda-bottles uno-deck valentines-chocolate "
                        "washroom-rf1fa wine-labels pills-sxdht lettuce-pallets",
    "People, safety and traffic": "construction-ppe construction-safety-gsnvb mask-wearing-608pr "
                                  "hand-gestures-jps7z sign-language-sokdr people-in-paintings "
                                  "soccer-players-5fuqs thermal-dogs-and-people-x6ejw smoke-uvylj "
                                  "road-signs-6ih4y road-traffic vehicles-q0x2v street-work "
                                  "truck-movement excavators-czvg9",
    "Animals and environment": "caltech-camera-traps animals-ij5d2 bees-jt5in insects-mytwu "
                               "pests-2xlvx thermal-cheetah-my4dp trail-camera shark-teeth-5atku "
                               "cloud-types taco",
    "Agriculture and plants": "gwhd plantdoc apples-fvpl5 cotton-20xz5 cotton-plant-disease "
                              "grass-weeds leaf-disease-nsdsr peanuts-sd4kf weed-crop-aerial",
    "Underwater": "aquarium-qlnqy coral-lwptl underwater-objects-5v7p8 underwater-pipes-4ng4t "
                  "peixos-fish fish-market-ggjso",
    "Aerial and earth": "aerial-cows aerial-pool aerial-spheres solar-panels-taxvb "
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
