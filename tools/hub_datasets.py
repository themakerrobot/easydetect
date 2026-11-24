# Apache-2.0
"""What easydetect-hub says about each dataset it carries: name, licence,
official page, the credit its licence asks for, and the paper to cite.

Public datasets whose licences allow sharing models trained on them; each
keeps its own terms (some pictures stay under third-party terms, such as
Flickr's), shown beside its models. Each licence was read on the dataset's
official page (Nov 2026).
"""

from __future__ import annotations

import re

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
    "coco-text": {
        "title": "COCO-Text",
        "license": "CC BY 4.0 (annotations); pictures from COCO, under the Flickr terms",
        "source": "https://bgshih.github.io/cocotext/",
        "attribution": "COCO-Text annotations, CC BY 4.0; pictures from COCO (Flickr terms).",
        "citation": "A. Veit et al. COCO-Text: Dataset and Benchmark for Text Detection and "
                    "Recognition in Natural Images. arXiv:1601.07140, 2016.",
        "note": "Every word is boxed, legible or not, as one class: text.",
    },
    "ena24": {
        "title": "ENA24 camera traps",
        "license": "CDLA-Permissive-1.0",
        "source": "https://lila.science/datasets/ena24detection",
        "attribution": "ENA24-detection (LILA BC), CDLA-Permissive-1.0.",
        "citation": "H. Yousif, R. Kays, Z. He. Dynamic Programming Selection of Object "
                    "Proposals for Sequence-Level Animal Species Classification in the Wild. "
                    "IEEE Transactions on Circuits and Systems for Video Technology, 2019.",
    },
    "boxes-on-bees": {
        "title": "Boxes on Bees and Pollen",
        "license": "CDLA-Permissive-1.0",
        "source": "https://lila.science/datasets/boxes-on-bees-and-pollen/",
        "attribution": "Boxes on Bees and Pollen (LILA BC), CDLA-Permissive-1.0.",
        "citation": "Noninvasive bee tracking in videos: deep learning algorithms and cloud "
                    "platform design specifications. Dataset, 2021.",
    },
    "bbbc038": {
        "title": "BBBC038 nuclei (Data Science Bowl 2018)",
        "license": "CC0 1.0",
        "source": "https://bbbc.broadinstitute.org/BBBC038",
        "attribution": "BBBC038v1 from the Broad Bioimage Benchmark Collection, CC0 1.0.",
        "citation": "J. C. Caicedo et al. Nucleus segmentation across imaging experiments: the "
                    "2018 Data Science Bowl. Nature Methods 16, 2019.",
    },
    "nuinsseg": {
        "title": "NuInsSeg (H&E nuclei)",
        "license": "CC BY 4.0",
        "source": "https://zenodo.org/records/10518968",
        "attribution": "NuInsSeg by A. Mahbod et al., CC BY 4.0.",
        "citation": "A. Mahbod et al. NuInsSeg: A fully annotated dataset for nuclei instance "
                    "segmentation in H&E-stained histological images. Scientific Data 11, 2024.",
    },
    "pped": {
        "title": "PPED (protective equipment)",
        "license": "CC BY 4.0",
        "source": "https://zenodo.org/records/6551758",
        "attribution": "Personal Protective Equipment Dataset (PPED), CC BY 4.0.",
        "citation": "Personal Protective Equipment Dataset (PPED). Zenodo, 2022.  "
                    "doi:10.5281/zenodo.6551758.",
        "note": "Photographed by its authors in a chemical plant.",
    },
    "loco": {
        "title": "LOCO (logistics objects)",
        "license": "CC0 1.0",
        "source": "https://github.com/tum-fml/loco",
        "attribution": "LOCO, Chair of Materials Handling, Material Flow, Logistics, TU Munich,  "
                       "CC0 1.0.",
        "citation": "C. Mayershofer, D.-M. Holm, B. Molter, J. Fottner. LOCO: Logistics Objects  "
                    "in Context. ICMLA 2020.",
    },
    "msu-apples": {
        "title": "MSU apples",
        "license": "CC BY 4.0",
        "source": "https://zenodo.org/records/7925397",
        "attribution": "MSUAppleData (Michigan State University), CC BY 4.0.",
        "citation": "MSUAppleData: an apple dataset with 'Gala' and 'Blondee' two varieties.  "
                    "Zenodo, 2023. doi:10.5281/zenodo.7925397.",
    },
    "swinepose": {
        "title": "SwinePose (pigs)",
        "license": "CC BY 4.0",
        "source": "https://zenodo.org/records/24221533",
        "attribution": "SwinePose v2.0, CC BY 4.0.",
        "citation": "SwinePose: A Lateral-View Benchmark Dataset for Pig Pose Estimation and  "
                    "Gait Analysis. Zenodo, 2026. doi:10.5281/zenodo.19358699.",
        "note": "Keypoints: SwinePose's 15 points a pig; the lines between them are  "
                "easydetect's.",
    },
    "bbbc039": {
        "title": "BBBC039 nuclei (fluorescence)",
        "license": "CC0 1.0",
        "source": "https://bbbc.broadinstitute.org/BBBC039",
        "attribution": "BBBC039v1 from the Broad Bioimage Benchmark Collection, CC0 1.0.",
        "citation": "J. C. Caicedo et al. Evaluation of Deep Learning Strategies for Nucleus  "
                    "Segmentation in Fluorescence Images. Cytometry Part A, 2019.",
    },
    "artf-clothes": {
        "title": "aRTF Clothes",
        "license": "CC BY 4.0",
        "source": "https://zenodo.org/records/14501499",
        "attribution": "aRTF Clothes dataset (Ghent University), CC BY 4.0.",
        "citation": "T. Lips, V.-L. De Gusseme, F. wyffels. Learning Keypoints for Robotic Cloth  "
                    "Manipulation using Synthetic Data. IEEE RA-L, 2024.",
    },
    "visa": {
        "title": "VisA (product defects)",
        "license": "CC BY 4.0",
        "source": "https://github.com/amazon-science/spot-diff",
        "attribution": "Visual Anomaly (VisA) dataset, Amazon, CC BY 4.0.",
        "citation": "Y. Zou et al. SPot-the-Difference Self-Supervised Pre-training for Anomaly  "
                    "Detection and Segmentation. ECCV 2022.",
        "note": "One class, defect: each defect region of its masks a box and an outline.",
    },
    "noaa-psnf": {
        "title": "NOAA Puget Sound nearshore fish",
        "license": "CDLA-Permissive-1.0",
        "source": "https://lila.science/datasets/noaa-puget-sound-nearshore-fish",
        "attribution": "NOAA Puget Sound Nearshore Fish (LILA BC), CDLA-Permissive-1.0.",
        "citation": "NOAA Fisheries. Puget Sound Nearshore Fish dataset. LILA BC, 2023.",
    },
    "armbench": {
        "title": "ARMBench (products in totes)",
        "license": "CC BY 4.0",
        "source": "https://huggingface.co/datasets/amazon/armbench-segmentation",
        "attribution": "ARMBench object segmentation, Amazon, CC BY 4.0.",
        "citation": "C. Mitash et al. ARMBench: An Object-centric Benchmark Dataset for Robotic  "
                    "Manipulation. ICRA 2023. arXiv:2303.16382.",
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


#: RF100-VL's own categories, as the hub groups them
RF100VL_CATEGORIES = {"Flora/Fauna": "Animals and environment",
                      "Industrial": "Industry and electronics",
                      "Lab Imaging": "Medical and microscopy", "Aerial": "Aerial and earth",
                      "Document": "Documents and screens", "Sport": "People, safety and traffic",
                      "Misc": "Other"}
RF100VL = {
    "license": "Apache-2.0 (as Roboflow100-VL's authors release it)",
    "citation": "P. Robicheaux et al. Roboflow100-VL: A Multi-Domain Object Detection Benchmark "
                "for Vision-Language Models. arXiv:2505.20612, 2025.",
}


def _rf100vl_name(key: str) -> str:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hub_data import RF100VL_SETS

    return RF100VL_SETS[key]


def lookup(key: str) -> dict:
    """The entry for ``key``; ``rf100-<name>`` for any Roboflow 100 dataset,
    ``rf100vl-<name>`` for an RF100-VL one."""
    if key in DATASETS:
        return dict(DATASETS[key])
    if key.startswith("rf100vl-"):
        name = _rf100vl_name(key)
        pretty = re.sub(r"[-_]+", " ", name).strip().capitalize()
        return {"title": f"{pretty} (RF100-VL)", "license": RF100VL["license"],
                "source": "https://github.com/roboflow/rf100-vl",
                "attribution": f"{name} from Roboflow100-VL (Roboflow Universe), Apache-2.0 as "
                               "its authors release it.",
                "citation": RF100VL["citation"]}
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
    # first on the page: a few clear, everyday classes, quick to show what a
    # detector does (a webcam and hands, cards, coins, animals, furniture);
    # full keys, listed ahead of their own category
    "Getting started": "rf100-furniture-ngpea rf100-animals-ij5d2 rf100-hand-gestures-jps7z  "
                       "rf100-uno-deck rf100-chess-pieces-mjzgj rf100-coins-1apki  "
                       "rf100-currency-v4f8j rf100-digits-t2eg6 rf100-soda-bottles  "
                       "rf100-aquarium-qlnqy rf100-road-signs-6ih4y rf100-mask-wearing-608pr  "
                       "rf100-apples-fvpl5 rf100-poker-cards-cxcvz rf100vl-fruitjes  "
                       "rf100vl-ball rf100vl-the-dreidel-project rf100vl-mahjong msu-apples  "
                       "artf-clothes",
    "General objects": "coco openimages lvis",
    "Everyday objects": "fashionpedia chess-pieces-mjzgj coins-1apki currency-v4f8j  "
                        "furniture-ngpea marbles poker-cards-cxcvz soda-bottles uno-deck  "
                        "valentines-chocolate washroom-rf1fa wine-labels pills-sxdht  "
                        "lettuce-pallets",
    "People, safety and traffic": "pped construction-ppe construction-safety-gsnvb  "
                                  "mask-wearing-608pr hand-gestures-jps7z sign-language-sokdr  "
                                  "people-in-paintings soccer-players-5fuqs  "
                                  "thermal-dogs-and-people-x6ejw smoke-uvylj road-signs-6ih4y  "
                                  "road-traffic vehicles-q0x2v street-work truck-movement  "
                                  "excavators-czvg9",
    "Animals and environment": "swinepose ap10k ena24 boxes-on-bees caltech-camera-traps  "
                               "animals-ij5d2 bees-jt5in insects-mytwu pests-2xlvx  "
                               "thermal-cheetah-my4dp trail-camera shark-teeth-5atku  "
                               "cloud-types taco",
    "Agriculture and plants": "gwhd plantdoc apples-fvpl5 cotton-20xz5 cotton-plant-disease  "
                              "grass-weeds leaf-disease-nsdsr peanuts-sd4kf weed-crop-aerial",
    "Underwater": "noaa-psnf fathomnet aquarium-qlnqy coral-lwptl underwater-objects-5v7p8  "
                  "underwater-pipes-4ng4t peixos-fish fish-market-ggjso",
    "Aerial and earth": "izembek aerial-cows aerial-pool aerial-spheres  "
                        "solar-panels-taxvb sedimentary-features-9eosf",
    "Medical and microscopy": "bbbc039 bbbc038 nuinsseg bccd abdomen-mri acl-x-ray axial-mri  "
                              "bacteria-ptywi bone-fracture-7fylg brain-tumor-m2pbp  "
                              "cavity-rs0uf cells-uyemf gynecology-mri liver-disease  "
                              "mitosis-gjs3g parasites-1s07h phages stomata-cells  "
                              "x-ray-rheumatology",
    "Documents and screens": "coco-text doclaynet activity-diagrams-qdobr document-parts  "
                             "signatures-xc8up paper-parts tweeter-posts tweeter-profile  "
                             "number-ops digits-t2eg6",
    "Industry and electronics": "loco armbench visa 4-fold-defect asbestos cable-damage  "
                                "cables-nl42k cell-towers corrosion-bi3q3 gauge-u2lwv  "
                                "printed-circuit-board secondary-chains wall-damage  "
                                "circuit-elements circuit-voltages radio-signal",
    "Games and video": "apex-videogame csgo-videogame farcry6-videogame  "
                       "halo-infinite-angel-videogame team-fight-tactics  "
                       "avatar-recognition-nuexe robomasters-285km",
}


def category(key: str) -> str:
    if key in CATEGORIES["Getting started"].split():
        return "Getting started"
    if key.startswith("rf100vl-"):
        import json
        from pathlib import Path

        cats = json.loads((Path(__file__).resolve().parent / "rf100vl.json").read_text())
        return RF100VL_CATEGORIES.get(cats["category"].get(_rf100vl_name(key)), "Other")
    name = key[len("rf100-"):] if key.startswith("rf100-") else key
    return next((c for c, names in CATEGORIES.items() if name in names.split()), "Other")
