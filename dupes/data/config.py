# config.py
"""
Configuration for data cleaning:
- manufacturer mapping
- column ordering
- product name cleanup patterns
"""

# Map normalized manufacturer names -> canonical name (still lowercase)
MANUFACTURER_MAP = {
    # L'Oreal brand variants
    "loreal": "loreal",
    "loreal paris": "loreal",
    "loreal professionnel": "loreal professionnel",
    "loreal professionnel paris": "loreal professionnel",
    "loreal or├®al": "loreal",   # safety in case of some weird decode remnants
    "loreal or├ëal": "loreal",

    # Vichy / La Roche-Posay
    "vichy": "vichy",
    "vichy france cai/caf 03 vichy france": "vichy",
    "la roche-posay laboratoire dermatologique cai": "la roche-posay",

    # Redken
    "redken": "redken",

    # Whitman
    "whitman lbs nv": "whitman lbs nv",

    # Wella
    "wella germany gmbh": "wella",
    "wella prestige sp, slu.": "wella",
    "wella professionals": "wella",

    # Beiersdorf
    "beiersdorf ag": "beiersdorf",
    "beiersdorf, s.a": "beiersdorf",

    # Apivita
    "apivita s.a.": "apivita",

    # Cocunat
    "cocunat s.l.": "cocunat",

    # Natura Siberica / Oblepikha / Natura Europa
    "natura siberica": "natura siberica",
    "oblepikha siberica": "natura siberica",
    "natura europa sas": "natura siberica",

    # Instituto Español
    "instituto espanol": "instituto espanol",

    # Beautyge
    "beautyge, s.l.": "beautyge",
    "beautyge sl": "beautyge",
    "beautyge s.l.": "beautyge",

    # Procter & Gamble
    "procter & gamble espana s.a.u.": "procter & gamble",

    # Rituals
    "rituals cosmetics": "rituals cosmetics",

    # Colors Productos...
    "colors productos de peluqueria y estetica sl.": "colors productos de peluqueria y estetica",

    # Lixone
    "lixone sl": "lixone",

    # Hairlust
    "hairlust": "hairlust",

    # Alfaparf
    "alfaparf group espana s.l.": "alfaparf",

    # New Flag
    "new flag gmbh": "new flag gmbh",

    # Douglas
    "douglas cosmetics gmbh": "douglas cosmetics gmbh",

    # Camomilla
    "camomilla srl": "camomilla srl",

    # Labeau
    "labeau fragrances s.l": "labeau fragrances s.l",

    # Essie (L'Oreal-owned but keep as brand)
    "essie": "essie",

    # Kapalua
    "kapalua trading s.a.": "kapalua trading s.a.",

    # Garnier
    "garnier": "garnier",

    # Kerasilk / Goldwell / Gold Haircare / Kao
    "kerasilk": "kerasilk",
    "goldwell": "goldwell",
    "gold haarpflege aps": "gold haarpflege aps",
    "goldwell kerasilk": "kerasilk",
    "kao germany gmbh": "kao",
    "kao corporation s.a.": "kao",

    # AB Parfums / Collistar
    "ab parfums s.p.a.": "ab parfums s.p.a.",
    "ab parfums iberia s.a.": "ab parfums iberia s.a.",
    "collistar": "collistar",
    "collistar s.p.a.": "collistar",

    # Uriage
    "uriage": "uriage",

    # Rahua
    "rahua": "rahua",

    # Balmain
    "balmain hair couture": "balmain hair couture",

    # Marlies Möller
    "marlies mller": "marlies mller",
    "marlies m┬âller": "marlies mller",

    # abril et nature
    "abril et nature": "abril et nature",

    # HH Simonsen
    "hh simonsen": "hh simonsen",

    # René Furterer / Pierre Fabre
    "rene furterer": "rene furterer",
    "pierre fabre laboratories": "pierre fabre laboratories",

    # Arom
    "arom  s.a": "arom s.a",

    # BLISS NATURE
    "bliss nature s.l": "bliss nature s.l",

    # Henkel + Schwarzkopf + Gliss Kur
    "henkel ag & co. kgaa": "henkel",
    "henkel corporation": "henkel",
    "henkel iberica, s.a.": "henkel",
    "schwarzkopf professional": "schwarzkopf professional",
    "gliss kur": "gliss kur",

    # Kérastase
    "kerastase": "kerastase",

    # Eurobio Lab
    "eurobio lab o": "eurobio lab",

    # Biosilk
    "biosilk": "biosilk",

    # Nuggela & Sule
    "nuggela & sule": "nuggela & sule",

    # Cosmetrade
    "cosmetrade, s.l.": "cosmetrade",
    "cosmetrade s.l": "cosmetrade",

    # JOICO
    "joico": "joico",

    # Maria Nila
    "maria nila": "maria nila",

    # CHI
    "chi": "chi",

    # Macadamia Beauty
    "macadamia beauty llc": "macadamia beauty llc",

    # Laboratorios Phergal
    "laboratorios phergal s.a.u": "laboratorios phergal s.a.u",

    # Creme of Nature
    "creme of nature": "creme of nature",

    # C Faces
    "c faces handelsgesellschaft ohg": "c faces handelsgesellschaft ohg",

    # Curlsmith
    "curlsmith": "curlsmith",

    # KMS
    "kms": "kms",

    # CE.way Regulatory
    "ce.way regulatory consultants slovenia": "ce.way regulatory consultants slovenia",

    # SEB MAN
    "seb man": "seb man",

    # Ducray
    "ducray": "ducray",

    # Björn Axén
    "bjorn axen": "bjorn axen",

    # MOHI Hair Care
    "mohi hair care": "mohi hair care",

    # Coco & Eve
    "coco & eve": "coco & eve",

    # Ludovico Martelli
    "ludovico martelli spa": "ludovico martelli spa",

    # Barberino
    "barberino srl sb": "barberino srl sb",

    # ICCS
    "international cosmetics and chemical services, ltd. (iccs ltd)": "iccs ltd",

    # L.A. Schmitt
    "l.a. schmitt gmbh": "l.a. schmitt gmbh",

    # FEKKAI
    "fekkai": "fekkai",

    # Biorius
    "biorius srl": "biorius",
    "biorius sprl": "biorius",

    # ONETECH
    "onetech": "onetech",

    # Rosental Organics
    "rosental organics gmbh": "rosental organics gmbh",

    # Bullfrog
    "bullfrog": "bullfrog",

    # Klorane
    "klorane": "klorane",

    # Camille & Clémentine
    "camille & clementine": "camille & clementine",

    # Beurskens
    "beurskens": "beurskens",

    # David Mallett
    "david mallett haircare": "david mallett haircare",

    # Laboratoire Nuxe
    "laboratoire nuxe": "laboratoire nuxe",

    # Paudiet Productos Naturales
    "paudiet productos naturales": "paudiet productos naturales",

    # OR HAY
    "or hay sarl": "or hay sarl",

    # Freshly Cosmetics
    "freshly cosmetics": "freshly cosmetics",

    # Activist Beauty
    "activist beauty": "activist beauty",

    # Beauty Insight
    "beauty insight associates ltd.": "beauty insight associates ltd.",

    # Fleurance Nature
    "fleurance nature espana sl": "fleurance nature espana sl",

    # Naturtint
    "naturtint": "naturtint",

    # Johnson's Baby
    "johnson's baby": "johnson's baby",

    # Kativa
    "kativa": "kativa",

    # Lola Cosmetics
    "lola cosmetics": "lola cosmetics",

    # Bumble and bumble
    "bumble and bumble.": "bumble and bumble",

    # T-LAB PROFESSIONAL
    "t-lab professional": "t-lab professional",

    # Grow Gorgeous
    "grow gorgeous": "grow gorgeous",

    # Manic Panic
    "manic panic": "manic panic",

    # Phytorelax
    "phytorelax": "phytorelax",

    # Australian Bodycare
    "australian bodycare": "australian bodycare",

    # Dr. Organic
    "dr. organic": "dr. organic",

    # Hair Rituel by Sisley
    "hair rituel by sisley": "hair rituel by sisley",
}

# Final column order
ORDERED_COLS = [
    "product_id",
    "product_name",
    "manufacturer_name",
    "price_eur",
    "volume_ml",
    "color_de_cabello",
    "tipo_de_cabello",
    "propiedad",
    "description",
    "ingredients_text",
]

# Phrases to strip from product_name (middle tags)
PRODUCT_NAME_PATTERNS = [
    r"\s+Champú capilar\s+",   # real encoding
    r"\s+Champ├║ capilar\s+",  # mojibake safety, in case it leaks into df
]
 