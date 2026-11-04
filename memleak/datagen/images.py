"""Image generation pipeline.

Release note: this module ships with the OpenAI gpt-image-2 backend only. The 64 pilot images
in the dataset were rendered with a Qwen-Image endpoint that is not publicly available; that
backend (and the image-edit call that depended on it) is omitted from the released code.
PILOT_IMAGE_PROMPTS below are the prompts that were used for the pilot images.

Generates images for profile facts.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Load .env
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

# OpenAI settings
OPENAI_IMAGE_MODEL = "gpt-image-2"
OPENAI_IMAGE_QUALITY = "medium"


def _generate_openai(prompt: str, output_path: Path, size: str) -> Path:
    """Generate image via OpenAI gpt-image-2."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    response = client.images.generate(
        model=OPENAI_IMAGE_MODEL,
        prompt=prompt,
        n=1,
        size=size,
        quality=OPENAI_IMAGE_QUALITY,
        output_format="png",
    )

    img_b64 = response.data[0].b64_json
    output_path.write_bytes(base64.b64decode(img_b64))
    logger.info(f"Generated (OpenAI): {output_path}")
    return output_path


def generate_image(
    prompt: str,
    output_path: str | Path,
    size: str = "1024x1024",
    seed: int | None = None,
    negative_prompt: str = "",
) -> Path:
    """Generate an image from a text prompt.

    Args:
        prompt: What to generate.
        output_path: Where to save the image (JPEG/PNG).
        size: Image dimensions (WxH).
        seed: Unused by the OpenAI backend (kept for signature compatibility).
        negative_prompt: Unused by the OpenAI backend (kept for signature compatibility).

    Returns:
        Path to the saved image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return _generate_openai(prompt, output_path, size)


def edit_image(
    input_path: str | Path,
    prompt: str,
    output_path: str | Path,
    size: str = "1024x1024",
    seed: int | None = None,
) -> Path:
    """Not available in the released code: the edit backend used during development is not public."""
    raise NotImplementedError("edit_image depended on a non-public image endpoint and is omitted from the release")


# ---------------------------------------------------------------------------
# Pilot image generation: prompt library
# ---------------------------------------------------------------------------

# Map from PLACEHOLDER paths to generation prompts
PILOT_IMAGE_PROMPTS: dict[str, dict] = {
    # --- Pilot 1: The Parisian Chef ---
    "PLACEHOLDER:haussmann_apartment_interior_eiffel_tower_view": {
        "prompt": "Interior of a classic Parisian Haussmann apartment, tall windows with white curtains, herringbone parquet floor, ornate crown molding, view of the Eiffel Tower through the window in the distance, warm afternoon light, photorealistic",
        "negative": "text, labels, watermark, cartoon",
    },
    "PLACEHOLDER:eiffel_tower_balcony_view": {
        "prompt": "View from a Parisian apartment balcony with wrought iron railing, Eiffel Tower visible across rooftops, potted plants on the balcony, golden hour light, photorealistic",
        "negative": "text, labels, watermark",
    },
    "PLACEHOLDER:chef_whites_michelin_plaque": {
        "prompt": "Professional chef in white chef's coat and hat standing in a restaurant entrance next to a Michelin star plaque on the wall, proud expression, restaurant interior visible behind, photorealistic",
        "negative": "cartoon, watermark",
    },
    "PLACEHOLDER:professional_kitchen_action": {
        "prompt": "Professional chef cooking in a commercial kitchen, stainless steel counters, flames from a pan, copper pots hanging overhead, other kitchen staff in background, photorealistic, action shot",
        "negative": "text, watermark",
    },
    "PLACEHOLDER:culinary_diploma_wall": {
        "prompt": "Framed culinary school diploma hanging on a wall next to other certificates, the diploma has ornate borders and calligraphy, partially readable French text, warm office lighting, photorealistic",
        "negative": "cartoon, clear text that says specific school name",
    },
    "PLACEHOLDER:farmers_market_basket_organic": {
        "prompt": "Person carrying a woven basket full of fresh organic vegetables at an outdoor farmers market, colorful produce stalls in background, morning light, photorealistic",
        "negative": "text, labels, watermark",
    },
    "PLACEHOLDER:couple_photo_gallery_opening": {
        "prompt": "Couple at an art gallery opening event, dressed smartly, holding wine glasses, modern art on white walls behind them, warm gallery lighting, name badges visible on lanyards, photorealistic",
        "negative": "cartoon, watermark",
    },
    "PLACEHOLDER:two_men_vineyard_rows": {
        "prompt": "Two men walking between rows of grapevines in a vineyard, rolling hills in background, autumn colors on the leaves, casual clothing, warm golden hour light, photorealistic",
        "negative": "text, labels, watermark",
    },
    "PLACEHOLDER:green_citroen_ds_parked": {
        "prompt": "Dark green vintage Citroen DS car parked on a cobblestone street in a European city, the iconic aerodynamic shape clearly visible, chrome details, photorealistic",
        "negative": "watermark, cartoon",
    },
    "PLACEHOLDER:bookshelf_vintage_cookbooks": {
        "prompt": "Wooden bookshelf filled with vintage and antique cookbooks, worn leather spines, some books tilted, warm reading lamp nearby, cozy home library atmosphere, photorealistic",
        "negative": "text, specific titles, watermark",
    },
    "PLACEHOLDER:jardin_luxembourg_sign": {
        "prompt": "Entrance to the Jardin du Luxembourg in Paris with the green metal sign visible reading 'Jardin du Luxembourg', iron gates, tree-lined path beyond, sunny day, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:loire_wine_bottles_shelf": {
        "prompt": "Wine shelf with several bottles of natural wine, some with Loire Valley region labels visible, rustic wooden shelf, warm ambient lighting, photorealistic",
        "negative": "watermark, cartoon",
    },
    "PLACEHOLDER:seafood_restaurant_menu": {
        "prompt": "Restaurant menu on a table showing seafood dishes including lobster, shrimp, and oysters, elegant typography, white tablecloth, wine glass nearby, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:paris_metro_station": {
        "prompt": "Paris Metro station platform with the distinctive Art Nouveau entrance sign, tiled walls, people waiting for the train, underground lighting, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:french_cooking_class": {
        "prompt": "Cooking class in session in a French kitchen, students in aprons around a large counter, instructor demonstrating technique, copper pots and ingredients visible, warm lighting, photorealistic",
        "negative": "text, watermark",
    },
    "PLACEHOLDER:wine_tasting_group": {
        "prompt": "Group of friends at a wine tasting in a cellar, holding wine glasses, wooden barrels in background, stone walls, warm candlelight, photorealistic",
        "negative": "text, watermark",
    },

    # --- Pilot 2: The Stanford Parent ---
    "PLACEHOLDER:palo_alto_tree_lined_street": {
        "prompt": "Tree-lined residential street in Palo Alto California, Mediterranean style homes, palm trees and oaks, clear blue sky, well-maintained lawns, photorealistic",
        "negative": "text, street signs, watermark",
    },
    "PLACEHOLDER:startup_office_whiteboard_architecture": {
        "prompt": "Modern startup office with a large whiteboard covered in system architecture diagrams, boxes and arrows, technical drawings, standing desks nearby, casual tech office vibe, photorealistic",
        "negative": "readable company name, watermark",
    },
    "PLACEHOLDER:google_campus_badge_photo": {
        "prompt": "Person wearing a Google employee badge on a lanyard at the Google campus, the colorful Google logo visible on the badge, modern campus buildings in background, sunny day, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:trail_running_gps_watch": {
        "prompt": "Close-up of a runner's wrist with a GPS sports watch showing trail running statistics, distance and pace displayed on screen, trail path in background, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:young_woman_campus_palm_trees": {
        "prompt": "Young woman walking on a university campus with sandstone arches, palm trees, and a quad with green lawn, carrying a backpack, California sunshine, photorealistic",
        "negative": "text, university name, watermark",
    },
    "PLACEHOLDER:woman_white_coat_stanford_hospital": {
        "prompt": "Woman doctor in a white medical coat with a hospital ID badge, standing in a modern hospital hallway, stethoscope around neck, professional and confident, photorealistic",
        "negative": "watermark, cartoon",
    },
    "PLACEHOLDER:family_outdoor_no_phones": {
        "prompt": "Family of four hiking on a trail in nature, no phones or electronic devices visible, smiling and enjoying the outdoors, trees and mountains in background, photorealistic",
        "negative": "phones, screens, electronics, watermark",
    },
    "PLACEHOLDER:physics_textbook_on_desk": {
        "prompt": "Physics textbook open on a desk with handwritten notes, equations visible on the pages, coffee cup nearby, study lamp, university dorm room setting, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:bay_area_text_message": {
        "prompt": "Smartphone screen showing a text message conversation, one message mentions nice weather, casual friendly tone, phone held in hand, blurred background, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:startup_celebration_series_b": {
        "prompt": "Office celebration with champagne glasses raised, confetti, small team of people cheering in a modern startup office, casual dress, photorealistic",
        "negative": "text, dollar amounts, watermark",
    },
    "PLACEHOLDER:rivian_r1s_forest_green_driveway": {
        "prompt": "Forest green Rivian R1S electric SUV parked in a suburban driveway, modern house in background, the distinctive Rivian front design clearly visible, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:coupa_cafe_sign": {
        "prompt": "Exterior of a cozy cafe with a visible sign reading 'Coupa Cafe', outdoor seating with people having coffee, California street scene, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:home_office_desk": {
        "prompt": "Modern home office with a standing desk, large monitor, ergonomic chair, plants on windowsill, natural light from a large window, clean minimalist style, photorealistic",
        "negative": "watermark",
    },

    # --- Pilot 3: The Vegetarian Surgeon ---
    "PLACEHOLDER:brownstone_back_bay_street": {
        "prompt": "Row of classic Boston brownstone buildings on a tree-lined street in Back Bay, red brick facades, bay windows, iron railings, fall foliage, photorealistic",
        "negative": "text, street signs with specific names, watermark",
    },
    "PLACEHOLDER:surgeon_scrubs_hospital_corridor": {
        "prompt": "Surgeon in blue scrubs and surgical cap walking down a hospital corridor, ID badge on lanyard, confident stride, fluorescent hospital lighting, photorealistic",
        "negative": "readable name on badge, watermark",
    },
    "PLACEHOLDER:plant_based_meals_collage": {
        "prompt": "Overhead view of several plant-based meal plates arranged on a table, colorful vegetables, grains, legumes, no meat visible, beautiful food photography, photorealistic",
        "negative": "meat, fish, chicken, watermark",
    },
    "PLACEHOLDER:man_harvard_law_building": {
        "prompt": "Man in a blazer standing in front of a grand brick university building with white columns, ivy-covered walls, academic campus setting, autumn day, photorealistic",
        "negative": "text, university name visible, watermark",
    },
    "PLACEHOLDER:two_doctors_hospital_cafeteria": {
        "prompt": "Two doctors in scrubs having coffee in a hospital cafeteria, casual conversation, trays of food on the table, hospital cafeteria interior, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:yoga_studio_mat_bag": {
        "prompt": "Yoga mat bag with a studio logo leaning against a wall, water bottle nearby, wooden floor of a yoga studio visible, warm peaceful atmosphere, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:medical_journal_desk_cardiac": {
        "prompt": "Desk covered with open medical journals and research papers, cardiac surgery diagrams visible, reading glasses, coffee cup, academic office setting, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:cozy_cabin_snow_vermont": {
        "prompt": "Cozy wooden cabin in a snowy New England landscape, smoke from chimney, pine trees covered in snow, warm light glowing from windows, winter evening, photorealistic",
        "negative": "text, location signs, watermark",
    },
    "PLACEHOLDER:medical_first_editions_glass_case": {
        "prompt": "Glass display case containing rare antique medical books with leather bindings, gold-embossed titles, museum-quality presentation, warm library lighting, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:harvard_book_store_sign": {
        "prompt": "Storefront of an independent bookstore with a sign reading 'Harvard Book Store', books displayed in the window, brick building, Cambridge street scene, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:farmers_market_vegetables": {
        "prompt": "Colorful vegetable stall at a farmers market, fresh organic produce arranged in baskets, leafy greens, tomatoes, root vegetables, morning market atmosphere, photorealistic",
        "negative": "watermark",
    },

    # --- Pilot 4: The Tesla Owner ---
    "PLACEHOLDER:austin_east_side_loft_sign": {
        "prompt": "Modern loft building in East Austin with visible street address numbers, industrial-chic architecture, murals on nearby walls, urban Texas neighborhood, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:wework_standing_desk_congress": {
        "prompt": "Standing desk setup in a modern coworking space, large windows showing a city skyline, multiple monitors, plants, other workers in background, photorealistic",
        "negative": "company name, watermark",
    },
    "PLACEHOLDER:ux_wireframes_fintech_screen": {
        "prompt": "Computer screen showing UX wireframes and mockups for a financial app, colorful interface designs, design tools open, modern desk setup, photorealistic",
        "negative": "company name, watermark",
    },
    "PLACEHOLDER:apple_park_badge_selfie": {
        "prompt": "Person taking a selfie at a modern tech campus, wearing an employee badge with Apple logo visible on the lanyard, futuristic circular building in background, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:la_marzocca_linea_mini_kitchen": {
        "prompt": "Professional espresso machine on a kitchen counter, stainless steel with the La Marzocco brand visible, freshly pulled espresso shot, modern kitchen background, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:vinyl_shelf_turntable_jazz": {
        "prompt": "Record shelf filled with vinyl albums, a turntable playing on a wooden sideboard, jazz album covers visible, warm ambient living room lighting, photorealistic",
        "negative": "text labels, watermark",
    },
    "PLACEHOLDER:couple_music_studio": {
        "prompt": "Two people in a small music studio, mixing board and monitors visible, acoustic foam on walls, creative casual atmosphere, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:man_spacex_facility_badge": {
        "prompt": "Man wearing a SpaceX employee badge standing near an industrial facility entrance, rocket or spacecraft hardware visible in background, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:matte_black_tesla_model3_driveway": {
        "prompt": "Matte black Tesla Model 3 parked in a residential driveway, the Tesla logo and Model 3 badge visible, modern house in background, clean and sharp, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:desk_macbook_ipad_apple_devices": {
        "prompt": "Clean desk setup with multiple Apple devices - a MacBook Pro laptop open, iPad Pro with Apple Pencil, and AirPods case, minimalist desk, photorealistic",
        "negative": "brand labels clearly readable, watermark",
    },
    "PLACEHOLDER:veracruz_taco_truck": {
        "prompt": "Food truck with colorful Mexican-style decorations serving tacos, customers waiting in line, outdoor picnic tables, Austin Texas street food scene, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:record_store_browsing": {
        "prompt": "Person browsing through vinyl records in an independent record store, shelves of albums, warm overhead lighting, music posters on walls, photorealistic",
        "negative": "watermark",
    },

    # --- Pilot 5: The Remote Worker ---
    "PLACEHOLDER:craftsman_house_rainy_street": {
        "prompt": "Craftsman style house on a rainy street in the Pacific Northwest, lush green vegetation, wet sidewalk, overcast sky, warm porch light on, photorealistic",
        "negative": "text, location signs, watermark",
    },
    "PLACEHOLDER:home_office_bloomberg_terminal": {
        "prompt": "Home office with a Bloomberg terminal on one monitor showing financial data and charts, second monitor with spreadsheets, professional desk setup, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:multi_monitor_financial_models": {
        "prompt": "Multiple computer monitors showing financial models, Python code with data analysis, stock charts and graphs, professional home office, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:mit_diploma_frame": {
        "prompt": "Framed PhD diploma from MIT on a wall, ornate frame, the MIT seal and 'Massachusetts Institute of Technology' text visible, warm office lighting, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:homemade_ramen_japanese_bowls": {
        "prompt": "Homemade ramen served in traditional Japanese ceramic bowls, rich broth, noodles, soft-boiled egg, nori, chopsticks resting on the bowl, steam rising, photorealistic food photography",
        "negative": "watermark",
    },
    "PLACEHOLDER:chess_tournament_clock_rating": {
        "prompt": "Chess tournament table with wooden chess pieces mid-game, chess clock showing time, rating card visible on the table, tournament hall setting, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:two_men_cooking_kitchen": {
        "prompt": "Two men cooking together in a well-equipped home kitchen, chopping vegetables, pots on the stove, warm kitchen lighting, casual and happy atmosphere, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:two_people_mit_reunion": {
        "prompt": "Two people posing together at a university reunion event, banner with university branding behind them, name tags on lanyards, formal-casual attire, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:houseboat_lake_union_dock": {
        "prompt": "Cabin-style houseboat docked on a calm lake with a city skyline in the distance, wooden dock, potted plants, peaceful waterfront scene, photorealistic",
        "negative": "text, location signs, watermark",
    },
    "PLACEHOLDER:signed_foundation_glass_case": {
        "prompt": "First edition book in a glass display case, old hardcover with gold lettering on the spine, a visible author signature on the title page, collector's display, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:storyville_coffee_cup": {
        "prompt": "Coffee cup with a café logo on it sitting on a wooden table, latte art visible, cozy coffee shop interior in background, warm morning light, photorealistic",
        "negative": "watermark",
    },
    "PLACEHOLDER:financial_newspaper_coffee": {
        "prompt": "Financial newspaper like the Financial Times or Wall Street Journal on a café table next to a coffee cup, stock market headlines visible, morning reading scene, photorealistic",
        "negative": "watermark",
    },
}
