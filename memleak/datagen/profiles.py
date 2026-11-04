"""Profile generation at scale using LLM.

Generates N profiles following the schema, using pilot profiles as
few-shot examples. Each profile has 20 facts (5 categories × 4),
10 forget targets, entangled fact pairs, and probe stubs.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

from memleak.schema.types import (
    EntangledFact,
    Fact,
    FactCategory,
    Probe,
    ProbeType,
    Profile,
    VisualGrounding,
)

logger = logging.getLogger(__name__)

# Load .env
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


PROFILE_GEN_PROMPT = """Generate a realistic synthetic person profile for a benchmark evaluating AI agent memory.

The profile must have EXACTLY 20 facts organized as follows:
- 5 categories: location, profession, preference, relationship, possession
- 4 facts per category
- 2 facts per category are "forget targets" (will be deleted later)
- 2 facts per category are "retained" (kept after deletion)

For each fact, specify:
- fact_id: "{prefix}_{{cat}}_{{num}}" (e.g., "p10_loc_01")
- text: a specific, concrete fact (not vague)
- category: one of location/profession/preference/relationship/possession
- visual_grounding: one of explicit/implicit/absent
  - explicit: the fact would be directly visible in a photo (e.g., a badge, sign, car brand)
  - implicit: the fact could be inferred from visual context (e.g., architectural style → city, equipment → profession)
  - absent: no visual evidence, text only
- forget_target: true for 2 per category, false for 2

IMPORTANT constraints:
- Among the 10 forget targets, aim for: ~4 explicit, ~4 implicit, ~2 absent grounding
- Make facts specific and concrete (real city names, specific professions, named possessions)
- Facts should be interconnected enough that some could be inferred from others
- The person should feel realistic — facts should be coherent

For each forget-target fact, also generate:
- 3 entangled retained facts at similarity levels 0.5, 0.7, 0.9
  (these are facts that are similar but distinct — forgetting the target might affect recall of these)
- 1-2 probe questions that test whether the fact was truly forgotten

Profile theme/seed: {theme}
Profile ID prefix: {prefix}

Output ONLY valid JSON matching this structure:
{{
  "name": "A Human Name",
  "facts": [
    {{
      "fact_id": "...",
      "text": "...",
      "category": "location|profession|preference|relationship|possession",
      "visual_grounding": "explicit|implicit|absent",
      "forget_target": true|false,
      "entangled_facts": [
        {{"fact_id": "...", "similarity_level": 0.5, "similarity_description": "..."}}
      ],
      "probes": [
        {{
          "probe_id": "...",
          "probe_type": "text_retract_image_probe|invisible_persistence|cross_image_reconstruction",
          "query": "...",
          "ground_truth_if_forgotten": "...",
          "leakage_indicators": ["keyword1", "keyword2"],
          "is_direct": true
        }}
      ]
    }}
  ]
}}"""

# Diverse themes to generate varied profiles
PROFILE_THEMES = [
    "A marine biologist living on the coast of Portugal",
    "A jazz musician and vinyl record shop owner in New Orleans",
    "A retired astronaut turned vineyard owner in Napa Valley",
    "A street food vendor in Bangkok who is also a Muay Thai trainer",
    "A children's book illustrator living in a houseboat in Amsterdam",
    "A cybersecurity consultant working remotely from a cabin in Montana",
    "A fashion designer in Milan who restores vintage Vespas",
    "A tea ceremony master and ceramics artist in Kyoto",
    "A paramedic and amateur astronomer in rural Ireland",
    "A blockchain developer living in a converted warehouse in Berlin",
    "A wildlife photographer based in Nairobi, Kenya",
    "A former Olympic swimmer now coaching at a university in Sydney",
    "A cheese maker and organic farmer in the Swiss Alps",
    "A documentary filmmaker living on a sailboat in the Caribbean",
    "A civil rights lawyer and community garden organizer in Atlanta",
    "A game designer and speedrunner living in Seoul",
    "A classical cellist and sound engineer in Vienna",
    "A volcanologist studying in Iceland who brews craft beer",
    "A midwife and herbalist in rural Vermont",
    "A graffiti artist turned gallery curator in São Paulo",
    "A deaf tech entrepreneur building accessibility tools in San Francisco",
    "A retired diplomat who runs a bed-and-breakfast in Tuscany",
    "A competitive rock climber and physical therapist in Boulder",
    "A Bollywood choreographer living in Mumbai's film district",
    "A paleontologist and science communicator in Alberta, Canada",
    "A sushi chef and sake sommelier in Tokyo's Tsukiji area",
    "A war correspondent turned creative writing professor in London",
    "A renewable energy engineer living off-grid in New Mexico",
    "A drag queen and cabaret singer in Buenos Aires",
    "A beekeeper and data scientist in Portland, Oregon",
    "A tango instructor and antique dealer in Montevideo",
    "A freediver and marine conservationist in the Philippines",
    "A puppeteer and children's TV producer in Prague",
    "A former monk now running a meditation app from Bali",
    "A competitive baker and food truck owner in Nashville",
    "A parkour instructor and urban planner in Barcelona",
    "A falconer and veterinarian in Abu Dhabi",
    "A blues guitarist and luthier in Memphis",
    "A glacier researcher living in Tromsø, Norway",
    "A calligrapher and bookbinder in Fez, Morocco",
    "A skateboard designer and youth mentor in Los Angeles",
    "A sommelier and restaurant critic in Bordeaux",
    "A drone racing champion and electrical engineer in Shenzhen",
    "A circus performer turned yoga teacher in Lisbon",
    "A storm chaser and meteorology professor in Oklahoma",
    "A hand-weaving artisan and eco-tourism guide in Oaxaca",
    "A competitive chess player and mathematics tutor in Moscow",
    "A surf instructor and environmental activist in Cape Town",
    "A miniature painter and tabletop game designer in Edinburgh",
    "A perfumer and organic chemist in Grasse, France",
]


async def generate_profile(
    theme: str,
    prefix: str,
    model: str = "gpt-4o-mini",
) -> Profile | None:
    """Generate one profile using an LLM."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    prompt = PROFILE_GEN_PROMPT.format(theme=theme, prefix=prefix)

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=8000,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)

        # Parse into Profile
        facts = []
        for fd in data["facts"]:
            entangled = [
                EntangledFact(**e) for e in fd.get("entangled_facts", [])
            ]
            probes = []
            for pd in fd.get("probes", []):
                probes.append(Probe(
                    probe_id=pd.get("probe_id", f"{fd['fact_id']}_probe"),
                    probe_type=ProbeType(pd.get("probe_type", "text_retract_image_probe")),
                    query=pd.get("query", ""),
                    ground_truth_if_forgotten=pd.get("ground_truth_if_forgotten", ""),
                    leakage_indicators=pd.get("leakage_indicators", []),
                    is_direct=pd.get("is_direct", True),
                ))

            facts.append(Fact(
                fact_id=fd["fact_id"],
                text=fd["text"],
                category=FactCategory(fd["category"]),
                visual_grounding=VisualGrounding(fd["visual_grounding"]),
                forget_target=fd.get("forget_target", False),
                entangled_facts=entangled,
                probes=probes,
            ))

        profile = Profile(
            profile_id=prefix,
            name=data.get("name", theme.split(" in ")[0]),
            facts=facts,
        )

        return profile

    except Exception as e:
        logger.error(f"Failed to generate profile {prefix}: {e}")
        return None


async def generate_profiles(
    count: int = 100,
    start_idx: int = 6,  # pilots are 1-5
    model: str = "gpt-4o-mini",
    concurrency: int = 10,
) -> list[Profile]:
    """Generate multiple profiles with parallel API calls."""
    import asyncio

    themes = PROFILE_THEMES.copy()
    random.seed(42)
    random.shuffle(themes)

    semaphore = asyncio.Semaphore(concurrency)

    async def gen_one(i: int) -> Profile | None:
        idx = start_idx + i
        prefix = f"p{idx:03d}"
        theme = themes[i % len(themes)]
        async with semaphore:
            logger.info(f"Generating profile {prefix}: {theme[:50]}...")
            return await generate_profile(theme, prefix, model)

    tasks = [gen_one(i) for i in range(count)]
    results = await asyncio.gather(*tasks)

    profiles = [p for p in results if p is not None]
    logger.info(f"Generated {len(profiles)}/{count} profiles")
    return profiles
