import json
import os
import config

class SkillOntology:
    """
    Skill Ontology handler for IT skills.
    
    Provides:
    - Normalization of skill names (canonical + synonyms).
    - Lookup of related skills.
    - Relationship scoring between skills.
    """
    def __init__(self, ontology_path: str):
        """
        Initialize the ontology by loading data and building lookup maps.
        """
        self.ontology = self._load_ontology(ontology_path)
        self.lookup_map = self._build_lookup_map()
        self.relationships = self.ontology.get("skill_relationships", {})

    def _load_ontology(self, path: str) -> dict:
        """
        Load ontology JSON file.
        
        Args:
            path (str): Path to ontology file.
        
        Returns:
            dict: Ontology data or empty dict if not found.
        """
        if not os.path.exists(path):
            print(f"(X) Warning: Ontology file not found at {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_lookup_map(self) -> dict:
        """
        Build a lookup map for fast synonym → canonical name resolution.
        
        Returns:
            dict: Mapping of lowercase skill names to canonical names.
        """
        lookup = {}
        skills_data = self.ontology.get("skills", {})

        for canonical_name, details in skills_data.items():
            canonical = details["canonical"]

            # Map itself and canonical name
            lookup[canonical_name.lower()] = canonical
            lookup[canonical.lower()] = canonical

            # Map synonyms
            for synonym in details.get("synonyms", []):
                lookup[synonym.lower()] = canonical

        return lookup

    def normalize_skill(self, skill_name: str) -> str:
        """
        Normalize a skill name to its canonical form.
        
        Args:
            skill_name (str): Raw skill name.
        
        Returns:
            str: Canonical skill name or original if not found.
        """
        if not skill_name:
            return ""
        return self.lookup_map.get(skill_name.strip().lower(), skill_name)

    def get_related_skills(self, skill_name: str) -> list:
        """
        Get related skills for a given skill.
        
        Args:
            skill_name (str): Skill name to query.
        
        Returns:
            list: Related skills or empty list.
        """
        norm_skill = self.normalize_skill(skill_name)
        for _, details in self.ontology.get("skills", {}).items():
            if details["canonical"] == norm_skill:
                return details.get("related_skills", [])
        return []

    def check_relationship(self, skill_a: str, skill_b: str) -> float:
        """
        Check relationship score between two skills.
        
        Scoring:
            - 1.0: Exact match / synonym
            - 0.6: Complementary skills
            - 0.4: Alternative skills
            - 0.3: Related skills
            - 0.0: Unrelated
        
        Args:
            skill_a (str): First skill.
            skill_b (str): Second skill.
        
        Returns:
            float: Relationship score.
        """
        a = self.normalize_skill(skill_a)
        b = self.normalize_skill(skill_b)

        if a.lower() == b.lower():
            return 1.0

        # Complements
        for pair in self.relationships.get("complements", []):
            normalized_pair = [self.normalize_skill(p) for p in pair]
            if a in normalized_pair and b in normalized_pair:
                return 0.6

        # Alternatives
        for pair in self.relationships.get("alternatives", []):
            normalized_pair = [self.normalize_skill(p) for p in pair]
            if a in normalized_pair and b in normalized_pair:
                return 0.4

        # Related skills
        if b in self.get_related_skills(a):
            return 0.3

        return 0.0

# =========================== SINGLETON INSTANCE =========================== #
ONTOLOGY_PATH = os.path.join(config.BASE_DIR, "it_skills_ontology.json")
skill_ontology = SkillOntology(ONTOLOGY_PATH)