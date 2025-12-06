import json
import os

class SkillOntology:
    
    def __init__(self, ontology_path):
        self.ontology = self._load_ontology(ontology_path)
        self.lookup_map = self._build_lookup_map()
        self.relationships = self.ontology.get("skill_relationships", {})

    def _load_ontology(self, path):
        if not os.path.exists(path):
            print(f"⚠️ Warning: Ontology file not found at {path}")
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_lookup_map(self):
        """
        Tạo một map để tra cứu nhanh từ Synonym -> Canonical Name.
        """
        lookup = {}
        skills_data = self.ontology.get("skills", {})
        
        for canonical_name, details in skills_data.items():
            # Map chính nó
            lookup[canonical_name.lower()] = details['canonical']
            # Map canonical name (lowercase)
            lookup[details['canonical'].lower()] = details['canonical']
            
            # Map synonyms
            for syn in details.get('synonyms', []):
                lookup[syn.lower()] = details['canonical']
        return lookup

    def normalize_skill(self, skill_name):
        """Chuẩn hóa tên kỹ năng. Nếu không có trong DB thì trả về gốc."""
        if not skill_name: return ""
        s_lower = skill_name.strip().lower()
        return self.lookup_map.get(s_lower, skill_name) 

    def get_related_skills(self, skill_name):
        """Lấy danh sách kỹ năng liên quan (cho gợi ý hoặc matching mở rộng)"""
        norm_skill = self.normalize_skill(skill_name)
        # Tìm trong definition
        for key, details in self.ontology.get("skills", {}).items():
            if details['canonical'] == norm_skill:
                return details.get('related_skills', [])
        return []

    def check_relationship(self, skill_a, skill_b):
        """
        Kiểm tra mối quan hệ giữa 2 skill:
        - 1.0: Trùng khớp (Exact match / Synonym)
        - 0.5: Bổ trợ (Complements) hoặc Thay thế (Alternatives)
        - 0.0: Không liên quan
        """
        a = self.normalize_skill(skill_a)
        b = self.normalize_skill(skill_b)

        if a.lower() == b.lower():
            return 1.0

        # Kiểm tra trong skill_relationships
        for pair in self.relationships.get('complements', []):
            p_norm = [self.normalize_skill(p) for p in pair]
            if a in p_norm and b in p_norm:
                return 0.6 

        # Alternatives
        for pair in self.relationships.get('alternatives', []):
            p_norm = [self.normalize_skill(p) for p in pair]
            if a in p_norm and b in p_norm:
                return 0.4 

        # Kiểm tra related_skills trong định nghĩa
        rels_a = self.get_related_skills(a)
        if b in rels_a: return 0.3
        
        return 0.0

# Singleton instance 
import config
ONTOLOGY_PATH = os.path.join(config.BASE_DIR, "it_skills_ontology.json")
skill_ontology = SkillOntology(ONTOLOGY_PATH)