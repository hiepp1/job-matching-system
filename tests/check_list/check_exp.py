import os
import json

cv_folder = "artifacts/json_cv"
print(f"Checking CVs in '{cv_folder}' for experience field...")
count = 0

if os.path.exists(cv_folder):
    for filename in os.listdir(cv_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(cv_folder, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    cv_data = json.load(f)
                    metrics = cv_data.get("metrics", {})
                    experience_years = metrics.get("years_experience", 0)

                    if experience_years > 0:
                        profile = cv_data.get('candidate_profile', {})
                        name = profile.get('name', 'Unknown')
                        print(f" - {filename}: {name} has {experience_years} years of experience.")
                        count += 1
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    print(f"\nTotal CVs with experience field found: {count}")
else:
    print(f"The folder '{cv_folder}' does not exist.")
