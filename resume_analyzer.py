import PyPDF2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import re
import joblib


class MLResumeAnalyzer:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.skills_dict = {
            'technical': [
                'python', 'java', 'c++', 'javascript', 'html', 'css', 'sql',
                'aws', 'docker', 'kubernetes', 'machine learning', 'react',
                'angular', 'node.js', 'git', 'agile', 'devops'
            ],
            'soft': [
                'leadership', 'communication', 'teamwork', 'problem solving',
                'analytical', 'project management', 'time management'
            ]
        }

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        return text

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
        return self.preprocess_text(text)

    def extract_features(self, text):
        features = {}
        for skill_type, skills in self.skills_dict.items():
            for skill in skills:
                count = len(re.findall(r'\b' + re.escape(skill) + r'\b', text))
                features[f"{skill_type}_{skill.replace(' ', '_')}"] = count

        education_terms = ['bachelor', 'master', 'phd', 'diploma']
        features['education_level'] = sum(text.count(term) for term in education_terms)
        features['experience_count'] = len(re.findall(r'\b(year|years|month|months)\b', text))
        features['project_count'] = len(re.findall(r'\b(project|projects)\b', text))
        return features

    def train_model(self, resumes_data):
        X = []
        y = []
        for text, score in resumes_data:
            features = self.extract_features(text)
            X.append(list(features.values()))
            y.append(score)
        X = np.array(X)
        y = np.array(y)
        self.classifier.fit(X, y)

    def analyze_resume(self, pdf_path):
        try:
            text = self.extract_text_from_pdf(pdf_path)
            features = self.extract_features(text)
            X = np.array(list(features.values())).reshape(1, -1)
            score = self.classifier.predict(X)[0]
            tech_skills = [skill for skill in self.skills_dict['technical']
                           if re.search(r'\b' + re.escape(skill) + r'\b', text)]
            soft_skills = [skill for skill in self.skills_dict['soft']
                           if re.search(r'\b' + re.escape(skill) + r'\b', text)]
            return {
                'overall_score': int(score),
                'technical_skills': tech_skills,
                'soft_skills': soft_skills,
                'metrics': {
                    'skill_count': len(tech_skills) + len(soft_skills),
                    'tech_skill_ratio': len(tech_skills) / len(self.skills_dict['technical']),
                    'soft_skill_ratio': len(soft_skills) / len(self.skills_dict['soft']),
                },
                'suggestions': self.generate_suggestions(features, tech_skills, soft_skills)
            }
        except Exception as e:
            return f"Error analyzing resume: {str(e)}"

    def generate_suggestions(self, features, tech_skills, soft_skills):
        suggestions = []
        if len(tech_skills) < 5:
            suggestions.append("Add more technical skills (aim for 5+)")
        if len(soft_skills) < 3:
            suggestions.append("Include more soft skills")
        if features.get('project_count', 0) < 2:
            suggestions.append("Add more project experiences")
        if features.get('education_level', 0) == 0:
            suggestions.append("Include education details")
        return suggestions
