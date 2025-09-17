import pandas as pd
import numpy as np
from pathlib import Path
import random

# Path to dataset
DATA_PATH = Path("Day4_JobSalaryPrediction/data/jobs.csv")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Dictionary of synthetic descriptions: 5 variations per common job_title
desc_variations = {
    "Data Scientist": [
        "Utilize statistical modeling and machine learning to analyze large datasets and provide actionable insights for business growth.",
        "Design experiments, build predictive models, and collaborate with teams to optimize data-driven decision-making processes.",
        "Apply advanced analytics techniques to uncover patterns in data and develop algorithms for forecasting and classification.",
        "Work on data visualization, feature engineering, and model deployment to support product development and strategy.",
        "Conduct A/B testing, interpret complex data, and communicate findings to stakeholders using tools like Python and SQL."
    ],
    "Data Engineer": [
        "Build and maintain scalable data pipelines to ensure efficient data flow from sources to analytics platforms.",
        "Design ETL processes, optimize databases, and integrate data from multiple systems for real-time processing.",
        "Develop infrastructure for big data handling, including cloud-based storage and distributed computing frameworks.",
        "Collaborate with data scientists to create robust data architectures that support machine learning workflows.",
        "Monitor data quality, implement data governance, and automate workflows using tools like Apache Spark and AWS."
    ],
    "Data Analyst": [
        "Analyze business data to identify trends, create reports, and support strategic decisions with visualizations.",
        "Query databases, clean datasets, and perform exploratory analysis to uncover insights for operational improvements.",
        "Develop dashboards in Tableau or Power BI, track KPIs, and present findings to non-technical teams.",
        "Conduct ad-hoc analyses, forecast metrics, and assist in data-driven marketing or product strategies.",
        "Use SQL and Excel to process data, identify anomalies, and recommend actions based on performance metrics."
    ],
    "Machine Learning Engineer": [
        "Implement and deploy machine learning models into production environments for scalable applications.",
        "Optimize algorithms for performance, integrate ML systems with software architecture, and monitor model drift.",
        "Collaborate on feature selection, model training, and A/B testing using frameworks like TensorFlow and PyTorch.",
        "Build end-to-end ML pipelines, handle large-scale data, and ensure models are efficient and reliable.",
        "Develop custom solutions for computer vision or NLP tasks, focusing on deployment and maintenance."
    ],
    "Analytics Engineer": [
        "Bridge data engineering and analysis by building tools for data transformation and visualization.",
        "Create reliable data models, automate reporting, and ensure analytics platforms are user-friendly.",
        "Work with BI tools to design schemas, optimize queries, and support self-service analytics.",
        "Integrate data sources, maintain data warehouses, and collaborate on metrics definitions.",
        "Focus on data quality engineering, version control for analytics code, and scalable solutions."
    ],
    "Research Scientist": [
        "Conduct cutting-edge research in AI, publish findings, and develop novel algorithms for complex problems.",
        "Explore theoretical models, run simulations, and apply results to real-world applications.",
        "Collaborate on interdisciplinary projects, analyze experimental data, and innovate in fields like deep learning.",
        "Design research protocols, evaluate hypotheses, and contribute to academic or industry advancements.",
        "Focus on ethical AI, robustness testing, and translating research into practical prototypes."
    ],
    "Data Science Manager": [
        "Lead a team of data scientists, oversee project roadmaps, and align data initiatives with business goals.",
        "Manage resource allocation, mentor junior staff, and ensure high-quality deliverables in analytics projects.",
        "Develop data strategy, foster collaboration across departments, and drive adoption of ML solutions.",
        "Monitor team performance, handle stakeholder communications, and promote best practices in data ethics.",
        "Build scalable data teams, evaluate tools and technologies, and measure ROI on data projects."
    ],
    "AI Architect": [
        "Design high-level AI systems architecture, integrating components for enterprise-scale solutions.",
        "Evaluate technologies, define blueprints for AI platforms, and ensure security and scalability.",
        "Collaborate on system integration, optimize for performance, and guide implementation teams.",
        "Focus on cloud AI architectures, hybrid models, and compliance with industry standards.",
        "Lead proof-of-concept developments, assess risks, and evolve architectures for emerging AI trends."
    ],
    "AI Engineer": [
        "Build AI-powered applications, from prototype to production, using modern frameworks.",
        "Implement neural networks, handle data preprocessing, and optimize for edge or cloud deployment.",
        "Work on AI ethics, bias mitigation, and integration with existing software ecosystems.",
        "Develop chatbots, recommendation systems, or autonomous agents with a focus on user experience.",
        "Test AI models in real-world scenarios, iterate based on feedback, and maintain systems."
    ],
    "AI Scientist": [
        "Advance AI research, explore new methodologies, and apply findings to solve domain-specific challenges.",
        "Develop algorithms for generative AI, reinforcement learning, or natural language understanding.",
        "Collaborate on multi-agent systems, evaluate state-of-the-art models, and publish innovations.",
        "Focus on explainable AI, federated learning, and ethical considerations in algorithm design.",
        "Bridge theory and practice by prototyping AI solutions for industry problems."
    ]
    # Add more titles if needed based on your dataset's unique ones
}

# Function to assign varied description
def assign_description(row):
    title = row["job_title"]
    if title in desc_variations:
        base_desc = random.choice(desc_variations[title])
        level = row["experience_level"]
        if level == "EN":
            base_desc += " Entry-level position requiring basic knowledge and eagerness to learn."
        elif level == "MI":
            base_desc += " Mid-level role involving independent projects and team collaboration."
        elif level == "SE":
            base_desc += " Senior position leading initiatives and mentoring juniors."
        elif level == "EX":
            base_desc += " Executive role overseeing strategy and innovation."
        # Add random skill variation
        skills = ["Python", "SQL", "TensorFlow", "AWS", "Tableau", "Spark"]
        base_desc += f" Key skills: {', '.join(random.sample(skills, 3))}."
        return base_desc
    return "General data role involving analysis and modeling."

# Apply to dataframe
df["job_description"] = df.apply(assign_description, axis=1)

# Save updated CSV
df.to_csv(DATA_PATH, index=False)
print("Updated jobs.csv with varied synthetic job descriptions.")
print(df[["job_title", "experience_level", "job_description"]].head(10))