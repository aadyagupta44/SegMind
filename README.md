# SegMind

# Project Overview
SegMind is an intelligent customer segmentation and marketing strategy generator that combines unsupervised machine learning with large language models to deliver personalized marketing insights. The system automatically categorizes customers into distinct segments using K-Means clustering and generates tailored marketing strategies using AI.

 # Key Features

- **Automated Customer Segmentation**: K-Means clustering algorithm identifies 4 distinct customer segments
- **AI-Powered Strategy Generation**: Integration with HuggingFace SmolLM3-3B (["HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)") for personalized marketing recommendations
- **Interactive Web Interface**: User-friendly tool for real-time customer analysis
- **Multi-dimensional Analysis**: Considers 8 key customer attributes for comprehensive profiling
- **Actionable Insights**: Generates specific, implementable marketing strategies

 # Customer Segments
<img width="791" height="324" alt="image" src="https://github.com/user-attachments/assets/fcc2bf46-cc65-408d-a44d-14c4ece65151" />

# Data Features
The model analyzes customers based on 8 key dimensions:

- Demographics: Age, Gender
- Financial: Income, Spending Score
- Behavioral: Purchase Frequency, Membership Years
- Transactional: Last Purchase Amount, Preferred Category

 # Technology Stack
- **Machine Learning**: K-Means Clustering (scikit-learn)
- **AI Model**: HuggingFace SmolLM3-3B
- **Frontend**: HTML/CSS/JavaScript
- **Backend**: Python
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib/Plotly 

# Model Performance

- **Clustering Algorithm**: K-Means with k=4
- **Feature Engineering**: 8-dimensional customer profile
- **Segmentation Accuracy**: Validated through silhouette analysis and inertia metrics
- **Strategy Generation**: Context-aware AI responses with segment-specific recommendations 

# Generated Marketing Strategies
The AI generates comprehensive strategies including:

- **Loyalty Programs**: Tiered rewards based on customer value
- **Subscription Models**: Recurring revenue optimization
- **Personalized Recommendations**: Purchase history-driven suggestions
- **Targeted Advertising**: Platform-specific campaign strategies
- **Strategic Partnerships**: Brand collaboration opportunities
- **Value Propositions**: Segment-specific messaging
- **Influencer Campaigns**: Credibility-driven marketing
- **Referral Programs**: Network effect utilization
- **Interactive Content**: Engagement-focused materials
- **Data-Driven Campaigns**: Performance-optimized targeting

# Getting Started
Prerequisites
pip install scikit-learn pandas numpy transformers torch
-  Installation
  git clone https://github.com/aadyagupta44/segmind
  cd segmind
  pip install -r requirements.txt
- Usage
  python# Example usage
  from segmind import CustomerSegmentation
  Initialize the segmentation tool
   segmenter = CustomerSegmentation()
  Predict customer segment
   customer_data = {
    'age': 35,
    'gender': 'Male',
    'income': 90000,
    'spending_score': 85,
    'membership_years': 2,
    'purchase_frequency': 3,
    'preferred_category': 'Electronics',
    'last_purchase_amount': 1200
   }

  segment, strategy = segmenter.predict_and_generate_strategy(customer_data)
  print(f"Customer Segment: {segment}")
  print(f"Marketing Strategy: {strategy}")

# Project Impact

- **Business Value**: Enables targeted marketing with potential 25-40% improvement in conversion rates
- **Automation**: Reduces manual segmentation time from hours to seconds
- **Scalability**: Handles large customer datasets with consistent performance
- **Personalization**: Delivers individualized strategies for enhanced customer experience

 # Future Enhancements

 - Real-time customer scoring API
 - Advanced clustering algorithms (DBSCAN, Hierarchical)
 - Integration with CRM systems
 - A/B testing framework for strategy validation
 - Multi-language strategy generation
 - Advanced visualization dashboard

 # Contributing

- Fork the repository
- Create a feature branch (git checkout -b feature/AmazingFeature)
- Commit changes (git commit -m 'Add AmazingFeature')
- Push to branch (git push origin feature/AmazingFeature)
- Open a Pull Request


 # Contact
- Aadya Gupta - aadyagupta.ag01@gmail.com
- Project Link: https://github.com/aadyagupta44/segmind
