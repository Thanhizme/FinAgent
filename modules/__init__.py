"""
FinAgent Modules Package
------------------------
Exposes the four core pipeline modules:
  - collector   : Data acquisition from external APIs and sources
  - processor   : Data cleaning, normalisation, and feature engineering
  - visualizer  : Chart generation (trend, heatmap, distribution, rolling stats)
  - ai_agent    : LLM-powered natural language analysis
"""

from .collector import DataCollector
from .processor import DataProcessor
from .visualizer import DataVisualizer
from .ai_agent import AIAgent

__all__ = ["DataCollector", "DataProcessor", "DataVisualizer", "AIAgent"]

