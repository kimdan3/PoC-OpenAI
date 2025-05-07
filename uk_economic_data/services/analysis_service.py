import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from functools import lru_cache, wraps
import asyncio
from openai import OpenAI
import os
import logging
from typing import Optional, Dict, List, Tuple
from uk_economic_data import UKEconomicDataFetcher
import json
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Constants
MARKETING_INSIGHTS_PROMPT = """
Product: {product}
Date of analysis: {date}
Customer segment: {segment}
Sales change rate: {change_rate:.1f}%

IMPORTANT CONTEXT CLARIFICATION:
- This analysis must focus **strictly on store-level observations**. Avoid any references to nationwide behavior or broad consumer psychology.
- Use the UK Background Context only to **support local interpretation**, never as a standalone explanation.
- The goal is to generate **clear, differentiated insights** grounded in product-specific performance and local customer behavior.

Generate insights using the following instructions:

1. 📌 Cause analysis:
   Write one cohesive paragraph that explains **why the product's sales declined in this specific store**.
   - Start from observed metrics and changes (e.g., product movement, stock turnover, display traffic)
   - Integrate product-specific dynamics (e.g., perishability, impulse vs planned purchase, cold chain issues, product format)
   - Use background context (e.g., weather, seasonal events) only when directly connected to in-store observations
   - If the reason is not clear, state: "Multiple product-level and environmental factors may be contributing to the decline." and focus on in-store hypotheses that can be tested or explored further.

2. 💡 Strategy suggestions:
   Propose **2 to 3 product-specific, executable strategies** that store staff could realistically apply. Each strategy should:
   - Be grounded in operational feasibility (consider product storage type, staff limits, retail fixtures)
   - Mention placement or layout **only if it's realistically changeable** (e.g., avoid saying "move frozen berries to checkout")
   - Consider sales uplift mechanisms like visual triggers, smart pricing, product bundling, or educational signage
   - Be measurable through before/after data or customer interaction

3. Language & Style Requirements:
   - Avoid vague expressions like "customers may be unaware of..." unless backed by a concrete store factor (e.g., hidden shelf)
   - Never refer to "UK shoppers" or "supermarket trends" — all analysis must relate to **this store's performance**
   - Avoid phrases like "this will help retailers..." — write as if you're describing a **change the store team can test today**

Use the following format:

📌 Cause analysis:
[Single paragraph with clear, grounded explanation of sales decline]

💡 Strategy suggestions:
[First strategy paragraph — specific, testable, actionable]
[Second strategy paragraph — different from the first]
[Optional third strategy — only if valuable, not repetitive]
"""

@dataclass
class ServiceConfig:
    """Configuration settings for the AnalysisService"""
    timeout_seconds: int = 30
    cache_size: int = 1000
    batch_size: int = 5
    api_key: Optional[str] = None
    cache_ttl: int = 3600  # 1 hour cache TTL
    
    @classmethod
    def from_env(cls):
        """Create configuration from environment variables"""
        return cls(
            timeout_seconds=int(os.getenv('TIMEOUT_SECONDS', 30)),
            cache_size=int(os.getenv('CACHE_SIZE', 1000)),
            batch_size=int(os.getenv('BATCH_SIZE', 5)),
            api_key=os.getenv('OPENAI_API_KEY'),
            cache_ttl=int(os.getenv('CACHE_TTL', 3600))
        )

class APIRateLimiter:
    """Rate limiter for API calls to prevent throttling"""
    
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
    
    async def acquire(self):
        """Acquire permission to make an API call"""
        now = time.time()
        self.requests = [req for req in self.requests if now - req < self.time_window]
        if len(self.requests) >= self.max_requests:
            await asyncio.sleep(self.time_window)
        self.requests.append(now)

class AnalysisServiceError(Exception):
    """Base exception for AnalysisService"""
    pass

class DataValidationError(AnalysisServiceError):
    """Raised when data validation fails"""
    pass

class UKDataFetchError(AnalysisServiceError):
    """Raised when UK economic data fetch fails"""
    pass

class APITimeoutError(AnalysisServiceError):
    """Raised when API calls timeout"""
    pass

class StructuredLogger:
    """Enhanced structured logging with better formatting and configuration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging with proper formatting"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _format_log(self, level: str, message: str, **kwargs) -> str:
        """Format log message with structured data"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            "service": "analysis_service",
            **kwargs
        }
        return json.dumps(log_data)
    
    def info(self, message: str, **kwargs):
        """Log info level message"""
        self.logger.info(self._format_log("INFO", message, **kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error level message"""
        self.logger.error(self._format_log("ERROR", message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning level message"""
        self.logger.warning(self._format_log("WARNING", message, **kwargs))

class AnalysisService:
    """Service class for analyzing retail sales data and generating marketing insights"""
    
    # Constants
    VALID_GENDERS = {'M', 'F'}
    VALID_PERIODS = {'daily', 'weekly', 'monthly'}
    MIN_AGE = 0
    MAX_AGE = 100
    
    # Required DataFrame columns
    REQUIRED_COLUMNS = ['date', 'product_name', 'age', 'gender', 'sales_amount']
    
    def __init__(self, config: Optional[ServiceConfig] = None):
        """Initialize the AnalysisService with configuration and dependencies"""
        self.config = config or ServiceConfig.from_env()
        
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required. Please set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.config.api_key)
        self.uk_data_fetcher = UKEconomicDataFetcher()
        self.logger = StructuredLogger()
        self.rate_limiter = APIRateLimiter(max_requests=5, time_window=1)
        self._insights_cache = {}
        
        # Validate API key
        self._validate_api_key()

    def _validate_api_key(self) -> None:
        """Validate OpenAI API key"""
        try:
            self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
        except Exception as e:
            raise ValueError(f"Invalid OpenAI API key: {str(e)}")

    def _validate_input_data(self, df: pd.DataFrame) -> bool:
        """
        Validate input DataFrame structure and data types
        
        Args:
            df: Input DataFrame to validate
            
        Returns:
            bool: True if validation passes
            
        Raises:
            DataValidationError: If validation fails
        """
        try:
            # Check required columns
            missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
            if missing_columns:
                raise DataValidationError(f"Missing required columns: {missing_columns}")
            
            # Validate data types
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                raise DataValidationError("Date column must be datetime type")
                
            if not pd.api.types.is_numeric_dtype(df['sales_amount']):
                raise DataValidationError("Sales amount must be numeric type")
                
            if not pd.api.types.is_numeric_dtype(df['age']):
                raise DataValidationError("Age must be numeric type")
            
            # Validate data ranges
            if not (self.MIN_AGE <= df['age'].min() <= df['age'].max() <= self.MAX_AGE):
                raise DataValidationError(f"Age must be between {self.MIN_AGE} and {self.MAX_AGE}")
                
            if not all(gender in self.VALID_GENDERS for gender in df['gender'].unique()):
                raise DataValidationError(f"Gender must be one of {self.VALID_GENDERS}")
                
            if (df['sales_amount'] < 0).any():
                raise DataValidationError("Sales amount cannot be negative")
                
            return True
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise DataValidationError(f"Data validation failed: {str(e)}")

    def clear_cache(self) -> None:
        """Clear the marketing insights cache"""
        self._insights_cache.clear()
        self.logger.info("Marketing insights cache cleared")

    async def generate_marketing_insights(
        self, 
        product: str, 
        date: str, 
        segment: str, 
        change_rate: float
    ) -> Tuple[str, List[str]]:
        """
        Generate marketing insights using OpenAI API with caching and UK economic context
        
        Args:
            product: Product name
            date: Analysis date
            segment: Customer segment
            change_rate: Sales change rate
            
        Returns:
            Tuple containing cause analysis and strategies
            
        Raises:
            UKDataFetchError: If UK economic data fetch fails
            APITimeoutError: If API call times out
        """
        cache_key = f"{product}_{date}_{segment}"
        if cache_key in self._insights_cache:
            cached_result, timestamp = self._insights_cache[cache_key]
            if time.time() - timestamp < self.config.cache_ttl:
                return cached_result
        
        try:
            background_info = await self.uk_data_fetcher.get_background_info(date)
        except Exception as e:
            self.logger.error(f"Failed to fetch UK economic data: {str(e)}")
            raise UKDataFetchError(f"Failed to fetch UK economic data: {str(e)}")
        
        prompt = MARKETING_INSIGHTS_PROMPT.format(
            product=product,
            date=date,
            segment=segment,
            change_rate=change_rate
        )

        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                await self.rate_limiter.acquire()
                response = self.client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": "You are a retail marketing expert specializing in UK supermarket operations and in-store marketing tactics."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=1000,
                    top_p=0.9,
                    frequency_penalty=0.3,
                    presence_penalty=0.3
                )
                
                content = response.choices[0].message.content
                result = self._parse_insights(content)
                
                # Cache the result
                self._insights_cache[cache_key] = (result, time.time())
                
                return result
                
        except asyncio.TimeoutError:
            raise APITimeoutError("OpenAI API call timed out")
        except Exception as e:
            self.logger.error(f"Error generating marketing insights: {e}")
            return "Unable to generate cause analysis.", ["Unable to generate strategies."]

    def _parse_insights(self, content: str) -> Tuple[str, List[str]]:
        """
        Parse insights from OpenAI response
        
        Args:
            content: Raw response content
            
        Returns:
            Tuple containing cause analysis and strategies
        """
        lines = content.split('\n')
        cause_analysis = ""
        strategies = []
        current_section = ""
        
        for line in lines:
            if "📌 Cause analysis" in line:
                current_section = "cause"
            elif "💡 Strategy suggestions" in line:
                current_section = "strategy"
            elif current_section == "cause" and line.strip() and not line.startswith("📌"):
                cause_analysis = line.strip()
            elif current_section == "strategy" and line.strip() and not line.startswith("💡"):
                strategies.append(line.strip())
        
        return cause_analysis, strategies

    async def get_top_products_with_insights(
        self,
        df: pd.DataFrame,
        gender: str,
        min_age: int = 18,
        max_age: int = 35,
        top_n: int = 5,
        threshold: float = 5.0,
        period: str = "weekly"
    ) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Return top N products with largest fluctuations and their marketing insights
        
        Args:
            df: Input DataFrame
            gender: Target gender ('M' or 'F')
            min_age: Minimum age
            max_age: Maximum age
            top_n: Number of top products to analyze
            threshold: Sales change threshold
            period: Analysis period
            
        Returns:
            Tuple containing list of products and their insights
            
        Raises:
            ValueError: If input parameters are invalid
            DataValidationError: If data validation fails
        """
        start_time = time.time()
        
        try:
            # Validate input data
            self._validate_input_data(df)
            
            # Validate parameters
            self._validate_parameters(gender, min_age, max_age, period)
            
            # Filter and process data
            filtered_df = self._filter_data(df, gender, min_age, max_age)
            if filtered_df.empty:
                self.logger.warning("No data available after filtering")
                return [], {}
            
            # Get top products
            products = self._get_top_products(filtered_df, top_n)
            
            # Generate insights
            insights = await self._generate_insights_for_products(filtered_df, products)
            
            self.logger.info("Analysis completed",
                           products_found=len(products),
                           processing_time=f"{time.time() - start_time:.2f}s")
            
            return products, insights
            
        except Exception as e:
            self.logger.error("Analysis failed",
                            error=str(e),
                            gender=gender,
                            age_range=f"{min_age}-{max_age}")
            return [], {}

    def _validate_parameters(self, gender: str, min_age: int, max_age: int, period: str) -> None:
        """Validate input parameters"""
        if gender not in self.VALID_GENDERS:
            raise ValueError(f"Invalid gender: {gender}. Must be one of {self.VALID_GENDERS}")
        
        if period not in self.VALID_PERIODS:
            raise ValueError(f"Invalid period: {period}. Must be one of {self.VALID_PERIODS}")
        
        if min_age < self.MIN_AGE or max_age > self.MAX_AGE:
            raise ValueError(f"Age range must be between {self.MIN_AGE} and {self.MAX_AGE}")
        
        if min_age >= max_age:
            raise ValueError("Minimum age must be less than maximum age")

    def _filter_data(self, df: pd.DataFrame, gender: str, min_age: int, max_age: int) -> pd.DataFrame:
        """Filter DataFrame by gender and age range"""
        # Create a copy to avoid SettingWithCopyWarning
        filtered_df = df.copy()
        return filtered_df[(filtered_df['age'] >= min_age) & 
                         (filtered_df['age'] <= max_age) & 
                         (filtered_df['gender'] == gender)]

    def _get_top_products(self, df: pd.DataFrame, top_n: int) -> List[str]:
        """Get top N products by sales fluctuation"""
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        df['fluctuation'] = df.groupby('product_name')['sales_amount'].transform(
            lambda x: abs(x.max() - x.min())
        )
        return df.sort_values('fluctuation', ascending=False).drop_duplicates('product_name').head(top_n)['product_name'].tolist()

    async def _generate_insights_for_products(
        self,
        df: pd.DataFrame,
        products: List[str]
    ) -> Dict[str, List[str]]:
        """Generate insights for each product using batch processing"""
        insights = {}
        
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Split products into batches
        for i in range(0, len(products), self.config.batch_size):
            batch = products[i:i + self.config.batch_size]
            tasks = []
            
            for product in batch:
                # Create a copy for each product to avoid SettingWithCopyWarning
                product_data = df[df['product_name'] == product].copy()
                max_sales_date = product_data.loc[product_data['sales_amount'].idxmax(), 'date']
                max_sales_gender = product_data.loc[product_data['sales_amount'].idxmax(), 'gender']
                dominant_age_group = self._get_dominant_age_group(df, product)
                segment = f"{max_sales_gender}, Age group {dominant_age_group}"
                max_sales = product_data['sales_amount'].max()
                min_sales = product_data['sales_amount'].min()
                change_rate = ((min_sales - max_sales) / max_sales) * 100
                
                task = self.generate_marketing_insights(
                    product,
                    max_sales_date.strftime('%Y-%m-%d'),
                    segment,
                    change_rate
                )
                tasks.append((product, task))
            
            # Process batch in parallel
            results = await asyncio.gather(*[task for _, task in tasks])
            
            # Process results
            for (product, _), (cause_analysis, strategies) in zip(tasks, results):
                insight_list = [
                    f"📊 Analysis period: {max_sales_date.strftime('%Y-%m-%d')}",
                    f"👥 Target customer: {segment}",
                    f"📈 Sales change: {change_rate:.1f}%",
                    "📌 Cause analysis:",
                    cause_analysis,
                    "💡 Strategy suggestions:"
                ]
                insight_list.extend(strategies)
                insights[product] = insight_list
            
            # Add delay between API calls to prevent rate limiting
            await asyncio.sleep(1)
            
        return insights

    def _get_age_group(self, age: int) -> str:
        """
        Groups age into 10-year intervals
        
        Args:
            age: Age value
            
        Returns:
            str: Age group (e.g., "20-29")
        """
        lower = (age // 10) * 10
        upper = lower + 9
        return f"{lower}-{upper}"

    def _get_dominant_age_group(self, df: pd.DataFrame, product: str) -> str:
        """
        Identifies the age group with the highest sales for a specific product
        
        Args:
            df: Sales data DataFrame
            product: Product name
            
        Returns:
            str: Age group with the highest sales
        """
        product_data = df[df['product_name'] == product].copy()
        product_data['age_group'] = product_data['age'].apply(self._get_age_group)
        
        # Calculate total sales by age group
        age_group_sales = product_data.groupby('age_group')['sales_amount'].sum()
        
        # Return the age group with the highest sales
        return age_group_sales.idxmax() 