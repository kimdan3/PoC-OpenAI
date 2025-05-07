import pandas as pd
from datetime import datetime
import logging
from typing import Optional, List
import os
import time

logger = logging.getLogger(__name__)

class DataLoaderError(Exception):
    """Base exception for DataLoader."""
    pass

class DataLoader:
    """Service class for loading and preprocessing retail sales data."""
    
    # Constants
    REQUIRED_COLUMNS = ['date', 'product_name', 'age', 'gender', 'sales_amount']
    VALID_GENDERS = {'M', 'F'}
    MIN_AGE = 0
    MAX_AGE = 100
    CSV_SIZE_THRESHOLD = 10 * 1024 * 1024  # 10MB
    LOAD_TIME_THRESHOLD = 1.0  # 1 second
    
    @staticmethod
    def load_data(path: str = "data/data.csv") -> Optional[pd.DataFrame]:
        """
        Read data file and convert to DataFrame.
        Automatically converts large CSV files to Parquet format for better performance.
        
        Args:
            path: Path to the data file
            
        Returns:
            Optional[pd.DataFrame]: Loaded DataFrame or None if loading fails
            
        Raises:
            DataLoaderError: If data loading fails
        """
        try:
            if not os.path.exists(path):
                raise DataLoaderError(f"Data file not found: {path}")
            
            # Check if Parquet version exists
            parquet_path = path.replace('.csv', '.parquet')
            if os.path.exists(parquet_path):
                logger.info(f"Loading Parquet file: {parquet_path}")
                return pd.read_parquet(parquet_path)
            
            # Check CSV file size and loading time
            file_size = os.path.getsize(path)
            start_time = time.time()
            
            # Read CSV with explicit data types and date parsing
            df = pd.read_csv(
                path,
                dtype={
                    'customer_id': str,
                    'store_id': str,
                    'age': int,
                    'gender': str,
                    'age_group': str,
                    'product_name': str,
                    'category': str,
                    'units_sold': int,
                    'unit_price': float,
                    'discount_applied': float,
                    'discounted': bool,
                    'sales_amount': float
                },
                parse_dates=['date'],
                date_parser=lambda x: pd.to_datetime(x, format='%Y-%m-%d'),
                na_values=['', 'NA', 'NULL', 'null', 'NaN', 'nan'],
                encoding='utf-8'
            )
            
            load_time = time.time() - start_time
            
            if file_size >= DataLoader.CSV_SIZE_THRESHOLD or load_time >= DataLoader.LOAD_TIME_THRESHOLD:
                logger.info(f"Converting CSV to Parquet for better performance (size: {file_size/1024/1024:.1f}MB, load time: {load_time:.1f}s)")
                df.to_parquet(parquet_path)
                logger.info(f"Created Parquet file: {parquet_path}")
            
            if df.empty:
                raise DataLoaderError(f"Data file is empty: {path}")
                
            logger.info(f"Successfully loaded data from {path}")
            return df
            
        except pd.errors.EmptyDataError:
            logger.error(f"Data file is empty: {path}")
            raise DataLoaderError(f"Data file is empty: {path}")
            
        except Exception as e:
            logger.error(f"Error loading data from {path}: {str(e)}")
            raise DataLoaderError(f"Error loading data: {str(e)}")

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Preprocess the input DataFrame.
        
        Args:
            df: Input DataFrame to preprocess
            
        Returns:
            Optional[pd.DataFrame]: Preprocessed DataFrame or None if preprocessing fails
            
        Raises:
            DataLoaderError: If preprocessing fails
        """
        try:
            if df is None or df.empty:
                raise DataLoaderError("Input DataFrame is None or empty")

            # Create a copy to avoid SettingWithCopyWarning
            df = df.copy()

            # Check required columns
            missing_columns = [col for col in DataLoader.REQUIRED_COLUMNS if col not in df.columns]
            if missing_columns:
                raise DataLoaderError(f"Missing required columns: {missing_columns}")

            # Keep only required columns
            df = df[DataLoader.REQUIRED_COLUMNS].copy()
            
            # Clean date column
            df.loc[:, 'date'] = df['date'].astype(str).str.strip()
            
            # Convert date column to datetime with enhanced error handling
            try:
                # First attempt: Try standard datetime conversion with explicit format
                df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
            except Exception as e:
                logger.warning(f"Explicit format date conversion failed: {str(e)}")
                try:
                    # Second attempt: Try standard datetime conversion
                    df.loc[:, 'date'] = pd.to_datetime(df['date'])
                except Exception as e:
                    logger.warning(f"Standard date conversion failed: {str(e)}")
                    try:
                        # Third attempt: Try with coerce=True to handle invalid dates
                        df.loc[:, 'date'] = pd.to_datetime(df['date'], errors='coerce')
                        # Remove rows with invalid dates
                        invalid_dates = df['date'].isna().sum()
                        if invalid_dates > 0:
                            logger.warning(f"Removed {invalid_dates} rows with invalid dates")
                            df = df.dropna(subset=['date'])
                    except Exception as e:
                        raise DataLoaderError(f"Failed to convert date column to datetime: {str(e)}")
            
            # Remove rows with missing values
            missing_before = df.isnull().sum().sum()
            df = df.dropna()
            missing_after = df.isnull().sum().sum()
            if missing_before > 0:
                logger.warning(f"Removed {missing_before - missing_after} rows with missing values")
            
            # Clean negative sales
            negative_sales = (df['sales_amount'] < 0).sum()
            if negative_sales > 0:
                logger.warning(f"Found {negative_sales} rows with negative sales amounts")
                df.loc[df['sales_amount'] < 0, 'sales_amount'] = 0
            
            # Validate data types
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                raise DataLoaderError("Failed to convert date column to datetime")
                
            if not pd.api.types.is_numeric_dtype(df['sales_amount']):
                raise DataLoaderError("Sales amount column is not numeric")
                
            if not pd.api.types.is_numeric_dtype(df['age']):
                raise DataLoaderError("Age column is not numeric")
                
            # Validate gender values
            invalid_genders = set(df['gender'].unique()) - DataLoader.VALID_GENDERS
            if invalid_genders:
                raise DataLoaderError(f"Invalid gender values found: {invalid_genders}")
                
            # Validate age range
            if (df['age'] < DataLoader.MIN_AGE).any() or (df['age'] > DataLoader.MAX_AGE).any():
                raise DataLoaderError(f"Age values must be between {DataLoader.MIN_AGE} and {DataLoader.MAX_AGE}")
            
            logger.info("Data preprocessing completed successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise DataLoaderError(f"Data preprocessing failed: {str(e)}")

    @staticmethod
    def get_product_categories(df: pd.DataFrame) -> List[str]:
        """
        Get unique product categories from the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List[str]: List of unique product categories
            
        Raises:
            DataLoaderError: If product categories cannot be retrieved
        """
        try:
            if df is None or df.empty:
                raise DataLoaderError("Input DataFrame is None or empty")
                
            if 'product_name' not in df.columns:
                raise DataLoaderError("Product name column not found in DataFrame")
                
            categories = sorted(df["product_name"].unique().tolist())
            logger.info(f"Found {len(categories)} unique product categories")
            return categories
            
        except Exception as e:
            logger.error(f"Error getting product categories: {str(e)}")
            raise DataLoaderError(f"Failed to get product categories: {str(e)}")

    @staticmethod
    def get_customer_segments(df: pd.DataFrame) -> List[str]:
        """
        Get unique customer segments from the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List[str]: List of unique customer segments
            
        Raises:
            DataLoaderError: If customer segments cannot be retrieved
        """
        try:
            if df is None or df.empty:
                raise DataLoaderError("Input DataFrame is None or empty")
                
            if 'age' not in df.columns or 'gender' not in df.columns:
                raise DataLoaderError("Age or gender columns not found in DataFrame")
                
            # Create age groups
            df['age_group'] = (df['age'] // 10) * 10
            segments = sorted(df.apply(lambda x: f"{x['age_group']}s_{x['gender']}", axis=1).unique().tolist())
            
            logger.info(f"Found {len(segments)} unique customer segments")
            return segments
            
        except Exception as e:
            logger.error(f"Error getting customer segments: {str(e)}")
            raise DataLoaderError(f"Failed to get customer segments: {str(e)}") 