"""
Environment-based Snowflake data accessor for 5X workspace deployment.
This uses environment variables instead of config files.
"""

import pandas as pd
import snowflake.connector
from typing import Optional, Dict, Any
from .base import DataAccessor
from ..exceptions import DataAccessError
from .env_snowflake_config import connect_to_snowflake_env, get_snowflake_config_from_env


class EnvSnowflakeAccessor(DataAccessor):
    """
    Environment-based Snowflake data accessor for 5X workspace deployment.
    
    This implementation uses environment variables for configuration
    and private key authentication (IDE-style).
    """
    
    def __init__(self, use_env_vars: bool = True):
        """
        Initialize Environment-based Snowflake accessor.
        
        Args:
            use_env_vars: If True, use environment variables; if False, use config file
        """
        self.use_env_vars = use_env_vars
        self.connection = None
        
        if use_env_vars:
            # Validate environment variables are available
            self.connection_config = get_snowflake_config_from_env()
            missing_vars = [k for k, v in self.connection_config.items() if not v]
            if missing_vars:
                raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        else:
            raise ValueError("This accessor requires use_env_vars=True for 5X workspace deployment")
    
    def read_data(self, table_name: str, columns: Optional[list] = None, 
                  where_clause: Optional[str] = None, schema: Optional[str] = None) -> pd.DataFrame:
        """
        Read data from Snowflake table using environment variables.
        
        Args:
            table_name: Name of the table to read from
            columns: List of columns to select (None for all)
            where_clause: WHERE clause for filtering
            schema: Schema name (if None, uses read_schema from config)
            
        Returns:
            DataFrame containing the data
            
        Raises:
            DataAccessError: If data access fails
        """
        try:
            # Use specified schema or default to read_schema
            if schema is None:
                schema = self.connection_config.get('read_schema', 'STAGE')
            
            # Build the SQL query with UPPERCASE column names (no quotes needed)
            if columns:
                # Convert column names to uppercase for Snowflake compatibility
                columns_upper = [col.upper() for col in columns]
                columns_str = ", ".join(columns_upper)
            else:
                columns_str = "*"
            
            query = f"SELECT {columns_str} FROM {schema.upper()}.{table_name.upper()}"
            
            if where_clause:
                query += f" WHERE {where_clause}"
            
            return self._execute_query(query)
            
        except Exception as e:
            raise DataAccessError(f"Failed to read data from {schema}.{table_name}: {str(e)}")
    
    def write_data(self, df: pd.DataFrame, table_name: str, 
                   if_exists: str = "append", index: bool = False, schema: Optional[str] = None) -> None:
        """
        Write DataFrame to Snowflake table using environment variables.
        
        Args:
            df: DataFrame to write
            table_name: Name of the target table
            if_exists: What to do if table exists ('append', 'replace', 'fail')
            index: Whether to write DataFrame index as a column
            schema: Schema name (if None, uses write_schema from config)
            
        Raises:
            DataAccessError: If data write fails
        """
        try:
            conn = self._get_connection()
            
            # Use specified schema or default to write_schema
            if schema is None:
                schema = self.connection_config.get('write_schema', 'TRANSFORMATION')
            
            # Convert column names to uppercase for Snowflake compatibility
            df_normalized = df.copy()
            df_normalized.columns = [col.upper() for col in df_normalized.columns]
            
            # Use pandas to_sql with Snowflake connector
            from snowflake.connector.pandas_tools import write_pandas
            
            # Set the schema context before writing
            cursor = conn.cursor()
            cursor.execute(f"USE SCHEMA {schema.upper()}")
            cursor.close()
            
            success, nchunks, nrows, _ = write_pandas(
                conn,
                df_normalized,
                table_name=table_name.upper(),
                auto_create_table=True,
                overwrite=(if_exists == "replace")
            )
            
            if not success:
                raise DataAccessError(f"Failed to write data to {schema}.{table_name}")
                
        except Exception as e:
            raise DataAccessError(f"Failed to write data to {schema}.{table_name}: {str(e)}")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a custom SQL query using environment variables.
        
        Args:
            query: SQL query to execute
            
        Returns:
            DataFrame containing query results
            
        Raises:
            DataAccessError: If query execution fails
        """
        return self._execute_query(query)
    
    def _execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame.
        
        Args:
            query: SQL query to execute
            
        Returns:
            DataFrame containing query results
        """
        conn = None
        try:
            conn = self._get_connection()
            return pd.read_sql(query, conn)
        except Exception as e:
            raise DataAccessError(f"Query execution failed: {str(e)}")
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
    
    def _get_connection(self):
        """
        Get a Snowflake connection using environment variables.
        
        Returns:
            Snowflake connection object
            
        Raises:
            DataAccessError: If connection fails
        """
        try:
            if self.use_env_vars:
                return connect_to_snowflake_env()
            else:
                raise ValueError("This accessor requires environment variables for 5X workspace")
        except Exception as e:
            raise DataAccessError(f"Failed to connect to Snowflake: {str(e)}")
    
    def has_file_changed(self, file_path: str) -> bool:
        """
        Check if a file has changed (not applicable for Snowflake).
        
        Args:
            file_path: Path to the file (not used for Snowflake)
            
        Returns:
            Always True for Snowflake (data is always fresh)
        """
        # For Snowflake, we assume data is always fresh
        return True
    
    def validate_connection(self) -> bool:
        """
        Validate that the Snowflake connection is working.
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            conn = self._get_connection()
            if conn is None:
                return False
            
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            return result[0] == 1
        except Exception:
            return False
    
    def get_data_info(self, path: str) -> Dict[str, Any]:
        """
        Get information about the data source.
        
        Args:
            path: Path to the data source (table name for Snowflake)
            
        Returns:
            Dictionary containing data information
        """
        try:
            conn = self._get_connection()
            if conn is None:
                return {"error": "No connection available"}
            
            # Get table information
            query = f"SELECT COUNT(*) as row_count FROM {path.upper()}"
            result = self._execute_query(query)
            
            if not result.empty:
                return {
                    "table_name": path.upper(),
                    "row_count": result.iloc[0]['ROW_COUNT'],
                    "source": "snowflake"
                }
            else:
                return {"error": "No data found"}
                
        except Exception as e:
            return {"error": str(e)}
