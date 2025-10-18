"""
Snowflake data accessor implementation using direct connection.
"""

import pandas as pd
import snowflake.connector
from typing import Optional, Dict, Any
from .base import DataAccessor
from ..exceptions import DataAccessError


class SnowflakeAccessor(DataAccessor):
    """
    Snowflake data accessor implementation using direct connection.
    
    This implementation uses direct snowflake.connector for
    connecting to Snowflake and performing data operations.
    """
    
    def __init__(self, connection_config: Dict[str, Any] = None):
        """
        Initialize Snowflake accessor.
        
        Args:
            connection_config: Snowflake connection configuration (required)
        """
        if not connection_config:
            raise ValueError("Snowflake connection_config is required. Please provide credentials in data_config.yaml")
        
        self.connection_config = connection_config
        self.connection = None
    
    def read_data(self, table_name: str, columns: Optional[list] = None, 
                  where_clause: Optional[str] = None) -> pd.DataFrame:
        """
        Read data from Snowflake table.
        
        Args:
            table_name: Name of the table to read from
            columns: List of columns to select (None for all)
            where_clause: WHERE clause for filtering
            
        Returns:
            DataFrame containing the data
            
        Raises:
            DataAccessError: If data access fails
        """
        try:
            # Build the SQL query with UPPERCASE column names (no quotes needed)
            if columns:
                # Convert column names to UPPERCASE for consistency
                columns_upper = [col.upper() for col in columns]
                columns_str = ", ".join(columns_upper)
                query = f"SELECT {columns_str} FROM {table_name}"
            else:
                query = f"SELECT * FROM {table_name}"
            
            if where_clause:
                query += f" WHERE {where_clause}"
            
            # Use direct connection to read data
            conn = self._get_connection()
            df = pd.read_sql(query, conn)
            conn.close()
            return df
            
        except Exception as e:
            raise DataAccessError(f"Failed to read data from {table_name}: {str(e)}")
    
    def write_data(self, df: pd.DataFrame, table_name: str, 
                   if_exists: str = 'replace') -> None:
        """
        Write data to Snowflake table.
        
        Args:
            df: DataFrame to write
            table_name: Name of the table to write to
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
            
        Raises:
            DataAccessError: If data write fails
        """
        try:
            # Use direct connection to write data
            conn = self._get_connection()
            
            # PHASE 1 FIX: Convert columns to UPPERCASE and remove quotes
            df_snowflake = df.copy()
            
            # Convert column names to UPPERCASE
            df_snowflake.columns = [col.upper() for col in df_snowflake.columns]
            
            # Convert datetime columns to strings for Snowflake compatibility
            for col in df_snowflake.columns:
                if df_snowflake[col].dtype == 'datetime64[ns]':
                    df_snowflake[col] = df_snowflake[col].astype(str)
                elif 'datetime' in str(df_snowflake[col].dtype):
                    df_snowflake[col] = df_snowflake[col].astype(str)
            
            # OPTIMIZATION: Try Snowflake's native write_pandas method first
            try:
                from snowflake.connector.pandas_tools import write_pandas
                
                print(f"🚀 Using Snowflake write_pandas for {len(df_snowflake)} rows...")
                
                # Use write_pandas with UPPERCASE columns (no quotes needed)
                success, nchunks, nrows, _ = write_pandas(
                    conn=conn,
                    df=df_snowflake,
                    table_name=table_name,
                    auto_create_table=(if_exists == 'replace'),
                    overwrite=(if_exists == 'replace'),
                    quote_identifiers=False  # No quotes for UPPERCASE columns
                )
                
                conn.close()
                
                if not success:
                    raise DataAccessError(f"write_pandas failed to write {nrows} rows")
                
                print(f"✅ write_pandas completed ({nrows} rows in {nchunks} chunks)")
                return
                    
            except ImportError:
                # FALLBACK: If write_pandas is not available, use manual method
                print("⚠️ write_pandas not available, using manual method")
                pass
            except Exception as e:
                # FALLBACK: If write_pandas fails, use manual method
                print(f"⚠️ write_pandas failed ({str(e)}), falling back to manual insert")
                pass
            
            # MANUAL METHOD (fallback if write_pandas fails):
            print(f"🔄 Using manual batch insert for {len(df_snowflake)} rows...")
            
            cursor = conn.cursor()
            
            if if_exists == 'replace':
                # Drop table if exists and create new one
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                
                # Create table with UPPERCASE column names (no quotes)
                columns = []
                for col, dtype in df_snowflake.dtypes.items():
                    if dtype == 'object':
                        col_type = 'VARCHAR(16777216)'
                    elif dtype == 'int64':
                        col_type = 'BIGINT'
                    elif dtype == 'float64':
                        col_type = 'FLOAT'
                    elif dtype == 'bool':
                        col_type = 'BOOLEAN'
                    else:
                        col_type = 'VARCHAR(16777216)'
                    # Use UPPERCASE column names without quotes
                    columns.append(f'{col} {col_type}')
                
                create_table_sql = f"CREATE OR REPLACE TABLE {table_name} ({', '.join(columns)})"
                cursor.execute(create_table_sql)
            elif if_exists == 'append':
                # Check if table exists, create if it doesn't
                try:
                    # Try to query the table to see if it exists
                    cursor.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
                except Exception:
                    # Table doesn't exist, create it
                    columns = []
                    for col, dtype in df_snowflake.dtypes.items():
                        if dtype == 'object':
                            col_type = 'VARCHAR(16777216)'
                        elif dtype == 'int64':
                            col_type = 'BIGINT'
                        elif dtype == 'float64':
                            col_type = 'FLOAT'
                        elif dtype == 'bool':
                            col_type = 'BOOLEAN'
                        else:
                            col_type = 'VARCHAR(16777216)'
                        # Use UPPERCASE column names without quotes
                        columns.append(f'{col} {col_type}')
                    
                    create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"
                    cursor.execute(create_table_sql)
            
            # Insert in batches of 100 for much better performance
            batch_size = 100
            total_rows = len(df_snowflake)
            
            for i in range(0, total_rows, batch_size):
                batch_df = df_snowflake.iloc[i:i+batch_size]
                
                # Prepare batch data
                batch_values = []
                for _, row in batch_df.iterrows():
                    row_values = []
                    for val in row.values:
                        if pd.isna(val):
                            row_values.append(None)
                        else:
                            row_values.append(val)
                    batch_values.append(tuple(row_values))
                
                # Batch insert using executemany with UPPERCASE column names (no quotes)
                placeholders = ', '.join(['%s'] * len(df_snowflake.columns))
                columns_str = ', '.join(df_snowflake.columns)  # No quotes for UPPERCASE columns
                insert_sql = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
                
                cursor.executemany(insert_sql, batch_values)
                
                # Progress update for large datasets
                if total_rows > 500:
                    progress = ((i + batch_size) / total_rows) * 100
                    print(f"📊 Batch insert progress: {progress:.1f}% ({min(i + batch_size, total_rows)}/{total_rows} rows)")
            
            print(f"✅ Manual batch insert completed ({total_rows} rows)")
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            raise DataAccessError(f"Failed to write data to {table_name}: {str(e)}")
    
    def validate_connection(self) -> bool:
        """
        Validate Snowflake connection.
        
        Returns:
            True if connection is valid
            
        Raises:
            DataAccessError: If connection validation fails
        """
        try:
            conn = self._get_connection()
            conn.close()
            return True
        except Exception as e:
            raise DataAccessError(f"Failed to validate Snowflake connection: {str(e)}")
    
    def get_data_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get information about a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary containing table information
            
        Raises:
            DataAccessError: If data info retrieval fails
        """
        try:
            conn = self._get_connection()
            
            # Get basic table information
            query = f"""
            SELECT 
                COUNT(*) as row_count
            FROM {table_name}
            """
            
            df = pd.read_sql(query, conn)
            
            # Get column information (table names are already UPPERCASE in Snowflake)
            columns_query = f"""
            SELECT 
                COLUMN_NAME,
                DATA_TYPE,
                IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = '{table_name.upper()}'
            ORDER BY ORDINAL_POSITION
            """
            
            columns_df = pd.read_sql(columns_query, conn)
            conn.close()
            
            return {
                'table_name': table_name,
                'row_count': df.iloc[0]['ROW_COUNT'] if not df.empty else 0,
                'unique_rows': df.iloc[0]['ROW_COUNT'] if not df.empty else 0,  # Simplified for now
                'columns': columns_df.to_dict('records') if not columns_df.empty else []
            }
            
        except Exception as e:
            raise DataAccessError(f"Failed to get data info for {table_name}: {str(e)}")
    
    def has_file_changed(self, file_path: str) -> bool:
        """
        Check if a file has changed (not applicable for Snowflake).
        
        Args:
            file_path: Path to the file (not used for Snowflake)
            
        Returns:
            Always returns True for Snowflake (data is always fresh)
        """
        # For Snowflake, we assume data is always fresh
        return True
    
    def _get_connection(self):
        """
        Get a Snowflake connection using the configured credentials.
        
        Returns:
            Snowflake connection object
            
        Raises:
            DataAccessError: If connection fails
        """
        try:
            return snowflake.connector.connect(
                account=self.connection_config['account'],
                user=self.connection_config['user'],
                password=self.connection_config['password'],
                role=self.connection_config['role'],
                warehouse=self.connection_config['warehouse'],
                database=self.connection_config['database'],
                schema=self.connection_config['schema'],
                login_timeout=30,  # 30 second login timeout
                network_timeout=60,  # 60 second network timeout
                application='restock_forecasting'
            )
        except Exception as e:
            raise DataAccessError(f"Failed to connect to Snowflake: {str(e)}") 