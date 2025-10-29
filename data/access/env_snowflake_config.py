"""
Environment-based Snowflake configuration for 5X workspace deployment.
This follows the same pattern as your other project (utils.py).
"""

import os
import snowflake.connector
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from typing import Optional, Dict, Any
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Use the same environment variable names as your successful project
SNOWFLAKE_DATABASE = os.getenv("FIVEX_SNOWFLAKE_DATABASE")
SNOWFLAKE_ACCOUNT = os.getenv("FIVEX_SNOWFLAKE_ACCOUNT")
SNOWFLAKE_ROLE = "ACCOUNTADMIN"  # Your role for restock project
SNOWFLAKE_WAREHOUSE = os.getenv("FIVEX_SNOWFLAKE_WAREHOUSE")
SNOWFLAKE_USER = os.getenv("FIVEX_SNOWFLAKE_USER")
SNOWFLAKE_PRIVATE_KEY_PATH = os.getenv("FIVEX_SNOWFLAKE_PRIVATE_KEY_FILE")
SNOWFLAKE_PRIVATE_KEY_PASSPHRASE = os.getenv("FIVEX_SNOWFLAKE_PRIVATE_KEY_FILE_PWD")
# Multi-schema configuration for read/write operations
SNOWFLAKE_READ_SCHEMA = os.getenv("FIVEX_SNOWFLAKE_READ_SCHEMA", "STAGE")
SNOWFLAKE_WRITE_SCHEMA = os.getenv("FIVEX_SNOWFLAKE_WRITE_SCHEMA", "TRANSFORMATION")
# Default schema for connection (used for initial connection)
SNOWFLAKE_SCHEMA = SNOWFLAKE_WRITE_SCHEMA

def get_raw_private_key():
    """Read the raw private key from file (from your other project pattern)"""
    try:
        if not SNOWFLAKE_PRIVATE_KEY_PATH:
            raise ValueError("Private key path is not set in environment variables")
        with open(SNOWFLAKE_PRIVATE_KEY_PATH, 'r') as key_file:
            return key_file.read()
    except Exception as e:
        logger.error(f"Error reading private key file: {e}")
        return None

def get_private_key():
    """Process private key for Snowflake authentication (from your other project pattern)"""
    try:
        raw_key = get_raw_private_key()
        if not raw_key:
            raise ValueError("Failed to read private key file")
        if not SNOWFLAKE_PRIVATE_KEY_PASSPHRASE:
            raise ValueError("Passphrase is required but not provided in environment variables")
        if not SNOWFLAKE_PRIVATE_KEY_PATH:
            raise ValueError("Private key path is not set in environment variables")
        
        with open(SNOWFLAKE_PRIVATE_KEY_PATH, 'rb') as key_file:
            key_bytes = key_file.read()
            try:
                p_key = serialization.load_pem_private_key(
                    key_bytes,
                    password=SNOWFLAKE_PRIVATE_KEY_PASSPHRASE.encode('utf-8'),
                    backend=default_backend()
                )
                private_key = p_key.private_bytes(
                    encoding=serialization.Encoding.DER,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                return private_key
            except Exception as e:
                logger.error(f"Error processing private key: {e}")
                return None
    except Exception as e:
        logger.error(f"Error reading private key: {e}")
        return None

def connect_to_snowflake_env():
    """Connect to Snowflake using environment variables (IDE-style authentication)"""
    try:
        # Validate required environment variables
        required_vars = {
            'FIVEX_SNOWFLAKE_ACCOUNT': SNOWFLAKE_ACCOUNT,
            'FIVEX_SNOWFLAKE_USER': SNOWFLAKE_USER,
            'FIVEX_SNOWFLAKE_WAREHOUSE': SNOWFLAKE_WAREHOUSE,
            'FIVEX_SNOWFLAKE_DATABASE': SNOWFLAKE_DATABASE,
            'FIVEX_SNOWFLAKE_PRIVATE_KEY_FILE': SNOWFLAKE_PRIVATE_KEY_PATH,
            'FIVEX_SNOWFLAKE_PRIVATE_KEY_FILE_PWD': SNOWFLAKE_PRIVATE_KEY_PASSPHRASE
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        private_key = get_private_key()
        if not private_key:
            raise Exception("Failed to load private key")
        
        conn = snowflake.connector.connect(
            account=SNOWFLAKE_ACCOUNT,
            user=SNOWFLAKE_USER,
            private_key=private_key,
            role=SNOWFLAKE_ROLE,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
            schema=SNOWFLAKE_SCHEMA,
            session_parameters={'QUERY_TAG': 'restock_forecasting_env'},
            connection_timeout=600,
            keep_alive_heartbeat=60
        )
        
        # Set context
        cursor = conn.cursor()
        cursor.execute(f"USE DATABASE {SNOWFLAKE_DATABASE}")
        cursor.execute(f"USE SCHEMA {SNOWFLAKE_SCHEMA}")
        cursor.close()
        
        return conn
    except Exception as e:
        logger.error(f"❌ Failed to connect to Snowflake: {e}")
        return None

def get_snowflake_config_from_env() -> Dict[str, Any]:
    """Get Snowflake configuration from environment variables"""
    return {
        'account': SNOWFLAKE_ACCOUNT,
        'user': SNOWFLAKE_USER,
        'role': SNOWFLAKE_ROLE,
        'warehouse': SNOWFLAKE_WAREHOUSE,
        'database': SNOWFLAKE_DATABASE,
        'schema': SNOWFLAKE_SCHEMA,
        'read_schema': SNOWFLAKE_READ_SCHEMA,
        'write_schema': SNOWFLAKE_WRITE_SCHEMA,
        'private_key_file': SNOWFLAKE_PRIVATE_KEY_PATH,
        'private_key_password': SNOWFLAKE_PRIVATE_KEY_PASSPHRASE
    }

def test_connection():
    """Test the Snowflake connection using environment variables"""
    try:
        conn = connect_to_snowflake_env()
        if conn is None:
            return False
        
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return result[0] == 1
    except Exception as e:
        logger.error(f"Connection test failed: {e}")
        return False