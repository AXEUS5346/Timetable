import os
import pandas as pd
import logging
from typing import Optional, List, Dict, Any, Union, Tuple
import shutil
from datetime import datetime

# Configure logging
logger = logging.getLogger("csv_database")
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)


class CSVDatabase:
    """
    A simple CSV-powered database system for the timetable management application.
    
    This class provides CRUD operations for CSV files, with basic data validation,
    backup functionality, and transaction-like operations.
    """
    
    def __init__(self, db_dir: str = "database"):
        """
        Initialize the CSV database.
        
        Args:
            db_dir: Directory where CSV files will be stored
        """
        self.db_dir = db_dir
        self._ensure_db_dir_exists()
        self.backup_dir = os.path.join(db_dir, "backups")
        self._ensure_backup_dir_exists()
        logger.info(f"CSV Database initialized at {os.path.abspath(db_dir)}")
    
    def _ensure_db_dir_exists(self) -> None:
        """Create the database directory if it doesn't exist."""
        if not os.path.exists(self.db_dir):
            os.makedirs(self.db_dir)
            logger.info(f"Created database directory: {self.db_dir}")
    
    def _ensure_backup_dir_exists(self) -> None:
        """Create the backup directory if it doesn't exist."""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
            logger.info(f"Created backup directory: {self.backup_dir}")
    
    def _get_table_path(self, table_name: str) -> str:
        """Get the file path for a table."""
        return os.path.join(self.db_dir, f"{table_name}.csv")
    
    def list_tables(self) -> List[str]:
        """
        List all tables (CSV files) in the database.
        
        Returns:
            List of table names without the .csv extension
        """
        tables = []
        for file in os.listdir(self.db_dir):
            if file.endswith(".csv") and os.path.isfile(os.path.join(self.db_dir, file)):
                tables.append(file[:-4])  # Remove .csv extension
        return tables
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if the table exists, False otherwise
        """
        return os.path.exists(self._get_table_path(table_name))
    
    def create_table(self, table_name: str, data: Optional[pd.DataFrame] = None,
                    columns: Optional[List[str]] = None) -> bool:
        """
        Create a new table. If data is provided, it will be used to populate the table.
        If columns are provided but no data, an empty table with those columns will be created.
        
        Args:
            table_name: Name of the table to create
            data: Optional DataFrame containing the data
            columns: Optional list of column names for an empty table
            
        Returns:
            True if successful, False otherwise
        """
        table_path = self._get_table_path(table_name)
        
        # Check if table already exists
        if os.path.exists(table_path):
            logger.warning(f"Table '{table_name}' already exists")
            return False
        
        try:
            if data is not None:
                # Save the DataFrame as a CSV file
                data.to_csv(table_path, index=False)
                logger.info(f"Created table '{table_name}' with {len(data)} rows and {len(data.columns)} columns")
            elif columns is not None:
                # Create an empty DataFrame with the specified columns
                empty_df = pd.DataFrame(columns=columns)
                empty_df.to_csv(table_path, index=False)
                logger.info(f"Created empty table '{table_name}' with columns: {columns}")
            else:
                # Create an empty table
                pd.DataFrame().to_csv(table_path, index=False)
                logger.info(f"Created empty table '{table_name}'")
            
            return True
        except Exception as e:
            logger.error(f"Failed to create table '{table_name}': {str(e)}")
            return False
    
    def read_table(self, table_name: str) -> Optional[pd.DataFrame]:
        """
        Read a table from the database.
        
        Args:
            table_name: Name of the table to read
            
        Returns:
            DataFrame containing the table data or None if the table doesn't exist
        """
        table_path = self._get_table_path(table_name)
        
        if not os.path.exists(table_path):
            logger.warning(f"Table '{table_name}' does not exist")
            return None
        
        try:
            df = pd.read_csv(table_path)
            logger.info(f"Read table '{table_name}' with {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to read table '{table_name}': {str(e)}")
            return None
    
    def write_table(self, table_name: str, data: pd.DataFrame, overwrite: bool = True) -> bool:
        """
        Write data to a table. If the table already exists and overwrite is True,
        the table will be replaced. Otherwise, the operation will fail.
        
        Args:
            table_name: Name of the table to write
            data: DataFrame containing the data
            overwrite: Whether to overwrite the table if it already exists
            
        Returns:
            True if successful, False otherwise
        """
        table_path = self._get_table_path(table_name)
        
        # Check if table already exists and we're not overwriting
        if os.path.exists(table_path) and not overwrite:
            logger.warning(f"Table '{table_name}' already exists and overwrite is False")
            return False
        
        try:
            # Backup existing table before overwriting
            if os.path.exists(table_path):
                self._backup_table(table_name)
            
            # Write the data
            data.to_csv(table_path, index=False)
            logger.info(f"Wrote {len(data)} rows to table '{table_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to write to table '{table_name}': {str(e)}")
            return False
    
    def append_to_table(self, table_name: str, data: pd.DataFrame) -> bool:
        """
        Append data to an existing table.
        
        Args:
            table_name: Name of the table to append to
            data: DataFrame containing the data to append
            
        Returns:
            True if successful, False otherwise
        """
        table_path = self._get_table_path(table_name)
        
        if not os.path.exists(table_path):
            logger.warning(f"Table '{table_name}' does not exist")
            return False
        
        try:
            # Read existing data
            existing_df = pd.read_csv(table_path)
            
            # Check if columns match
            if set(existing_df.columns) != set(data.columns):
                logger.error(f"Column mismatch: existing columns {existing_df.columns} vs. "
                           f"new columns {data.columns}")
                return False
            
            # Backup existing table
            self._backup_table(table_name)
            
            # Append data
            combined_df = pd.concat([existing_df, data], ignore_index=True)
            combined_df.to_csv(table_path, index=False)
            
            logger.info(f"Appended {len(data)} rows to table '{table_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to append to table '{table_name}': {str(e)}")
            return False
    
    def update_table(self, table_name: str, update_fn: callable) -> bool:
        """
        Update a table using a function that modifies the DataFrame.
        
        Args:
            table_name: Name of the table to update
            update_fn: Function that takes a DataFrame and returns a modified DataFrame
            
        Returns:
            True if successful, False otherwise
        """
        table_path = self._get_table_path(table_name)
        
        if not os.path.exists(table_path):
            logger.warning(f"Table '{table_name}' does not exist")
            return False
        
        try:
            # Read existing data
            df = pd.read_csv(table_path)
            
            # Backup existing table
            self._backup_table(table_name)
            
            # Apply update function
            updated_df = update_fn(df)
            
            # Check if the function returned a DataFrame
            if not isinstance(updated_df, pd.DataFrame):
                logger.error(f"Update function did not return a DataFrame")
                return False
            
            # Write updated data
            updated_df.to_csv(table_path, index=False)
            
            logger.info(f"Updated table '{table_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to update table '{table_name}': {str(e)}")
            return False
    
    def delete_table(self, table_name: str) -> bool:
        """
        Delete a table from the database.
        
        Args:
            table_name: Name of the table to delete
            
        Returns:
            True if successful, False otherwise
        """
        table_path = self._get_table_path(table_name)
        
        if not os.path.exists(table_path):
            logger.warning(f"Table '{table_name}' does not exist")
            return False
        
        try:
            # Backup before deletion
            self._backup_table(table_name)
            
            # Delete the file
            os.remove(table_path)
            
            logger.info(f"Deleted table '{table_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete table '{table_name}': {str(e)}")
            return False
    
    def query_table(self, table_name: str, query_fn: callable) -> Optional[pd.DataFrame]:
        """
        Query a table using a function that filters the DataFrame.
        
        Args:
            table_name: Name of the table to query
            query_fn: Function that takes a DataFrame and returns a filtered DataFrame
            
        Returns:
            Filtered DataFrame or None if the table doesn't exist or an error occurs
        """
        df = self.read_table(table_name)
        
        if df is None:
            return None
        
        try:
            result = query_fn(df)
            logger.info(f"Query on table '{table_name}' returned {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Failed to query table '{table_name}': {str(e)}")
            return None
    
    def _backup_table(self, table_name: str) -> bool:
        """
        Create a backup of a table.
        
        Args:
            table_name: Name of the table to backup
            
        Returns:
            True if successful, False otherwise
        """
        table_path = self._get_table_path(table_name)
        
        if not os.path.exists(table_path):
            logger.warning(f"Cannot backup table '{table_name}' as it does not exist")
            return False
        
        try:
            # Create a timestamp for the backup file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_dir, f"{table_name}_{timestamp}.csv")
            
            # Copy the file
            shutil.copy2(table_path, backup_path)
            
            logger.info(f"Created backup of table '{table_name}' at {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to backup table '{table_name}': {str(e)}")
            return False
    
    def restore_table(self, table_name: str, backup_timestamp: Optional[str] = None) -> bool:
        """
        Restore a table from a backup.
        
        Args:
            table_name: Name of the table to restore
            backup_timestamp: Timestamp of the backup to restore (format: YYYYMMDD_HHMMSS)
                              If None, the most recent backup will be used
            
        Returns:
            True if successful, False otherwise
        """
        # Find backup files for this table
        backup_files = []
        for file in os.listdir(self.backup_dir):
            if file.startswith(f"{table_name}_") and file.endswith(".csv"):
                backup_files.append(file)
        
        if not backup_files:
            logger.warning(f"No backups found for table '{table_name}'")
            return False
        
        try:
            if backup_timestamp:
                # Find specific backup
                backup_file = f"{table_name}_{backup_timestamp}.csv"
                if backup_file not in backup_files:
                    logger.warning(f"Backup with timestamp '{backup_timestamp}' not found for table '{table_name}'")
                    return False
            else:
                # Sort backups by timestamp (newest first)
                backup_files.sort(reverse=True)
                backup_file = backup_files[0]
            
            backup_path = os.path.join(self.backup_dir, backup_file)
            table_path = self._get_table_path(table_name)
            
            # Backup current version before restoring
            if os.path.exists(table_path):
                self._backup_table(table_name)
            
            # Copy backup to table location
            shutil.copy2(backup_path, table_path)
            
            logger.info(f"Restored table '{table_name}' from backup {backup_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore table '{table_name}': {str(e)}")
            return False
    
    def list_backups(self, table_name: Optional[str] = None) -> Dict[str, List[str]]:
        """
        List all backups, optionally filtered by table name.
        
        Args:
            table_name: Optional name of the table to filter backups
            
        Returns:
            Dictionary mapping table names to lists of backup timestamps
        """
        backups = {}
        
        for file in os.listdir(self.backup_dir):
            if file.endswith(".csv"):
                parts = file.split("_")
                if len(parts) >= 2:
                    table = parts[0]
                    timestamp = "_".join(parts[1:]).rstrip(".csv")
                    
                    if table_name and table != table_name:
                        continue
                    
                    if table not in backups:
                        backups[table] = []
                    
                    backups[table].append(timestamp)
        
        # Sort timestamps in descending order (newest first)
        for table in backups:
            backups[table].sort(reverse=True)
        
        return backups
    
    def vacuum(self, keep_latest: int = 5) -> int:
        """
        Delete old backups, keeping only the specified number of latest backups per table.
        
        Args:
            keep_latest: Number of latest backups to keep for each table
            
        Returns:
            Number of backup files deleted
        """
        backups = self.list_backups()
        deleted_count = 0
        
        for table, timestamps in backups.items():
            if len(timestamps) > keep_latest:
                # Keep only the newest 'keep_latest' backups
                to_delete = timestamps[keep_latest:]
                
                for timestamp in to_delete:
                    backup_file = f"{table}_{timestamp}.csv"
                    backup_path = os.path.join(self.backup_dir, backup_file)
                    
                    try:
                        os.remove(backup_path)
                        deleted_count += 1
                        logger.info(f"Deleted old backup: {backup_file}")
                    except Exception as e:
                        logger.error(f"Failed to delete backup {backup_file}: {str(e)}")
        
        logger.info(f"Vacuum completed, deleted {deleted_count} old backup files")
        return deleted_count
    
    def get_table_metadata(self, table_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary containing metadata or None if the table doesn't exist
        """
        table_path = self._get_table_path(table_name)
        
        if not os.path.exists(table_path):
            logger.warning(f"Table '{table_name}' does not exist")
            return None
        
        try:
            df = pd.read_csv(table_path)
            
            # Get file stats
            file_stats = os.stat(table_path)
            
            metadata = {
                "name": table_name,
                "rows": len(df),
                "columns": list(df.columns),
                "column_count": len(df.columns),
                "dtypes": {col: str(df[col].dtype) for col in df.columns},
                "file_size_bytes": file_stats.st_size,
                "file_size_kb": file_stats.st_size / 1024,
                "file_size_mb": file_stats.st_size / (1024 * 1024),
                "created_time": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                "null_counts": df.isnull().sum().to_dict(),
                "backup_count": len(self.list_backups(table_name).get(table_name, [])),
            }
            
            return metadata
        except Exception as e:
            logger.error(f"Failed to get metadata for table '{table_name}': {str(e)}")
            return None
    
    def export_table(self, table_name: str, export_path: str, format: str = "csv") -> bool:
        """
        Export a table to a file in the specified format.
        
        Args:
            table_name: Name of the table to export
            export_path: Path where the exported file will be saved
            format: Export format (csv, json, excel, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        df = self.read_table(table_name)
        
        if df is None:
            return False
        
        try:
            format = format.lower()
            
            if format == "csv":
                df.to_csv(export_path, index=False)
            elif format == "json":
                df.to_json(export_path, orient="records", lines=True)
            elif format in ["excel", "xlsx", "xls"]:
                df.to_excel(export_path, index=False)
            elif format == "html":
                df.to_html(export_path, index=False)
            elif format == "parquet":
                df.to_parquet(export_path, index=False)
            elif format == "feather":
                df.to_feather(export_path)
            else:
                logger.error(f"Unsupported export format: {format}")
                return False
            
            logger.info(f"Exported table '{table_name}' to {export_path} in {format} format")
            return True
        except Exception as e:
            logger.error(f"Failed to export table '{table_name}': {str(e)}")
            return False
    
    def import_table(self, table_name: str, import_path: str, format: str = "csv", 
                    overwrite: bool = False) -> bool:
        """
        Import a table from a file.
        
        Args:
            table_name: Name of the table to create
            import_path: Path to the file to import
            format: File format (csv, json, excel, etc.)
            overwrite: Whether to overwrite an existing table
            
        Returns:
            True if successful, False otherwise
        """
        if self.table_exists(table_name) and not overwrite:
            logger.warning(f"Table '{table_name}' already exists and overwrite is False")
            return False
        
        try:
            format = format.lower()
            
            if format == "csv":
                df = pd.read_csv(import_path)
            elif format == "json":
                df = pd.read_json(import_path, orient="records", lines=True)
            elif format in ["excel", "xlsx", "xls"]:
                df = pd.read_excel(import_path)
            elif format == "parquet":
                df = pd.read_parquet(import_path)
            elif format == "feather":
                df = pd.read_feather(import_path)
            else:
                logger.error(f"Unsupported import format: {format}")
                return False
            
            # Write to the database
            result = self.write_table(table_name, df, overwrite=overwrite)
            
            if result:
                logger.info(f"Imported table '{table_name}' from {import_path} in {format} format")
            
            return result
        except Exception as e:
            logger.error(f"Failed to import table '{table_name}': {str(e)}")
            return False
