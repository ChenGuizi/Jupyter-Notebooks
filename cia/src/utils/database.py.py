from urllib.parse import quote
from sqlalchemy import create_engine
import pandas as pd
import time
import logging

def create_db_engine(user, password, host, port, db):
    """
    Creates a SQLAlchemy engine for connecting to a MariaDB database.

    Parameters:
    user (str): Database username.
    password (str): Database password.
    host (str): Database host address.
    port (int): Port number for the database connection.
    db (str): Name of the database.

    Returns:
    sqlalchemy.engine.base.Engine: SQLAlchemy engine for database connection.
    """
    connection_string = f"mysql+pymysql://{user}:{quote(password)}@{host}:{port}/{db}"
    db_engine = create_engine(connection_string)
    return db_engine

def call_db(engine, sql_query, df_for_upload=None, table_name=None, action="read", retries=4, delays=[5, 10, 30, 60]):
    """
    Function to interact with DB with retry mechanism

    Parameters:
    db_engine (Engine): SQLAlchemy engine
    sql_query (str): SQL query for read db.
    df_for_upload (DataFrame): Data for upload to DB.
    table_name (str): DB table name for upload.
    action (str): Whether to read or upload to DB.
    retries (int): Number of times to retry
    delays (list[int]): Seconds delay after each retry.

    Returns:
    df (pd.DataFrame): Dataframe if action is read
    None: if action is upload
    """
    for i in range(retries + 1):
        try:
            if action == "read":
                df = pd.read_sql(sql_query, engine)
                logging.info("Read from DB Success!")
                db_engine.dispose()
                return df
            elif action == "upload":
                df_for_upload.to_sql(table_name, con=db_engine, if_exists="append", index=False)
                logging.info("Upload to DB (table_name) Success!")
                db_engine.dispose()
                return
        except Exception as e:
            db_engine.dispose()
            if i < retries:
                logging.error(f"DB call failed for {action} DB after {i + 1} times for {action} DB, will try again in {delays[i]} seconds...")
                time.sleep(delays[i])
            else:
                logging.error(f"Call DB failed for {action} DB after {retries} attempts, unable to complete. Error: {e}")
                raise Exception(f"Call DB failed for {action} DB after {retries} attempts: {e}")