import pandas as pd
import sqlite3
import os
import uuid
from pathlib import Path

class DBManager:
    def __init__(self, db_path="data.db"):
        self.db_path = db_path
        
    def dataframe_to_sql(self, df, table_name=None):
        """
        Converte um DataFrame para SQLite.
        Retorna o nome da tabela criada.
        """
        if table_name is None:
            # Gera um nome único para a tabela se não for fornecido
            table_name = f"data_{uuid.uuid4().hex[:8]}"
            
        with sqlite3.connect(self.db_path) as conn:
            # Salva o DataFrame no SQLite
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            
        return table_name
    
    def read_sql_table(self, table_name, columns=None, chunk_size=None):
        """
        Lê dados de uma tabela SQLite.
        Suporta leitura em chunks para grandes volumes de dados.
        """
        if chunk_size:
            with sqlite3.connect(self.db_path) as conn:
                if columns:
                    query = f"SELECT {', '.join(columns)} FROM {table_name}"
                else:
                    query = f"SELECT * FROM {table_name}"
                    
                return pd.read_sql_query(query, conn, chunksize=chunk_size)
        else:
            with sqlite3.connect(self.db_path) as conn:
                if columns:
                    return pd.read_sql_query(f"SELECT {', '.join(columns)} FROM {table_name}", conn)
                else:
                    return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    
    def get_table_info(self, table_name):
        """
        Retorna informações sobre a tabela.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            return cursor.fetchall()
    
    def table_exists(self, table_name):
        """
        Verifica se uma tabela existe no banco de dados.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
            return cursor.fetchone() is not None
            
    def drop_table(self, table_name):
        """
        Remove uma tabela do banco de dados.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.commit()
    
    def get_table_sample(self, table_name, n_rows=5):
        """
        Retorna uma amostra da tabela.
        """
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {n_rows}", conn)
    
    def get_column_names(self, table_name):
        """
        Retorna os nomes das colunas de uma tabela.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            return [row[1] for row in cursor.fetchall()]
            
    def execute_query(self, query, params=None):
        """
        Executa uma query SQL personalizada.
        """
        with sqlite3.connect(self.db_path) as conn:
            if params:
                return pd.read_sql_query(query, conn, params=params)
            return pd.read_sql_query(query, conn)